from pathlib import Path
import math
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Config
# ============================================================

TSV_PATH = Path("data/weather_with_images.tsv")

# Debe coincidir con el multimodal
W = 8

BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3
TRAIN_RATIO = 0.9

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Si te da problemas con LSTM/CUDA/cuDNN, deja esto en False.
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False

SEED = 42
NUM_WORKERS = 2

MODEL_OUT = "best_lstm_tabular_same_samples_as_multimodal_no_autoreg.pt"
PREDICTIONS_OUT = "eval_predictions_lstm_tabular_same_samples_as_multimodal_no_autoreg.tsv"
PLOT_OUT = "eval_timeline_lstm_tabular_same_samples_as_multimodal_no_autoreg.png"

# Early stopping
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MIN_DELTA = 1e-4

# Para MAPE: ignora valores reales bajos.
# Con PV, MAPE con producción muy baja exagera muchísimo errores pequeños.
MAPE_MIN_Y = 1000.0

# Capacidad de referencia para nMAE/nRMSE.
# Se calcula en main() como percentil 99.5 de production:
# CAPACITY = np.percentile(df["production"].dropna(), 99.5)
CAPACITY = None
CAPACITY_PERCENTILE = 99.5

# Debe coincidir con el multimodal
MIN_PRODUCTION_FOR_SAMPLE = 0.0
MAX_IMAGE_AGE_MINUTES = 120

EPS = 1e-8


# IMPORTANTE:
# production NO se usa como input.
# image_path NO se usa como input.
# image_path SOLO se usa para construir exactamente los mismos samples que el multimodal.
IGNORE_COLUMNS = [
    "timestamp",
    "image_path",
    "production",
]

PREFERRED_INPUT_COLUMNS = [
    "humidity",
    "irradiation",
    "precipitation",
    "temperature",
    "winddirection",
    "windspeed",
]


# ============================================================
# Reproducibilidad
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Utilidades imagen
# ============================================================

def image_path_exists(p):
    """
    Criterio rápido:
    hay imagen si image_path no está vacío.

    NO carga la imagen.
    NO verifica que exista en disco.
    """
    if not isinstance(p, str):
        return False

    return p.strip() != ""


# ============================================================
# Escalado MinMax
# ============================================================

def fit_minmax(df, input_cols):
    x = df[input_cols].values.astype(np.float32)
    y = df["production"].values.astype(np.float32)

    x_min = x.min(axis=0)
    x_max = x.max(axis=0)

    x_range = x_max - x_min
    x_range[x_range < EPS] = 1.0

    y_min = float(y.min())
    y_max = float(y.max())
    y_range = y_max - y_min

    if y_range < EPS:
        y_range = 1.0

    return x_min, x_range, y_min, y_range


def minmax_x(x, x_min, x_range):
    return (x - x_min) / x_range


def minmax_y(y, y_min, y_range):
    return (y - y_min) / y_range


def inverse_minmax_y(y_norm, y_min, y_range):
    return y_norm * y_range + y_min


# ============================================================
# Correlaciones
# ============================================================

def print_correlations(df, input_cols):
    print()
    print("Correlación Pearson con production:")

    corr_rows = []

    for col in input_cols:
        tmp = df[[col, "production"]].dropna()

        if len(tmp) < 2:
            corr = np.nan
        else:
            corr = tmp[col].corr(tmp["production"], method="pearson")

        corr_rows.append((col, corr))

    corr_rows = sorted(
        corr_rows,
        key=lambda x: abs(x[1]) if np.isfinite(x[1]) else -1,
        reverse=True,
    )

    for col, corr in corr_rows:
        print(f"  {col:15s} corr={corr: .4f}")

    print()
    print("Correlación Spearman con production:")

    spear_rows = []

    for col in input_cols:
        tmp = df[[col, "production"]].dropna()

        if len(tmp) < 2:
            corr = np.nan
        else:
            corr = tmp[col].corr(tmp["production"], method="spearman")

        spear_rows.append((col, corr))

    spear_rows = sorted(
        spear_rows,
        key=lambda x: abs(x[1]) if np.isfinite(x[1]) else -1,
        reverse=True,
    )

    for col, corr in spear_rows:
        print(f"  {col:15s} corr={corr: .4f}")


# ============================================================
# Construcción de samples
# MISMA QUE MULTIMODAL
# ============================================================

def row_has_valid_values(df, row_idx, input_cols):
    x_values = df.loc[row_idx, input_cols].values.astype(np.float32)
    y_value = df.loc[row_idx, "production"]

    if not np.isfinite(x_values).all():
        return False

    if not np.isfinite(float(y_value)):
        return False

    return True


def build_multimodal_samples(
    df,
    input_cols,
    window,
    max_image_age_minutes,
):
    """
    Versión rápida.

    Misma idea que la anterior:
      - label_idx es el instante a predecir
      - tab_indices son las W filas anteriores
      - image_indices son las W imágenes anteriores, nunca la imagen actual

    Diferencia:
      - no reescanea todas las imágenes anteriores en cada fila
      - mantiene solo las imágenes dentro de MAX_IMAGE_AGE_MINUTES
    """
    from collections import deque
    import time

    print()
    print("Construyendo samples multimodales/tabulares rápidos...", flush=True)

    samples = []
    previous_images = deque()

    n = len(df)
    max_age_ns = pd.Timedelta(minutes=max_image_age_minutes).value

    # Timestamps a int64 nanosegundos para comparar rápido
    timestamps_ns = df["timestamp"].astype("int64").to_numpy()

    # image_path no vacío
    has_image = (
        df["image_path"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .to_numpy()
    )

    # Valores tabulares y producción como numpy
    x_values = df[input_cols].values.astype(np.float32)
    y_values = df["production"].values.astype(np.float32)

    # Mantengo la lógica antigua:
    # para una fila tabular válida se exigía X finito y production finita.
    valid_tab_row = (
        np.isfinite(x_values).all(axis=1)
        & np.isfinite(y_values)
    )

    start_time = time.time()

    for label_idx in range(window, n):
        if label_idx == window or label_idx % 5000 == 0 or label_idx == n - 1:
            pct = 100.0 * label_idx / max(1, n - 1)
            elapsed = time.time() - start_time
            print(
                f"  build samples: {label_idx}/{n - 1} "
                f"({pct:.2f}%) | samples={len(samples)} | "
                f"imagenes_en_ventana={len(previous_images)} | "
                f"elapsed={elapsed/60:.1f} min",
                flush=True,
            )

        current_ts = timestamps_ns[label_idx]
        cutoff_ts = current_ts - max_age_ns

        # Quitamos imágenes demasiado antiguas.
        # Así previous_images solo contiene imágenes dentro de MAX_IMAGE_AGE_MINUTES.
        while previous_images and timestamps_ns[previous_images[0]] < cutoff_ts:
            previous_images.popleft()

        start_idx = label_idx - window
        end_idx = label_idx

        y = y_values[label_idx]

        sample_ok = True

        if not np.isfinite(y):
            sample_ok = False

        if sample_ok and float(y) < MIN_PRODUCTION_FOR_SAMPLE:
            sample_ok = False

        if sample_ok and not valid_tab_row[start_idx:end_idx].all():
            sample_ok = False

        if sample_ok and len(previous_images) >= window:
            image_indices = list(previous_images)[-window:]
            tab_indices = list(range(start_idx, end_idx))

            samples.append(
                {
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "label_idx": label_idx,
                    "tab_indices": tab_indices,
                    "image_indices": image_indices,
                }
            )

        # Importante:
        # añadimos la imagen actual DESPUÉS para no usar la imagen del mismo timestamp.
        if has_image[label_idx]:
            previous_images.append(label_idx)

    elapsed = time.time() - start_time

    print()
    print("Construcción de samples terminada.", flush=True)
    print(f"  Samples válidos: {len(samples)}", flush=True)
    print(f"  Tiempo: {elapsed:.2f} s ({elapsed/60:.2f} min)", flush=True)

    return samples


# ============================================================
# Dataset LSTM tabular usando samples multimodales
# ============================================================

class TabularSameSamplesAsMultimodalDataset(Dataset):
    def __init__(
        self,
        df,
        input_cols,
        samples,
        x_min,
        x_range,
        y_min,
        y_range,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.input_cols = input_cols
        self.samples = samples

        self.x_min = x_min
        self.x_range = x_range
        self.y_min = y_min
        self.y_range = y_range

        x_raw = self.df[self.input_cols].values.astype(np.float32)
        y_raw = self.df["production"].values.astype(np.float32)

        self.x = minmax_x(x_raw, self.x_min, self.x_range).astype(np.float32)
        self.y = minmax_y(y_raw, self.y_min, self.y_range).astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        tab_indices = sample["tab_indices"]
        label_idx = sample["label_idx"]

        # Exactamente la misma ventana tabular que el multimodal.
        # NO usa production.
        # NO usa imágenes.
        x_window = self.x[tab_indices]

        y = self.y[label_idx]

        return (
            torch.tensor(x_window, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


# ============================================================
# Modelo LSTM puro
# ============================================================

class PureLSTMRegressor(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, W, C]
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        y_hat = self.regressor(last).squeeze(1)
        return y_hat


# ============================================================
# Train / Eval
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    total_loss = 0.0

    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx % 20 == 0:
            print(f"  batch {batch_idx}/{len(loader)}", flush=True)

        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        pred = model(x)
        loss = criterion(pred, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, y_min, y_range, return_arrays=False):
    model.eval()

    total_loss = 0.0
    preds_all = []
    y_all = []

    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx % 20 == 0:
            print(f"  eval batch {batch_idx}/{len(loader)}", flush=True)

        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        pred = model(x)
        loss = criterion(pred, y)

        total_loss += loss.item() * x.size(0)

        pred_real = inverse_minmax_y(
            pred.detach().cpu().numpy(),
            y_min,
            y_range,
        )

        y_real = inverse_minmax_y(
            y.detach().cpu().numpy(),
            y_min,
            y_range,
        )

        preds_all.append(pred_real)
        y_all.append(y_real)

    preds_all = np.concatenate(preds_all)
    y_all = np.concatenate(y_all)

    errors = preds_all - y_all

    # MAE:
    # error absoluto medio en unidades reales de producción.
    mae = float(np.mean(np.abs(errors)))

    # RMSE:
    # raíz del error cuadrático medio; penaliza más los errores grandes/picos.
    rmse = float(math.sqrt(np.mean(errors ** 2)))

    # MAE% mean / RMSE% mean:
    # error relativo respecto a la producción media absoluta del fold.
    mean_y_abs = float(np.mean(np.abs(y_all)))

    if mean_y_abs > EPS:
        mae_rel_pct = 100.0 * mae / mean_y_abs
        rmse_rel_pct = 100.0 * rmse / mean_y_abs
    else:
        mae_rel_pct = float("nan")
        rmse_rel_pct = float("nan")

    # nMAE/capacity y nRMSE/capacity:
    # error normalizado por capacidad de referencia.
    # En main() CAPACITY se define como percentil 99.5 de production.
    if CAPACITY is not None and np.isfinite(CAPACITY) and CAPACITY > EPS:
        capacity = float(CAPACITY)
        nmae_capacity_pct = 100.0 * mae / capacity
        nrmse_capacity_pct = 100.0 * rmse / capacity
        mape_10pct_capacity_threshold = 0.10 * capacity
    else:
        capacity = float("nan")
        nmae_capacity_pct = float("nan")
        nrmse_capacity_pct = float("nan")
        mape_10pct_capacity_threshold = float("nan")

    # MAPE@>1000:
    # error porcentual punto a punto solo cuando hay producción relevante.
    mask_1000 = np.abs(y_all) > MAPE_MIN_Y

    if mask_1000.sum() > 0:
        mape_1000_pct = float(
            np.mean(
                np.abs((preds_all[mask_1000] - y_all[mask_1000]) / y_all[mask_1000])
            ) * 100.0
        )
    else:
        mape_1000_pct = float("nan")

    # MAPE@>10% capacity:
    # alternativa más física: solo evalúa cuando y_true supera el 10% de la capacidad.
    if np.isfinite(mape_10pct_capacity_threshold):
        mask_10cap = np.abs(y_all) > mape_10pct_capacity_threshold
    else:
        mask_10cap = np.zeros_like(y_all, dtype=bool)

    if mask_10cap.sum() > 0:
        mape_10pct_capacity_pct = float(
            np.mean(
                np.abs(
                    (preds_all[mask_10cap] - y_all[mask_10cap])
                    / y_all[mask_10cap]
                )
            ) * 100.0
        )
    else:
        mape_10pct_capacity_pct = float("nan")

    metrics = {
        "loss": total_loss / len(loader.dataset),

        # Métricas en unidades reales.
        "mae": mae,
        "rmse": rmse,

        # Relativas a producción media del fold.
        "mae_rel_pct": mae_rel_pct,
        "rmse_rel_pct": rmse_rel_pct,
        "mean_y_abs": mean_y_abs,

        # Relativas a capacidad de referencia.
        "capacity": capacity,
        "nmae_capacity_pct": nmae_capacity_pct,
        "nrmse_capacity_pct": nrmse_capacity_pct,

        # MAPE con producción relevante.
        "mape_pct": mape_1000_pct,
        "mape_1000_pct": mape_1000_pct,
        "mape_min_y": MAPE_MIN_Y,
        "mape_10pct_capacity_pct": mape_10pct_capacity_pct,
        "mape_10pct_capacity_threshold": mape_10pct_capacity_threshold,
    }

    if return_arrays:
        metrics["preds"] = preds_all
        metrics["y_true"] = y_all

    return metrics


@torch.no_grad()
def evaluate_mean_train_baseline(loader, y_min, y_range, train_y_mean_real):
    preds_all = []
    y_all = []

    for x, y in loader:
        y_real = inverse_minmax_y(
            y.cpu().numpy(),
            y_min,
            y_range,
        )

        pred_real = np.full_like(
            y_real,
            fill_value=train_y_mean_real,
            dtype=np.float32,
        )

        preds_all.append(pred_real)
        y_all.append(y_real)

    preds_all = np.concatenate(preds_all)
    y_all = np.concatenate(y_all)

    errors = preds_all - y_all

    mae = float(np.mean(np.abs(errors)))
    rmse = float(math.sqrt(np.mean(errors ** 2)))
    mean_y_abs = float(np.mean(np.abs(y_all)))

    mae_rel_pct = 100.0 * mae / mean_y_abs if mean_y_abs > EPS else float("nan")
    rmse_rel_pct = 100.0 * rmse / mean_y_abs if mean_y_abs > EPS else float("nan")

    if CAPACITY is not None and np.isfinite(CAPACITY) and CAPACITY > EPS:
        capacity = float(CAPACITY)
        nmae_capacity_pct = 100.0 * mae / capacity
        nrmse_capacity_pct = 100.0 * rmse / capacity
        mape_10pct_capacity_threshold = 0.10 * capacity
    else:
        capacity = float("nan")
        nmae_capacity_pct = float("nan")
        nrmse_capacity_pct = float("nan")
        mape_10pct_capacity_threshold = float("nan")

    mask_1000 = np.abs(y_all) > MAPE_MIN_Y

    if mask_1000.sum() > 0:
        mape_1000_pct = float(
            np.mean(
                np.abs((preds_all[mask_1000] - y_all[mask_1000]) / y_all[mask_1000])
            ) * 100.0
        )
    else:
        mape_1000_pct = float("nan")

    if np.isfinite(mape_10pct_capacity_threshold):
        mask_10cap = np.abs(y_all) > mape_10pct_capacity_threshold
    else:
        mask_10cap = np.zeros_like(y_all, dtype=bool)

    if mask_10cap.sum() > 0:
        mape_10pct_capacity_pct = float(
            np.mean(
                np.abs(
                    (preds_all[mask_10cap] - y_all[mask_10cap])
                    / y_all[mask_10cap]
                )
            ) * 100.0
        )
    else:
        mape_10pct_capacity_pct = float("nan")

    print()
    print("Baseline media train:")
    print(f"  MAE={mae:.2f}")
    print(f"  RMSE={rmse:.2f}")
    print(f"  MAE% mean={mae_rel_pct:.2f}%")
    print(f"  RMSE% mean={rmse_rel_pct:.2f}%")
    print(f"  nMAE/capacity={nmae_capacity_pct:.2f}%")
    print(f"  nRMSE/capacity={nrmse_capacity_pct:.2f}%")
    print(f"  MAPE@>{MAPE_MIN_Y:.0f}={mape_1000_pct:.2f}%")
    print(
        f"  MAPE@>10%capacity "
        f"(>{mape_10pct_capacity_threshold:.2f})={mape_10pct_capacity_pct:.2f}%"
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "mae_rel_pct": mae_rel_pct,
        "rmse_rel_pct": rmse_rel_pct,
        "mean_y_abs": mean_y_abs,
        "capacity": capacity,
        "nmae_capacity_pct": nmae_capacity_pct,
        "nrmse_capacity_pct": nrmse_capacity_pct,
        "mape_pct": mape_1000_pct,
        "mape_1000_pct": mape_1000_pct,
        "mape_min_y": MAPE_MIN_Y,
        "mape_10pct_capacity_pct": mape_10pct_capacity_pct,
        "mape_10pct_capacity_threshold": mape_10pct_capacity_threshold,
    }


# ============================================================
# Plot timeline
# ============================================================

def save_eval_timeline_plot(timestamps, y_true, y_pred, out_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(18, 6))

    plt.plot(
        timestamps,
        y_true,
        label="Ground truth",
        linewidth=2,
    )

    plt.plot(
        timestamps,
        y_pred,
        label="Prediction",
        linewidth=2,
        alpha=0.8,
    )

    plt.xlabel("Timestamp")
    plt.ylabel("PV production")
    plt.title("Evaluation timeline: Ground truth vs Prediction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Timeline de evaluación guardado en: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)

    print(f"Usando dispositivo: {DEVICE}")
    print(f"Leyendo: {TSV_PATH}")
    print(f"W: {W}")
    print(f"MAX_IMAGE_AGE_MINUTES: {MAX_IMAGE_AGE_MINUTES}")
    print("Modo: LSTM tabular con EXACTAMENTE los mismos samples que el multimodal")
    print("production como input: NO")
    print("imágenes como input: NO")
    print("image_path usado solo para construir los mismos samples: SÍ")

    df = pd.read_csv(TSV_PATH, sep="\t")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    if "image_path" not in df.columns:
        raise RuntimeError("No existe la columna obligatoria 'image_path'.")

    df["image_path"] = df["image_path"].fillna("").astype(str)

    if "production" not in df.columns:
        raise RuntimeError("No existe la columna obligatoria 'production'.")

    input_cols = [
        col for col in PREFERRED_INPUT_COLUMNS
        if col in df.columns and col not in IGNORE_COLUMNS
    ]

    if "production" in input_cols:
        raise RuntimeError("Error: production no debe estar en input_cols.")

    if len(input_cols) == 0:
        raise RuntimeError("No hay columnas de entrada válidas.")

    numeric_cols = list(dict.fromkeys(input_cols + ["production"]))

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    after = len(df)

    global CAPACITY
    CAPACITY = float(
        np.percentile(
            df["production"].dropna().values.astype(np.float32),
            CAPACITY_PERCENTILE,
        )
    )

    print()
    print(f"CAPACITY p{CAPACITY_PERCENTILE:.1f} production: {CAPACITY:.2f}")
    print(f"Umbral MAPE@>10%capacity: {0.10 * CAPACITY:.2f}")

    print()
    print("Columnas usadas como entrada LSTM:")
    for col in input_cols:
        print(f"  - {col}")

    print()
    print("Columna objetivo:")
    print("  - production")

    print()
    print(f"Filas originales: {before}")
    print(f"Filas tras dropna numérico: {after}")
    print(f"Filas eliminadas: {before - after}")
    print(f"Filas con image_path vacío: {(df['image_path'].str.strip() == '').sum()}")
    print(f"Filas con image_path no vacío: {df['image_path'].str.strip().ne('').sum()}")
    print(f"W: {W}")
    print(f"MAX_IMAGE_AGE_MINUTES: {MAX_IMAGE_AGE_MINUTES}")
    print(f"MIN_PRODUCTION_FOR_SAMPLE: {MIN_PRODUCTION_FOR_SAMPLE}")

    print_correlations(df, input_cols)

    # CLAVE:
    # Usamos la MISMA función de samples que el multimodal.
    all_samples = build_multimodal_samples(
        df=df,
        input_cols=input_cols,
        window=W,
        max_image_age_minutes=MAX_IMAGE_AGE_MINUTES,
    )

    if len(all_samples) == 0:
        raise RuntimeError(
            "No hay samples válidos. "
            "Revisa image_path, W, MAX_IMAGE_AGE_MINUTES o MIN_PRODUCTION_FOR_SAMPLE."
        )

    sample_split_idx = int(len(all_samples) * TRAIN_RATIO)

    train_samples = all_samples[:sample_split_idx]
    eval_samples = all_samples[sample_split_idx:]

    if len(train_samples) == 0:
        raise RuntimeError("No hay samples de entrenamiento.")

    if len(eval_samples) == 0:
        raise RuntimeError("No hay samples de evaluación.")

    # Mismo ajuste que el multimodal:
    # escaladores SOLO con filas usadas como label en train.
    train_label_indices = [s["label_idx"] for s in train_samples]
    train_scaler_df = df.iloc[train_label_indices].reset_index(drop=True)

    x_min, x_range, y_min, y_range = fit_minmax(
        train_scaler_df,
        input_cols,
    )

    train_dataset = TabularSameSamplesAsMultimodalDataset(
        df=df,
        input_cols=input_cols,
        samples=train_samples,
        x_min=x_min,
        x_range=x_range,
        y_min=y_min,
        y_range=y_range,
    )

    eval_dataset = TabularSameSamplesAsMultimodalDataset(
        df=df,
        input_cols=input_cols,
        samples=eval_samples,
        x_min=x_min,
        x_range=x_range,
        y_min=y_min,
        y_range=y_range,
    )

    print()
    print("Diagnóstico samples:")
    print(f"Samples válidos totales, mismos que multimodal: {len(all_samples)}")
    print(f"Samples train: {len(train_dataset)}")
    print(f"Samples eval: {len(eval_dataset)}")
    print(f"y_min train: {y_min:.4f}")
    print(f"y_range train: {y_range:.4f}")
    print(f"y_max train: {y_min + y_range:.4f}")
    print(f"x_min train: {x_min}")
    print(f"x_range train: {x_range}")

    first = train_samples[0]
    print()
    print("Primer sample train:")
    print(f"  Ventana tabular: [{first['start_idx']}, {first['end_idx']})")
    print(f"  Label idx: {first['label_idx']}")
    print(f"  Label timestamp: {df.loc[first['label_idx'], 'timestamp']}")
    print(f"  Label production: {df.loc[first['label_idx'], 'production']:.2f}")
    print("  Índices tabulares, usados por la LSTM:")
    for idx in first["tab_indices"]:
        age = df.loc[first["label_idx"], "timestamp"] - df.loc[idx, "timestamp"]
        print(
            f"    {idx} | "
            f"{df.loc[idx, 'timestamp']} | "
            f"age={age}"
        )
    print("  Imágenes anteriores, solo usadas para que el sample exista:")
    for idx in first["image_indices"]:
        age = df.loc[first["label_idx"], "timestamp"] - df.loc[idx, "timestamp"]
        print(
            f"    {idx} | "
            f"{df.loc[idx, 'timestamp']} | "
            f"age={age} | "
            f"{df.loc[idx, 'image_path']}"
        )

    first_eval = eval_samples[0]
    print()
    print("Primer sample eval:")
    print(f"  Ventana tabular: [{first_eval['start_idx']}, {first_eval['end_idx']})")
    print(f"  Label idx: {first_eval['label_idx']}")
    print(f"  Label timestamp: {df.loc[first_eval['label_idx'], 'timestamp']}")
    print(f"  Label production: {df.loc[first_eval['label_idx'], 'production']:.2f}")
    print("  Índices tabulares, usados por la LSTM:")
    for idx in first_eval["tab_indices"]:
        age = df.loc[first_eval["label_idx"], "timestamp"] - df.loc[idx, "timestamp"]
        print(
            f"    {idx} | "
            f"{df.loc[idx, 'timestamp']} | "
            f"age={age}"
        )
    print("  Imágenes anteriores, solo usadas para que el sample exista:")
    for idx in first_eval["image_indices"]:
        age = df.loc[first_eval["label_idx"], "timestamp"] - df.loc[idx, "timestamp"]
        print(
            f"    {idx} | "
            f"{df.loc[idx, 'timestamp']} | "
            f"age={age} | "
            f"{df.loc[idx, 'image_path']}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    train_y_mean_real = float(train_scaler_df["production"].mean())

    mean_baseline_metrics = evaluate_mean_train_baseline(
        eval_loader,
        y_min=y_min,
        y_range=y_range,
        train_y_mean_real=train_y_mean_real,
    )

    model = PureLSTMRegressor(
        num_features=len(input_cols),
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
    ).to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        print()
        print(f"Epoch {epoch:03d}/{EPOCHS}")
        print("Train:")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        print("Eval:")

        metrics = evaluate(
            model,
            eval_loader,
            criterion,
            y_min=y_min,
            y_range=y_range,
        )

        eval_loss = metrics["loss"]
        eval_mae = metrics["mae"]
        eval_rmse = metrics["rmse"]
        eval_mae_rel_pct = metrics["mae_rel_pct"]
        eval_rmse_rel_pct = metrics["rmse_rel_pct"]
        eval_nmae_capacity_pct = metrics["nmae_capacity_pct"]
        eval_nrmse_capacity_pct = metrics["nrmse_capacity_pct"]
        eval_mape_pct = metrics["mape_1000_pct"]
        eval_mape_10pct_capacity_pct = metrics["mape_10pct_capacity_pct"]
        eval_mape_10pct_capacity_threshold = metrics["mape_10pct_capacity_threshold"]
        eval_mean_y_abs = metrics["mean_y_abs"]

        scheduler.step(eval_rmse)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | "
            f"lr={current_lr:.2e} | "
            f"train_loss={train_loss:.5f} | "
            f"eval_loss={eval_loss:.5f} | "
            f"eval_MAE={eval_mae:.2f} | "
            f"eval_RMSE={eval_rmse:.2f} | "
            f"eval_MAE%mean={eval_mae_rel_pct:.2f}% | "
            f"eval_RMSE%mean={eval_rmse_rel_pct:.2f}% | "
            f"eval_nMAE/cap={eval_nmae_capacity_pct:.2f}% | "
            f"eval_nRMSE/cap={eval_nrmse_capacity_pct:.2f}% | "
            f"eval_MAPE@>{MAPE_MIN_Y:.0f}={eval_mape_pct:.2f}% | "
            f"eval_MAPE@>10%cap(>{eval_mape_10pct_capacity_threshold:.0f})="
            f"{eval_mape_10pct_capacity_pct:.2f}% | "
            f"eval_mean_abs_y={eval_mean_y_abs:.2f}"
        )

        improved = eval_rmse < (best_rmse - EARLY_STOPPING_MIN_DELTA)

        if improved:
            best_rmse = eval_rmse
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_cols": input_cols,
                    "x_min": x_min,
                    "x_range": x_range,
                    "y_min": y_min,
                    "y_range": y_range,
                    "window": W,
                    "model_type": "pure_lstm_tabular_same_samples_as_multimodal_no_autoreg",
                    "uses_production_as_input": False,
                    "uses_images_as_input": False,
                    "uses_multimodal_sample_builder": True,
                    "image_paths_used_only_for_sample_selection": True,
                    "max_image_age_minutes": MAX_IMAGE_AGE_MINUTES,
                    "best_epoch": best_epoch,
                    "best_rmse": best_rmse,
                    "best_metrics": metrics,
                    "mean_baseline_metrics": mean_baseline_metrics,
                    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                    "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
                    "mape_min_y": MAPE_MIN_Y,
                    "min_production_for_sample": MIN_PRODUCTION_FOR_SAMPLE,
                },
                MODEL_OUT,
            )

            print(
                f"  Nuevo mejor modelo guardado: "
                f"{MODEL_OUT} | "
                f"RMSE={best_rmse:.2f} | "
                f"RMSE%mean={eval_rmse_rel_pct:.2f}% | "
                f"nRMSE/cap={eval_nrmse_capacity_pct:.2f}%"
            )

        else:
            epochs_without_improvement += 1
            print(
                f"  Sin mejora en RMSE: "
                f"{epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}"
            )

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print()
                print(
                    "Early stopping activado: "
                    f"no mejora durante {EARLY_STOPPING_PATIENCE} epochs."
                )
                break

    print()
    print("Cargando mejor modelo para guardar timeline y predicciones...")

    checkpoint = torch.load(
        MODEL_OUT,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    final_metrics = evaluate(
        model,
        eval_loader,
        criterion,
        y_min=y_min,
        y_range=y_range,
        return_arrays=True,
    )

    eval_label_indices = [s["label_idx"] for s in eval_samples]
    eval_timestamps = df.loc[eval_label_indices, "timestamp"].values

    pred_df = pd.DataFrame(
        {
            "timestamp": eval_timestamps,
            "y_true": final_metrics["y_true"],
            "y_pred": final_metrics["preds"],
            "error": final_metrics["preds"] - final_metrics["y_true"],
            "abs_error": np.abs(final_metrics["preds"] - final_metrics["y_true"]),
            "capacity": final_metrics["capacity"],
            "abs_error_pct_capacity": (
                100.0
                * np.abs(final_metrics["preds"] - final_metrics["y_true"])
                / final_metrics["capacity"]
            ),
        }
    )

    pred_df.to_csv(PREDICTIONS_OUT, sep="\t", index=False)
    print(f"Predicciones eval guardadas en: {PREDICTIONS_OUT}")

    save_eval_timeline_plot(
        timestamps=eval_timestamps,
        y_true=final_metrics["y_true"],
        y_pred=final_metrics["preds"],
        out_path=PLOT_OUT,
    )

    print()
    print("Entrenamiento terminado.")
    print(f"Mejor epoch: {best_epoch}")
    print(f"Mejor RMSE eval: {best_rmse:.2f}")
    print(f"Modelo guardado en: {MODEL_OUT}")
    print(f"Predicciones guardadas en: {PREDICTIONS_OUT}")
    print(f"Plot guardado en: {PLOT_OUT}")


if __name__ == "__main__":
    main()