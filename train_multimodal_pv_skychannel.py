from pathlib import Path
import math
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# Config
# ============================================================

TSV_PATH = Path("data/weather_with_images.tsv")

# Ventana temporal. Si tus datos son cada 5 min:
# W=8 -> 40 minutos
W = 8

IMG_SIZE = 64

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

MODEL_OUT = "best_multimodal_lstm_weather_img_no_autoreg.pt"
PREDICTIONS_OUT = "eval_predictions_multimodal_lstm_weather_img_no_autoreg.tsv"
PLOT_OUT = "eval_timeline_multimodal_lstm_weather_img_no_autoreg.png"

# Early stopping
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MIN_DELTA = 1e-4

# Para MAPE: ignora valores reales muy cercanos a 0.
MAPE_MIN_Y = 100.0

# Si quieres quitar noche/producciones muy bajas del entrenamiento/evaluación.
# Déjalo en 0.0 para usar todo.
MIN_PRODUCTION_FOR_SAMPLE = 0.0

# Edad máxima de imágenes anteriores respecto al timestamp objetivo.
MAX_IMAGE_AGE_MINUTES = 120

# Normalización MinMax de entradas y salida.
# Se ajusta SOLO con train para evitar fuga de información.
EPS = 1e-8

# IMPORTANTE:
# production NO se usa como input.
# production queda SOLO como target y.
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

# Embeddings:
# La parte tabular manda.
# La imagen entra pequeña y con puerta suave.
TAB_EMB_DIM = 128
IMG_EMB_DIM = 128


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
# Imagen: RGB -> canal azul/verdoso + blanco
# ============================================================

def rgb_to_sky_channel(img):
    """
    Convierte una imagen RGB a 1 canal que prioriza:
      - brillo general
      - azul/verdoso
      - blanco/nubes

    Devuelve PIL Image en escala de grises.
    """
    img = img.convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    brightness = (r + g + b) / 3.0
    blue_green = (g + b) / 2.0 - 0.5 * r
    color_std = np.std(arr, axis=2)
    whiteness = brightness - color_std

    channel = 0.60 * brightness + 0.30 * blue_green + 0.10 * whiteness
    channel = np.clip(channel, 0.0, 1.0)

    return Image.fromarray((channel * 255).astype(np.uint8))


def image_path_exists(p):
    if not isinstance(p, str):
        return False

    p = p.strip()

    if p == "":
        return False

    return Path(p).exists()


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
# Samples multimodales
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
    Para cada label_idx:
      - y = production[label_idx]
      - X_tab = variables tabulares en las W filas anteriores: label_idx-W ... label_idx-1
      - X_img = W imágenes anteriores existentes, nunca la imagen del mismo timestamp actual

    IMPORTANTE:
      production NO está en input_cols.
    """
    samples = []
    previous_image_indices = []
    max_image_age = pd.Timedelta(minutes=max_image_age_minutes)

    has_image = df["image_path"].apply(image_path_exists).values

    for label_idx in range(window, len(df)):
        start_idx = label_idx - window
        end_idx = label_idx

        tab_indices = list(range(start_idx, end_idx))

        y = df.loc[label_idx, "production"]

        if not np.isfinite(float(y)):
            if has_image[label_idx]:
                previous_image_indices.append(label_idx)
            continue

        if float(y) < MIN_PRODUCTION_FOR_SAMPLE:
            if has_image[label_idx]:
                previous_image_indices.append(label_idx)
            continue

        valid_tabular = all(
            row_has_valid_values(df, idx, input_cols)
            for idx in tab_indices
        )

        if not valid_tabular:
            if has_image[label_idx]:
                previous_image_indices.append(label_idx)
            continue

        current_ts = df.loc[label_idx, "timestamp"]

        valid_previous_images = [
            idx for idx in previous_image_indices
            if current_ts - df.loc[idx, "timestamp"] <= max_image_age
        ]

        if len(valid_previous_images) >= window:
            image_indices = valid_previous_images[-window:]

            samples.append(
                {
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "label_idx": label_idx,
                    "tab_indices": tab_indices,
                    "image_indices": image_indices,
                }
            )

        # Añadimos la imagen actual DESPUÉS para evitar usar imagen del mismo timestamp.
        if has_image[label_idx]:
            previous_image_indices.append(label_idx)

    return samples


# ============================================================
# Dataset
# ============================================================

class MultimodalPVForecastDataset(Dataset):
    def __init__(
        self,
        df,
        input_cols,
        samples,
        x_min,
        x_range,
        y_min,
        y_range,
        img_size=64,
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

        self.img_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        tab_indices = sample["tab_indices"]
        image_indices = sample["image_indices"]
        label_idx = sample["label_idx"]

        x_tab = self.x[tab_indices]
        y = self.y[label_idx]

        image_paths = self.df.iloc[image_indices]["image_path"].tolist()

        imgs = []

        for p in image_paths:
            img = Image.open(p)
            img = rgb_to_sky_channel(img)
            img = self.img_transform(img)
            imgs.append(img)

        imgs = torch.stack(imgs, dim=0)
        # imgs: [W, 1, IMG_SIZE, IMG_SIZE]

        return (
            torch.tensor(x_tab, dtype=torch.float32),
            imgs,
            torch.tensor(y, dtype=torch.float32),
        )


# ============================================================
# Encoder tabular: misma filosofía LSTM pura
# ============================================================

class TabularLSTMEncoder(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        emb_dim=128,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.proj = nn.Sequential(
            nn.Linear(hidden_size, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
        )

    def forward(self, x):
        # x: [B, W, C]
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        emb = self.proj(last)
        return emb


# ============================================================
# Encoder visual pequeño
# ============================================================

class GentleImageEncoder(nn.Module):
    def __init__(
        self,
        img_emb_dim=32,
        lstm_hidden=64,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        self.proj = nn.Sequential(
            nn.Linear(lstm_hidden, img_emb_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
        )

    def forward(self, imgs):
        # imgs: [B, W, 1, H, W]
        b, t, c, h, w = imgs.shape

        x = imgs.reshape(b * t, c, h, w)
        x = self.cnn(x)
        x = x.reshape(b, t, 32)

        out, _ = self.lstm(x)
        last = out[:, -1, :]

        img_emb = self.proj(last)
        return img_emb


# ============================================================
# Modelo multimodal suave
# ============================================================

class GentleMultimodalPVModel(nn.Module):
    def __init__(
        self,
        num_features,
        tab_emb_dim=128,
        img_emb_dim=32,
    ):
        super().__init__()

        self.tab_encoder = TabularLSTMEncoder(
            num_features=num_features,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            emb_dim=tab_emb_dim,
        )

        self.img_encoder = GentleImageEncoder(
            img_emb_dim=img_emb_dim,
            lstm_hidden=64,
        )

        # Proyectamos la imagen pequeña al espacio tabular.
        self.img_to_tab = nn.Sequential(
            nn.Linear(img_emb_dim, tab_emb_dim),
            nn.Tanh(),
        )

        # Puerta aprendible: permite que el modelo use imagen,
        # pero no la deja dominar de golpe.
        self.img_gate = nn.Sequential(
            nn.Linear(tab_emb_dim + img_emb_dim, tab_emb_dim),
            nn.Sigmoid(),
        )

        # Fusión inicial para probar:
        # concat + multiplicación.
        # No meto abs(tab-img) todavía para no sobredimensionar la rama visual.
        fusion_dim = tab_emb_dim * 3

        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_tab, imgs):
        tab_emb = self.tab_encoder(x_tab)
        img_small = self.img_encoder(imgs)

        img_proj = self.img_to_tab(img_small)

        gate_input = torch.cat([tab_emb, img_small], dim=1)
        gate = self.img_gate(gate_input)

        img_soft = gate * img_proj

        fusion = torch.cat(
            [
                tab_emb,
                img_soft,
                tab_emb * img_soft,
            ],
            dim=1,
        )

        y_hat = self.regressor(fusion).squeeze(1)
        return y_hat


# ============================================================
# Train / Eval
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    total_loss = 0.0

    for batch_idx, (x_tab, imgs, y) in enumerate(loader):
        if batch_idx % 20 == 0:
            print(f"  batch {batch_idx}/{len(loader)}", flush=True)

        x_tab = x_tab.to(DEVICE, non_blocking=True)
        imgs = imgs.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        pred = model(x_tab, imgs)
        loss = criterion(pred, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        total_loss += loss.item() * x_tab.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, y_min, y_range, return_arrays=False):
    model.eval()

    total_loss = 0.0
    preds_all = []
    y_all = []

    for batch_idx, (x_tab, imgs, y) in enumerate(loader):
        if batch_idx % 20 == 0:
            print(f"  eval batch {batch_idx}/{len(loader)}", flush=True)

        x_tab = x_tab.to(DEVICE, non_blocking=True)
        imgs = imgs.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        pred = model(x_tab, imgs)
        loss = criterion(pred, y)

        total_loss += loss.item() * x_tab.size(0)

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

    mae = float(np.mean(np.abs(errors)))
    rmse = float(math.sqrt(np.mean(errors ** 2)))

    mean_y_abs = float(np.mean(np.abs(y_all)))

    if mean_y_abs > EPS:
        mae_rel_pct = 100.0 * mae / mean_y_abs
        rmse_rel_pct = 100.0 * rmse / mean_y_abs
    else:
        mae_rel_pct = float("nan")
        rmse_rel_pct = float("nan")

    mask = np.abs(y_all) > MAPE_MIN_Y

    if mask.sum() > 0:
        mape_pct = float(
            np.mean(
                np.abs((preds_all[mask] - y_all[mask]) / y_all[mask])
            ) * 100.0
        )
    else:
        mape_pct = float("nan")

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "mae": mae,
        "rmse": rmse,
        "mae_rel_pct": mae_rel_pct,
        "rmse_rel_pct": rmse_rel_pct,
        "mape_pct": mape_pct,
        "mean_y_abs": mean_y_abs,
    }

    if return_arrays:
        metrics["preds"] = preds_all
        metrics["y_true"] = y_all

    return metrics


@torch.no_grad()
def evaluate_mean_train_baseline(loader, y_min, y_range, train_y_mean_real):
    preds_all = []
    y_all = []

    for x_tab, imgs, y in loader:
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

    mask = np.abs(y_all) > MAPE_MIN_Y

    if mask.sum() > 0:
        mape_pct = float(
            np.mean(
                np.abs((preds_all[mask] - y_all[mask]) / y_all[mask])
            ) * 100.0
        )
    else:
        mape_pct = float("nan")

    print()
    print("Baseline media train:")
    print(f"  MAE={mae:.2f}")
    print(f"  RMSE={rmse:.2f}")
    print(f"  MAE%={mae_rel_pct:.2f}%")
    print(f"  RMSE%={rmse_rel_pct:.2f}%")
    print(f"  MAPE@>{MAPE_MIN_Y:.0f}={mape_pct:.2f}%")

    return {
        "mae": mae,
        "rmse": rmse,
        "mae_rel_pct": mae_rel_pct,
        "rmse_rel_pct": rmse_rel_pct,
        "mape_pct": mape_pct,
        "mean_y_abs": mean_y_abs,
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
    print(f"IMG_SIZE: {IMG_SIZE}")
    print(f"MAX_IMAGE_AGE_MINUTES: {MAX_IMAGE_AGE_MINUTES}")
    print("Modo: multimodal weather + imágenes")
    print("production como input: NO")

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
    print(f"Filas con imagen existente: {df['image_path'].apply(image_path_exists).sum()}")
    print(f"MIN_PRODUCTION_FOR_SAMPLE: {MIN_PRODUCTION_FOR_SAMPLE}")

    print_correlations(df, input_cols)

    all_samples = build_multimodal_samples(
        df=df,
        input_cols=input_cols,
        window=W,
        max_image_age_minutes=MAX_IMAGE_AGE_MINUTES,
    )

    if len(all_samples) == 0:
        raise RuntimeError(
            "No hay samples multimodales válidos. "
            "Revisa image_path, W o MAX_IMAGE_AGE_MINUTES."
        )

    sample_split_idx = int(len(all_samples) * TRAIN_RATIO)

    train_samples = all_samples[:sample_split_idx]
    eval_samples = all_samples[sample_split_idx:]

    if len(train_samples) == 0:
        raise RuntimeError("No hay samples de entrenamiento.")

    if len(eval_samples) == 0:
        raise RuntimeError("No hay samples de evaluación.")

    # Fittear escaladores SOLO con filas usadas como label en train.
    train_label_indices = [s["label_idx"] for s in train_samples]
    train_scaler_df = df.iloc[train_label_indices].reset_index(drop=True)

    x_min, x_range, y_min, y_range = fit_minmax(
        train_scaler_df,
        input_cols,
    )

    train_dataset = MultimodalPVForecastDataset(
        df=df,
        input_cols=input_cols,
        samples=train_samples,
        x_min=x_min,
        x_range=x_range,
        y_min=y_min,
        y_range=y_range,
        img_size=IMG_SIZE,
    )

    eval_dataset = MultimodalPVForecastDataset(
        df=df,
        input_cols=input_cols,
        samples=eval_samples,
        x_min=x_min,
        x_range=x_range,
        y_min=y_min,
        y_range=y_range,
        img_size=IMG_SIZE,
    )

    print()
    print("Diagnóstico samples:")
    print(f"Samples multimodales válidos totales: {len(all_samples)}")
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
    print("  Índices tabulares:")
    for idx in first["tab_indices"]:
        age = df.loc[first["label_idx"], "timestamp"] - df.loc[idx, "timestamp"]
        print(
            f"    {idx} | "
            f"{df.loc[idx, 'timestamp']} | "
            f"age={age}"
        )
    print("  Imágenes anteriores:")
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
    print("  Imágenes anteriores:")
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
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    train_y_mean_real = float(train_scaler_df["production"].mean())

    mean_baseline_metrics = evaluate_mean_train_baseline(
        eval_loader,
        y_min=y_min,
        y_range=y_range,
        train_y_mean_real=train_y_mean_real,
    )

    model = GentleMultimodalPVModel(
        num_features=len(input_cols),
        tab_emb_dim=TAB_EMB_DIM,
        img_emb_dim=IMG_EMB_DIM,
    ).to(DEVICE)


    print(model)

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
        eval_mape_pct = metrics["mape_pct"]
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
            f"eval_MAE%={eval_mae_rel_pct:.2f}% | "
            f"eval_RMSE%={eval_rmse_rel_pct:.2f}% | "
            f"eval_MAPE@>{MAPE_MIN_Y:.0f}={eval_mape_pct:.2f}% | "
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
                    "img_size": IMG_SIZE,
                    "model_type": "gentle_multimodal_lstm_weather_img_no_autoreg",
                    "uses_production_as_input": False,
                    "fusion": "concat_tab_imgsoft_mul",
                    "tab_emb_dim": TAB_EMB_DIM,
                    "img_emb_dim": IMG_EMB_DIM,
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
                f"RMSE%={eval_rmse_rel_pct:.2f}%"
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