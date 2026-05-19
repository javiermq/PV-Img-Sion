from pathlib import Path
from collections import deque
import math
import random
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Config
# ============================================================

TSV_PATH = Path("data/weather_with_images.tsv")
IMAGE_CACHE_PATH = Path("data/images_cache.pt")

# Ventana temporal. Si tus datos son cada 5 min:
# W=8 -> 40 minutos
W = 8
IMG_SIZE = 64
EXPECTED_CACHE_MODE = "bluewhite"
EXPECTED_IMAGE_CHANNELS = 1

BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Si te da problemas con LSTM/CUDA/cuDNN, deja esto en False.
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False

SEED = 42
NUM_WORKERS = 2

MODEL_OUT = "best_multimodal_lstm_weather_img_cache_bluewhite_no_autoreg.pt"
PREDICTIONS_OUT = "eval_predictions_multimodal_lstm_weather_img_cache_bluewhite_no_autoreg.tsv"
PLOT_OUT = "eval_timeline_multimodal_lstm_weather_img_cache_bluewhite_no_autoreg.png"

# Early stopping: mantengo la variable de patience del multimodal.
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MIN_DELTA = 1e-4

# Para MAPE: ignora valores reales bajos.
# Con PV, MAPE con producción muy baja exagera muchísimo errores pequeños.
MAPE_MIN_Y = 1000.0

# Capacidad de referencia para nMAE/nRMSE.
# Se calcula en main() como percentil 99.5 de production.
CAPACITY = None
CAPACITY_PERCENTILE = 99.5

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

# Embeddings simétricos tabular/imagen.
TAB_EMB_DIM = 128
IMG_EMB_DIM = 128

# ============================================================
# Cross-validation temporal por escenas/folds
# ============================================================

N_FOLDS = 5

CV_SUMMARY_OUT = "cv5_temporal_scene_metrics_multimodal_cache_bluewhite_no_autoreg.tsv"
CV_ALL_PREDICTIONS_OUT = "cv5_temporal_all_predictions_multimodal_cache_bluewhite_no_autoreg.tsv"
CV_PRODUCTION_SCENE_METRICS_OUT = "cv5_production_scene_metrics_multimodal_cache_bluewhite_no_autoreg.tsv"


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
# Cache de imágenes
# ============================================================

def load_image_cache(cache_path):
    print()
    print(f"Leyendo cache de imágenes: {cache_path}")

    if not cache_path.exists():
        raise RuntimeError(
            f"No existe el cache de imágenes: {cache_path}.\n"
            "Créalo antes con algo como:\n"
            "python create_images_cache.py --tsv data/weather_with_images.tsv "
            "--out data/images_cache.pt --root . --image-size 64 --mode bluewhite"
        )

    cache = torch.load(
        cache_path,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = ["images", "path_to_idx", "image_size", "channels", "mode"]
    for key in required_keys:
        if key not in cache:
            raise RuntimeError(f"El cache no tiene la clave obligatoria: {key}")

    images = cache["images"]
    path_to_idx = cache["path_to_idx"]
    image_size = int(cache["image_size"])
    channels = int(cache["channels"])
    mode = str(cache["mode"])

    if not torch.is_tensor(images):
        raise RuntimeError("cache['images'] no es un tensor torch.")

    if images.dtype != torch.uint8:
        raise RuntimeError(
            f"Se esperaba cache['images'] dtype=torch.uint8, pero es {images.dtype}."
        )

    if images.ndim != 4:
        raise RuntimeError(
            f"Se esperaba cache['images'] con shape [N,C,H,W], pero tiene {tuple(images.shape)}."
        )

    if mode != EXPECTED_CACHE_MODE:
        raise RuntimeError(
            f"El cache está en modo '{mode}', pero este script espera '{EXPECTED_CACHE_MODE}'.\n"
            "Recréalo con: --mode bluewhite"
        )

    if channels != EXPECTED_IMAGE_CHANNELS or images.shape[1] != EXPECTED_IMAGE_CHANNELS:
        raise RuntimeError(
            f"El cache tiene {channels} canales / tensor C={images.shape[1]}, "
            f"pero este script espera {EXPECTED_IMAGE_CHANNELS} canal."
        )

    if image_size != IMG_SIZE or images.shape[2] != IMG_SIZE or images.shape[3] != IMG_SIZE:
        raise RuntimeError(
            f"El cache tiene image_size={image_size} y tensor {tuple(images.shape)}, "
            f"pero este script espera IMG_SIZE={IMG_SIZE}."
        )

    if len(path_to_idx) == 0:
        raise RuntimeError("cache['path_to_idx'] está vacío.")

    print("Cache OK:")
    print(f"  images: {tuple(images.shape)}")
    print(f"  dtype: {images.dtype}")
    print(f"  mode: {mode}")
    print(f"  channels: {channels}")
    print(f"  image_size: {image_size}")
    print(f"  rutas cacheadas: {len(path_to_idx)}")

    return cache


def normalize_path_str(p):
    if not isinstance(p, str):
        return ""
    return p.strip()


def get_cache_idx_for_path(path_str, path_to_idx):
    """
    Busca image_path en el cache.

    Primero prueba coincidencia exacta, que es lo normal porque create_images_cache.py
    guarda las rutas tal como aparecen en el TSV. Añade un fallback suave para casos
    donde aparezca './data/...' en un lado y 'data/...' en el otro.
    """
    p = normalize_path_str(path_str)

    if p == "":
        return None

    if p in path_to_idx:
        return int(path_to_idx[p])

    if p.startswith("./"):
        p2 = p[2:]
        if p2 in path_to_idx:
            return int(path_to_idx[p2])
    else:
        p2 = "./" + p
        if p2 in path_to_idx:
            return int(path_to_idx[p2])

    return None


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
# Construcción rápida de samples multimodales desde cache
# ============================================================

def build_multimodal_samples_from_cache(
    df,
    input_cols,
    window,
    max_image_age_minutes,
    path_to_idx,
):
    """
    Para cada label_idx:
      - y = production[label_idx]
      - X_tab = variables tabulares en las W filas anteriores: label_idx-W ... label_idx-1
      - X_img = W imágenes anteriores existentes EN CACHE, nunca la imagen del mismo timestamp actual

    IMPORTANTE:
      production NO está en input_cols.
      Las imágenes NO se leen de disco; se exige que image_path esté en path_to_idx.
    """
    print()
    print("Construyendo samples multimodales rápidos desde cache...", flush=True)

    samples = []
    previous_images = deque()

    n = len(df)
    max_age_ns = pd.Timedelta(minutes=max_image_age_minutes).value

    timestamps_ns = df["timestamp"].astype("int64").to_numpy()

    image_paths = (
        df["image_path"]
        .fillna("")
        .astype(str)
        .str.strip()
        .to_numpy()
    )

    image_cache_indices = np.full(n, -1, dtype=np.int64)

    for i, p in enumerate(image_paths):
        cache_idx = get_cache_idx_for_path(p, path_to_idx)
        if cache_idx is not None:
            image_cache_indices[i] = cache_idx

    has_cached_image = image_cache_indices >= 0

    x_values = df[input_cols].values.astype(np.float32)
    y_values = df["production"].values.astype(np.float32)

    # Mantengo la lógica usada en los scripts anteriores:
    # para una fila tabular válida se exige X finito y production finita.
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
            image_cache_idx_window = image_cache_indices[image_indices].astype(np.int64).tolist()

            if min(image_cache_idx_window) < 0:
                raise RuntimeError("Error interno: image_indices contiene imagen no cacheada.")

            tab_indices = list(range(start_idx, end_idx))

            samples.append(
                {
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "label_idx": label_idx,
                    "tab_indices": tab_indices,
                    "image_indices": image_indices,
                    "image_cache_indices": image_cache_idx_window,
                }
            )

        # Añadimos la imagen actual DESPUÉS para evitar usar imagen del mismo timestamp.
        if has_cached_image[label_idx]:
            previous_images.append(label_idx)

    elapsed = time.time() - start_time

    print()
    print("Construcción de samples terminada.", flush=True)
    print(f"  Samples válidos: {len(samples)}", flush=True)
    print(f"  Tiempo: {elapsed:.2f} s ({elapsed/60:.2f} min)", flush=True)

    return samples, has_cached_image


# ============================================================
# Dataset multimodal usando imágenes cacheadas
# ============================================================

class CachedMultimodalPVForecastDataset(Dataset):
    def __init__(
        self,
        df,
        input_cols,
        samples,
        x_min,
        x_range,
        y_min,
        y_range,
        cache_images_uint8,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.input_cols = input_cols
        self.samples = samples

        self.x_min = x_min
        self.x_range = x_range
        self.y_min = y_min
        self.y_range = y_range

        self.cache_images_uint8 = cache_images_uint8

        x_raw = self.df[self.input_cols].values.astype(np.float32)
        y_raw = self.df["production"].values.astype(np.float32)

        self.x = minmax_x(x_raw, self.x_min, self.x_range).astype(np.float32)
        self.y = minmax_y(y_raw, self.y_min, self.y_range).astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        tab_indices = sample["tab_indices"]
        image_cache_indices = sample["image_cache_indices"]
        label_idx = sample["label_idx"]

        x_tab = self.x[tab_indices]
        y = self.y[label_idx]

        # imgs uint8: [W, 1, IMG_SIZE, IMG_SIZE]
        imgs = self.cache_images_uint8[image_cache_indices]

        # Mismo efecto que ToTensor() + Normalize(mean=[0.5], std=[0.5]):
        # uint8 0..255 -> float 0..1 -> float -1..1
        imgs = imgs.to(dtype=torch.float32).div(255.0)
        imgs = (imgs - 0.5) / 0.5

        return (
            torch.from_numpy(x_tab.copy()).float(),
            imgs,
            torch.tensor(y, dtype=torch.float32),
        )


# ============================================================
# Modelo
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


class GentleImageEncoder(nn.Module):
    def __init__(
        self,
        img_emb_dim=128,
        lstm_hidden=128,
        lstm_layers=2,
        lstm_dropout=0.2,
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

            nn.AdaptiveAvgPool2d((2, 2)),
        )

        self.lstm = nn.LSTM(
            input_size=32 * 2 * 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
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
        x = x.reshape(b, t, 32 * 2 * 2)

        out, _ = self.lstm(x)
        last = out[:, -1, :]

        img_emb = self.proj(last)
        return img_emb


class GentleMultimodalPVModel(nn.Module):
    def __init__(
        self,
        num_features,
        tab_emb_dim=128,
        img_emb_dim=128,
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
            lstm_hidden=128,
            lstm_layers=2,
            lstm_dropout=0.2,
        )

        # Proyectamos imagen al espacio tabular.
        self.img_to_tab = nn.Sequential(
            nn.Linear(img_emb_dim, tab_emb_dim),
            nn.Tanh(),
        )

        # Puerta aprendible: deja usar la imagen, pero evita que domine de golpe.
        self.img_gate = nn.Sequential(
            nn.Linear(tab_emb_dim + img_emb_dim, tab_emb_dim),
            nn.Sigmoid(),
        )

        fusion_dim = tab_emb_dim * 3

        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(64, 1),
        )

    def forward(self, x_tab, imgs):
        tab_emb = self.tab_encoder(x_tab)
        img_emb = self.img_encoder(imgs)

        img_proj = self.img_to_tab(img_emb)

        gate_input = torch.cat([tab_emb, img_emb], dim=1)
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
# Métricas
# ============================================================

def compute_metrics_from_arrays(y_true, y_pred, capacity):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if len(y_true) == 0:
        return {
            "n_samples": 0,
            "mean_y_true": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "mae_pct_mean": float("nan"),
            "rmse_pct_mean": float("nan"),
            "nmae_capacity_pct": float("nan"),
            "nrmse_capacity_pct": float("nan"),
            "mape_1000_pct": float("nan"),
            "mape_10pct_capacity_pct": float("nan"),
        }

    errors = y_pred - y_true

    mae = float(np.mean(np.abs(errors)))
    rmse = float(math.sqrt(np.mean(errors ** 2)))

    mean_y_abs = float(np.mean(np.abs(y_true)))

    if mean_y_abs > EPS:
        mae_rel_pct = 100.0 * mae / mean_y_abs
        rmse_rel_pct = 100.0 * rmse / mean_y_abs
    else:
        mae_rel_pct = float("nan")
        rmse_rel_pct = float("nan")

    if capacity is not None and np.isfinite(capacity) and capacity > EPS:
        nmae_capacity_pct = 100.0 * mae / capacity
        nrmse_capacity_pct = 100.0 * rmse / capacity
        threshold_10cap = 0.10 * capacity
    else:
        nmae_capacity_pct = float("nan")
        nrmse_capacity_pct = float("nan")
        threshold_10cap = float("nan")

    mask_1000 = np.abs(y_true) > MAPE_MIN_Y

    if mask_1000.sum() > 0:
        mape_1000_pct = float(
            np.mean(
                np.abs(
                    (y_pred[mask_1000] - y_true[mask_1000])
                    / y_true[mask_1000]
                )
            ) * 100.0
        )
    else:
        mape_1000_pct = float("nan")

    if np.isfinite(threshold_10cap):
        mask_10cap = np.abs(y_true) > threshold_10cap
    else:
        mask_10cap = np.zeros_like(y_true, dtype=bool)

    if mask_10cap.sum() > 0:
        mape_10pct_capacity_pct = float(
            np.mean(
                np.abs(
                    (y_pred[mask_10cap] - y_true[mask_10cap])
                    / y_true[mask_10cap]
                )
            ) * 100.0
        )
    else:
        mape_10pct_capacity_pct = float("nan")

    return {
        "n_samples": int(len(y_true)),
        "mean_y_true": mean_y_abs,
        "mae": mae,
        "rmse": rmse,
        "mae_pct_mean": mae_rel_pct,
        "rmse_pct_mean": rmse_rel_pct,
        "nmae_capacity_pct": nmae_capacity_pct,
        "nrmse_capacity_pct": nrmse_capacity_pct,
        "mape_1000_pct": mape_1000_pct,
        "mape_10pct_capacity_pct": mape_10pct_capacity_pct,
    }


def assign_production_scene(y_true, capacity):
    threshold_1000 = MAPE_MIN_Y
    threshold_10cap = 0.10 * capacity
    threshold_50cap = 0.50 * capacity

    if y_true < threshold_1000:
        return "01_noche_o_muy_baja_<1000"

    if y_true < threshold_10cap:
        return "02_baja_1000_a_10pct_capacity"

    if y_true < threshold_50cap:
        return "03_media_10pct_a_50pct_capacity"

    return "04_alta_mayor_50pct_capacity"


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
def evaluate(model, loader, criterion, y_min, y_range, capacity, return_arrays=False):
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

    arr_metrics = compute_metrics_from_arrays(
        y_true=y_all,
        y_pred=preds_all,
        capacity=capacity,
    )

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "mae": arr_metrics["mae"],
        "rmse": arr_metrics["rmse"],
        "mae_rel_pct": arr_metrics["mae_pct_mean"],
        "rmse_rel_pct": arr_metrics["rmse_pct_mean"],
        "mean_y_abs": arr_metrics["mean_y_true"],
        "capacity": capacity,
        "nmae_capacity_pct": arr_metrics["nmae_capacity_pct"],
        "nrmse_capacity_pct": arr_metrics["nrmse_capacity_pct"],
        "mape_pct": arr_metrics["mape_1000_pct"],
        "mape_1000_pct": arr_metrics["mape_1000_pct"],
        "mape_min_y": MAPE_MIN_Y,
        "mape_10pct_capacity_pct": arr_metrics["mape_10pct_capacity_pct"],
        "mape_10pct_capacity_threshold": 0.10 * capacity if capacity > EPS else float("nan"),
    }

    if return_arrays:
        metrics["preds"] = preds_all
        metrics["y_true"] = y_all

    return metrics


@torch.no_grad()
def evaluate_mean_train_baseline(loader, y_min, y_range, train_y_mean_real, capacity):
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

    arr_metrics = compute_metrics_from_arrays(
        y_true=y_all,
        y_pred=preds_all,
        capacity=capacity,
    )

    print()
    print("Baseline media train:")
    print(f"  MAE={arr_metrics['mae']:.2f}")
    print(f"  RMSE={arr_metrics['rmse']:.2f}")
    print(f"  MAE% mean={arr_metrics['mae_pct_mean']:.2f}%")
    print(f"  RMSE% mean={arr_metrics['rmse_pct_mean']:.2f}%")
    print(f"  nMAE/capacity={arr_metrics['nmae_capacity_pct']:.2f}%")
    print(f"  nRMSE/capacity={arr_metrics['nrmse_capacity_pct']:.2f}%")
    print(f"  MAPE@>{MAPE_MIN_Y:.0f}={arr_metrics['mape_1000_pct']:.2f}%")
    print(
        f"  MAPE@>10%capacity "
        f"(>{0.10 * capacity:.2f})={arr_metrics['mape_10pct_capacity_pct']:.2f}%"
    )

    return {
        "mae": arr_metrics["mae"],
        "rmse": arr_metrics["rmse"],
        "mae_rel_pct": arr_metrics["mae_pct_mean"],
        "rmse_rel_pct": arr_metrics["rmse_pct_mean"],
        "mean_y_abs": arr_metrics["mean_y_true"],
        "capacity": capacity,
        "nmae_capacity_pct": arr_metrics["nmae_capacity_pct"],
        "nrmse_capacity_pct": arr_metrics["nrmse_capacity_pct"],
        "mape_pct": arr_metrics["mape_1000_pct"],
        "mape_1000_pct": arr_metrics["mape_1000_pct"],
        "mape_min_y": MAPE_MIN_Y,
        "mape_10pct_capacity_pct": arr_metrics["mape_10pct_capacity_pct"],
        "mape_10pct_capacity_threshold": 0.10 * capacity if capacity > EPS else float("nan"),
    }


# ============================================================
# Plots y tablas finales
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


def save_production_scene_metrics_table(pred_df, out_path):
    if "capacity" not in pred_df.columns:
        raise RuntimeError("pred_df no tiene columna 'capacity'.")

    capacity = float(pred_df["capacity"].iloc[0])
    pred_df = pred_df.copy()

    pred_df["production_scene"] = pred_df["y_true"].apply(
        lambda y: assign_production_scene(
            y_true=float(y),
            capacity=capacity,
        )
    )

    rows = []

    # Métricas por escena física de producción, agregando todos los folds.
    for scene, scene_df in pred_df.groupby("production_scene", sort=True):
        metrics = compute_metrics_from_arrays(
            y_true=scene_df["y_true"].values,
            y_pred=scene_df["y_pred"].values,
            capacity=capacity,
        )

        rows.append(
            {
                "group": "all_folds",
                "fold": "all",
                "production_scene": scene,
                "capacity": capacity,
                **metrics,
            }
        )

    global_metrics = compute_metrics_from_arrays(
        y_true=pred_df["y_true"].values,
        y_pred=pred_df["y_pred"].values,
        capacity=capacity,
    )

    rows.append(
        {
            "group": "all_folds",
            "fold": "all",
            "production_scene": "99_global",
            "capacity": capacity,
            **global_metrics,
        }
    )

    # Métricas por fold temporal y escena física.
    for fold, fold_df in pred_df.groupby("fold", sort=True):
        for scene, scene_df in fold_df.groupby("production_scene", sort=True):
            metrics = compute_metrics_from_arrays(
                y_true=scene_df["y_true"].values,
                y_pred=scene_df["y_pred"].values,
                capacity=capacity,
            )

            rows.append(
                {
                    "group": "by_fold",
                    "fold": fold,
                    "production_scene": scene,
                    "capacity": capacity,
                    **metrics,
                }
            )

    scene_metrics_df = pd.DataFrame(rows)
    scene_metrics_df.to_csv(out_path, sep="\t", index=False)

    print()
    print("Tabla de métricas por escena física de producción:")
    print(
        scene_metrics_df[
            [
                "group",
                "fold",
                "production_scene",
                "n_samples",
                "mean_y_true",
                "mae",
                "rmse",
                "mae_pct_mean",
                "rmse_pct_mean",
                "nmae_capacity_pct",
                "nrmse_capacity_pct",
                "mape_1000_pct",
                "mape_10pct_capacity_pct",
            ]
        ].to_string(index=False)
    )

    print(f"Tabla de métricas por escena física guardada en: {out_path}")

    return scene_metrics_df


def make_fold_paths(fold_id):
    fold_tag = f"fold_{fold_id:02d}"

    model_out = f"{fold_tag}_{MODEL_OUT}"
    predictions_out = f"{fold_tag}_{PREDICTIONS_OUT}"
    plot_out = f"{fold_tag}_{PLOT_OUT}"

    return model_out, predictions_out, plot_out


def add_mean_std_rows(summary_df):
    metric_cols = [
        "best_eval_mae",
        "best_eval_rmse",
        "best_eval_mae_pct_mean",
        "best_eval_rmse_pct_mean",
        "best_eval_nmae_capacity_pct",
        "best_eval_nrmse_capacity_pct",
        "best_eval_mape_1000_pct",
        "best_eval_mape_10pct_capacity_pct",
        "baseline_mae",
        "baseline_rmse",
        "baseline_mae_pct_mean",
        "baseline_rmse_pct_mean",
        "baseline_nmae_capacity_pct",
        "baseline_nrmse_capacity_pct",
        "baseline_mape_1000_pct",
        "baseline_mape_10pct_capacity_pct",
    ]

    mean_row = {
        "fold": "mean",
        "temporal_scene": "mean",
        "eval_start": "",
        "eval_end": "",
        "train_samples": summary_df["train_samples"].mean(),
        "eval_samples": summary_df["eval_samples"].mean(),
        "capacity": summary_df["capacity"].mean(),
        "best_epoch": summary_df["best_epoch"].mean(),
    }

    std_row = {
        "fold": "std",
        "temporal_scene": "std",
        "eval_start": "",
        "eval_end": "",
        "train_samples": summary_df["train_samples"].std(ddof=1),
        "eval_samples": summary_df["eval_samples"].std(ddof=1),
        "capacity": summary_df["capacity"].std(ddof=1),
        "best_epoch": summary_df["best_epoch"].std(ddof=1),
    }

    for col in metric_cols:
        mean_row[col] = summary_df[col].mean()
        std_row[col] = summary_df[col].std(ddof=1)

    stats_df = pd.DataFrame([mean_row, std_row])
    summary_with_stats = pd.concat([summary_df, stats_df], ignore_index=True)

    return summary_with_stats


# ============================================================
# Fold temporal
# ============================================================

def run_one_temporal_fold(
    fold_id,
    df,
    input_cols,
    all_samples,
    train_samples,
    eval_samples,
    cache_images_uint8,
    capacity,
):
    print()
    print("=" * 80)
    print(f"ESCENA TEMPORAL / FOLD {fold_id:02d}/{N_FOLDS}")
    print("=" * 80)

    if len(train_samples) == 0:
        raise RuntimeError(f"Fold {fold_id}: no hay samples de entrenamiento.")

    if len(eval_samples) == 0:
        raise RuntimeError(f"Fold {fold_id}: no hay samples de evaluación.")

    model_out, predictions_out, plot_out = make_fold_paths(fold_id)

    train_label_indices = [s["label_idx"] for s in train_samples]
    eval_label_indices = [s["label_idx"] for s in eval_samples]

    train_scaler_df = df.iloc[train_label_indices].reset_index(drop=True)

    x_min, x_range, _, _ = fit_minmax(
        train_scaler_df,
        input_cols,
    )

    # Target en unidades de capacidad, no MinMax por fold.
    # Usa la capacidad real si la sabes. Si no, usa CAPACITY calculada.
    y_min = 0.0
    y_range = float(capacity)

    train_dataset = CachedMultimodalPVForecastDataset(
        df=df,
        input_cols=input_cols,
        samples=train_samples,
        x_min=x_min,
        x_range=x_range,
        y_min=y_min,
        y_range=y_range,
        cache_images_uint8=cache_images_uint8,
    )

    eval_dataset = CachedMultimodalPVForecastDataset(
        df=df,
        input_cols=input_cols,
        samples=eval_samples,
        x_min=x_min,
        x_range=x_range,
        y_min=y_min,
        y_range=y_range,
        cache_images_uint8=cache_images_uint8,
    )

    eval_timestamps = df.loc[eval_label_indices, "timestamp"].values
    train_timestamps = df.loc[train_label_indices, "timestamp"].values

    eval_start = pd.to_datetime(eval_timestamps.min())
    eval_end = pd.to_datetime(eval_timestamps.max())
    train_start = pd.to_datetime(train_timestamps.min())
    train_end = pd.to_datetime(train_timestamps.max())

    print()
    print("Diagnóstico fold temporal:")
    print(f"  Samples totales: {len(all_samples)}")
    print(f"  Samples train: {len(train_dataset)}")
    print(f"  Samples eval: {len(eval_dataset)}")
    print(f"  Train desde: {train_start} hasta {train_end}")
    print(f"  Eval  desde: {eval_start} hasta {eval_end}")
    print(f"  y_min train: {y_min:.4f}")
    print(f"  y_range train: {y_range:.4f}")
    print(f"  y_max train: {y_min + y_range:.4f}")
    print(f"  x_min train: {x_min}")
    print(f"  x_range train: {x_range}")

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
        capacity=capacity,
    )

    # Semilla distinta pero reproducible por fold.
    set_seed(SEED + fold_id)

    model = GentleMultimodalPVModel(
        num_features=len(input_cols),
        tab_emb_dim=TAB_EMB_DIM,
        img_emb_dim=IMG_EMB_DIM,
    ).to(DEVICE)

    print()
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
    best_metrics = None
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        print()
        print(f"Fold {fold_id:02d} | Epoch {epoch:03d}/{EPOCHS}")
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
            capacity=capacity,
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
            f"Fold {fold_id:02d} | "
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
            best_metrics = metrics.copy()
            epochs_without_improvement = 0

            best_metrics_to_save = {
                k: v
                for k, v in best_metrics.items()
                if k not in ["preds", "y_true"]
            }

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
                    "image_cache_path": str(IMAGE_CACHE_PATH),
                    "image_cache_mode": EXPECTED_CACHE_MODE,
                    "image_channels": EXPECTED_IMAGE_CHANNELS,
                    "model_type": "gentle_multimodal_lstm_weather_img_cache_bluewhite_no_autoreg_cv5_temporal",
                    "fold_id": fold_id,
                    "n_folds": N_FOLDS,
                    "uses_production_as_input": False,
                    "uses_images_as_input": True,
                    "uses_cached_images": True,
                    "fusion": "concat_tab_imgsoft_mul",
                    "tab_emb_dim": TAB_EMB_DIM,
                    "img_emb_dim": IMG_EMB_DIM,
                    "img_lstm_hidden": 128,
                    "img_lstm_layers": 2,
                    "img_lstm_dropout": 0.2,
                    "max_image_age_minutes": MAX_IMAGE_AGE_MINUTES,
                    "capacity": capacity,
                    "capacity_percentile": CAPACITY_PERCENTILE,
                    "best_epoch": best_epoch,
                    "best_rmse": best_rmse,
                    "best_metrics": best_metrics_to_save,
                    "mean_baseline_metrics": mean_baseline_metrics,
                    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                    "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
                    "mape_min_y": MAPE_MIN_Y,
                    "min_production_for_sample": MIN_PRODUCTION_FOR_SAMPLE,
                    "train_sample_count": len(train_samples),
                    "eval_sample_count": len(eval_samples),
                    "eval_start": str(eval_start),
                    "eval_end": str(eval_end),
                },
                model_out,
            )

            print(
                f"  Nuevo mejor modelo fold {fold_id:02d}: "
                f"{model_out} | "
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
    print(f"Cargando mejor modelo del fold {fold_id:02d} para guardar predicciones...")

    checkpoint = torch.load(
        model_out,
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
        capacity=capacity,
        return_arrays=True,
    )

    pred_df = pd.DataFrame(
        {
            "fold": fold_id,
            "temporal_scene": f"fold_{fold_id:02d}",
            "eval_start": str(eval_start),
            "eval_end": str(eval_end),
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

    pred_df.to_csv(predictions_out, sep="\t", index=False)
    print(f"Predicciones fold {fold_id:02d} guardadas en: {predictions_out}")

    save_eval_timeline_plot(
        timestamps=eval_timestamps,
        y_true=final_metrics["y_true"],
        y_pred=final_metrics["preds"],
        out_path=plot_out,
    )

    if best_metrics is None:
        best_metrics = final_metrics

    result = {
        "fold": fold_id,
        "temporal_scene": f"fold_{fold_id:02d}",
        "eval_start": str(eval_start),
        "eval_end": str(eval_end),
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "capacity": float(capacity),
        "best_epoch": best_epoch,

        "best_eval_loss": best_metrics["loss"],
        "best_eval_mae": best_metrics["mae"],
        "best_eval_rmse": best_metrics["rmse"],
        "best_eval_mae_pct_mean": best_metrics["mae_rel_pct"],
        "best_eval_rmse_pct_mean": best_metrics["rmse_rel_pct"],
        "best_eval_nmae_capacity_pct": best_metrics["nmae_capacity_pct"],
        "best_eval_nrmse_capacity_pct": best_metrics["nrmse_capacity_pct"],
        "best_eval_mape_1000_pct": best_metrics["mape_1000_pct"],
        "best_eval_mape_10pct_capacity_pct": best_metrics["mape_10pct_capacity_pct"],
        "best_eval_mean_y_abs": best_metrics["mean_y_abs"],

        "baseline_mae": mean_baseline_metrics["mae"],
        "baseline_rmse": mean_baseline_metrics["rmse"],
        "baseline_mae_pct_mean": mean_baseline_metrics["mae_rel_pct"],
        "baseline_rmse_pct_mean": mean_baseline_metrics["rmse_rel_pct"],
        "baseline_nmae_capacity_pct": mean_baseline_metrics["nmae_capacity_pct"],
        "baseline_nrmse_capacity_pct": mean_baseline_metrics["nrmse_capacity_pct"],
        "baseline_mape_1000_pct": mean_baseline_metrics["mape_1000_pct"],
        "baseline_mape_10pct_capacity_pct": mean_baseline_metrics["mape_10pct_capacity_pct"],

        "model_out": model_out,
        "predictions_out": predictions_out,
        "plot_out": plot_out,
    }

    print()
    print(f"Fold {fold_id:02d} terminado.")
    print(f"  Eval: {eval_start} -> {eval_end}")
    print(f"  Mejor epoch: {best_epoch}")
    print(f"  Mejor RMSE eval: {best_rmse:.2f}")
    print(f"  Modelo guardado en: {model_out}")
    print(f"  Predicciones guardadas en: {predictions_out}")
    print(f"  Plot guardado en: {plot_out}")

    return result, pred_df


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)

    print(f"Usando dispositivo: {DEVICE}")
    print(f"Leyendo: {TSV_PATH}")
    print(f"Cache imágenes: {IMAGE_CACHE_PATH}")
    print(f"W: {W}")
    print(f"IMG_SIZE: {IMG_SIZE}")
    print(f"N_FOLDS: {N_FOLDS}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"EARLY_STOPPING_PATIENCE: {EARLY_STOPPING_PATIENCE}")
    print(f"MAX_IMAGE_AGE_MINUTES: {MAX_IMAGE_AGE_MINUTES}")
    print("Modo: multimodal weather + imágenes cacheadas 1 canal bluewhite")
    print("Validación: 5-fold cross-validation por bloques temporales ordenados")
    print("Cada escena temporal es un 20% consecutivo de los samples.")
    print("production como input: NO")
    print("Carga imágenes desde disco durante entrenamiento: NO")

    cache = load_image_cache(IMAGE_CACHE_PATH)
    cache_images_uint8 = cache["images"]
    path_to_idx = cache["path_to_idx"]

    df = pd.read_csv(TSV_PATH, sep="\t")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    if "image_path" not in df.columns:
        raise RuntimeError("No existe la columna obligatoria 'image_path'.")

    df["image_path"] = df["image_path"].fillna("").astype(str).str.strip()

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

    cached_flags = df["image_path"].apply(
        lambda p: get_cache_idx_for_path(p, path_to_idx) is not None
    )

    print()
    print(f"Filas originales: {before}")
    print(f"Filas tras dropna numérico: {after}")
    print(f"Filas eliminadas: {before - after}")
    print(f"Filas con image_path vacío: {(df['image_path'].str.strip() == '').sum()}")
    print(f"Filas con image_path en cache: {cached_flags.sum()}")
    print(f"Filas con image_path NO vacío pero fuera de cache: {((df['image_path'].str.strip() != '') & (~cached_flags)).sum()}")
    print(f"W: {W}")
    print(f"N_FOLDS: {N_FOLDS}")
    print(f"MAX_IMAGE_AGE_MINUTES: {MAX_IMAGE_AGE_MINUTES}")
    print(f"MIN_PRODUCTION_FOR_SAMPLE: {MIN_PRODUCTION_FOR_SAMPLE}")

    print_correlations(df, input_cols)

    all_samples, has_cached_image = build_multimodal_samples_from_cache(
        df=df,
        input_cols=input_cols,
        window=W,
        max_image_age_minutes=MAX_IMAGE_AGE_MINUTES,
        path_to_idx=path_to_idx,
    )

    if len(all_samples) == 0:
        raise RuntimeError(
            "No hay samples multimodales válidos. "
            "Revisa image_path, cache, W, MAX_IMAGE_AGE_MINUTES o MIN_PRODUCTION_FOR_SAMPLE."
        )

    if len(all_samples) < N_FOLDS:
        raise RuntimeError(
            f"Hay menos samples ({len(all_samples)}) que folds ({N_FOLDS})."
        )

    print()
    print("Diagnóstico samples global:")
    print(f"Samples multimodales válidos totales: {len(all_samples)}")
    print(f"Filas con imagen cacheada: {int(has_cached_image.sum())}")

    first = all_samples[0]
    print()
    print("Primer sample global:")
    print(f"  Ventana tabular: [{first['start_idx']}, {first['end_idx']})")
    print(f"  Label idx: {first['label_idx']}")
    print(f"  Label timestamp: {df.loc[first['label_idx'], 'timestamp']}")
    print(f"  Label production: {df.loc[first['label_idx'], 'production']:.2f}")
    print("  Imágenes anteriores cacheadas:")
    for idx, cache_idx in zip(first["image_indices"], first["image_cache_indices"]):
        age = df.loc[first["label_idx"], "timestamp"] - df.loc[idx, "timestamp"]
        print(
            f"    row={idx} | cache_idx={cache_idx} | "
            f"{df.loc[idx, 'timestamp']} | "
            f"age={age} | "
            f"{df.loc[idx, 'image_path']}"
        )

    # all_samples ya está ordenado temporalmente porque df está ordenado por timestamp.
    # Dividimos en 5 bloques consecutivos: las escenas temporales.
    sample_indices = np.arange(len(all_samples))
    fold_indices = np.array_split(sample_indices, N_FOLDS)

    cv_results = []
    all_pred_dfs = []

    for fold_zero_idx, eval_sample_indices in enumerate(fold_indices):
        fold_id = fold_zero_idx + 1

        train_sample_indices = np.concatenate(
            [
                fold_indices[i]
                for i in range(N_FOLDS)
                if i != fold_zero_idx
            ]
        )

        train_sample_indices = np.sort(train_sample_indices)
        eval_sample_indices = np.sort(eval_sample_indices)

        train_samples = [all_samples[i] for i in train_sample_indices]
        eval_samples = [all_samples[i] for i in eval_sample_indices]

        result, pred_df = run_one_temporal_fold(
            fold_id=fold_id,
            df=df,
            input_cols=input_cols,
            all_samples=all_samples,
            train_samples=train_samples,
            eval_samples=eval_samples,
            cache_images_uint8=cache_images_uint8,
            capacity=CAPACITY,
        )

        cv_results.append(result)
        all_pred_dfs.append(pred_df)

    summary_df = pd.DataFrame(cv_results)
    summary_with_stats_df = add_mean_std_rows(summary_df)

    summary_with_stats_df.to_csv(
        CV_SUMMARY_OUT,
        sep="\t",
        index=False,
    )

    all_predictions_df = pd.concat(all_pred_dfs, ignore_index=True)
    all_predictions_df = all_predictions_df.sort_values(["timestamp", "fold"])
    all_predictions_df.to_csv(
        CV_ALL_PREDICTIONS_OUT,
        sep="\t",
        index=False,
    )

    save_production_scene_metrics_table(
        pred_df=all_predictions_df,
        out_path=CV_PRODUCTION_SCENE_METRICS_OUT,
    )

    print()
    print("=" * 80)
    print("TABLA FINAL POR ESCENAS TEMPORALES / 5-FOLD CV")
    print("=" * 80)

    display_cols = [
        "fold",
        "temporal_scene",
        "eval_start",
        "eval_end",
        "eval_samples",
        "best_epoch",
        "best_eval_mae",
        "best_eval_rmse",
        "best_eval_mae_pct_mean",
        "best_eval_rmse_pct_mean",
        "best_eval_nmae_capacity_pct",
        "best_eval_nrmse_capacity_pct",
        "best_eval_mape_1000_pct",
        "best_eval_mape_10pct_capacity_pct",
    ]

    print()
    print(summary_with_stats_df[display_cols].to_string(index=False))

    print()
    print("Definición de métricas:")
    print("  MAE: error absoluto medio en unidades reales.")
    print("  RMSE: penaliza más los errores grandes/picos.")
    print("  MAE% mean: MAE relativo a producción media del fold.")
    print("  RMSE% mean: RMSE relativo a producción media del fold.")
    print("  nMAE/capacity: MAE relativo a CAPACITY.")
    print("  nRMSE/capacity: RMSE relativo a CAPACITY.")
    print(f"  MAPE@>{MAPE_MIN_Y:.0f}: MAPE solo con y_true > {MAPE_MIN_Y:.0f}.")
    print("  MAPE@>10%capacity: MAPE solo con y_true > 10% de CAPACITY.")

    print()
    print("Cross-validation temporal multimodal terminada.")
    print(f"Resumen por escenas temporales guardado en: {CV_SUMMARY_OUT}")
    print(f"Predicciones de todos los folds guardadas en: {CV_ALL_PREDICTIONS_OUT}")
    print(f"Métricas por escena física guardadas en: {CV_PRODUCTION_SCENE_METRICS_OUT}")


if __name__ == "__main__":
    main()
