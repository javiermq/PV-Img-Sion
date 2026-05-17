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

# Ventana temporal: usa las W lecturas anteriores para predecir production actual.
# Si tus datos están cada 5 minutos, W=12 equivale aprox. a 1 hora.
W = 5
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3

TRAIN_RATIO = 0.8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ablation principal:
#   False = PV-only: usa solo production histórica, sin imágenes y sin meteorología.
#   True  = PV + imágenes previas, para comparar contra el multimodal.
INCLUDE_IMG = False

# En PV-only no se usan sensores meteorológicos.
# Si algún día quieres PV + meteo, cambia INPUT_MODE a "pv_weather".
INPUT_MODE = "pv_only"  # "pv_only" o "pv_weather"

# Si te daba CUDNN_STATUS_NOT_INITIALIZED con LSTM, deja esto en False.
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False

SEED = 42
MAX_IMAGE_AGE_MINUTES = 120
NUM_WORKERS = 2

MODEL_OUT = (
    "best_pv_only_ablation.pt"
    if not INCLUDE_IMG
    else "best_pv_plus_img_ablation.pt"
)

# Early stopping
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 1e-4

# Para MAPE: ignora valores reales muy cercanos a 0.
MAPE_MIN_Y = 100.0

# Columnas que nunca se usan como sensores meteorológicos.
DROP_INPUT_COLUMNS = [
    "timestamp",
    "production",
    "image_path",
]


# ============================================================
# Reproducibilidad básica
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Imagen: RGB -> canal azul/verdoso + blanco
# Solo se usa si INCLUDE_IMG=True
# ============================================================

def rgb_to_sky_channel(img):
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
# Scalers
# ============================================================

def get_feature_columns(df):
    if INPUT_MODE == "pv_only":
        return ["production"]

    if INPUT_MODE == "pv_weather":
        return [col for col in df.columns if col not in DROP_INPUT_COLUMNS]

    raise ValueError("INPUT_MODE debe ser 'pv_only' o 'pv_weather'.")


def fit_scalers_from_train_rows(df, feature_cols):
    x = df[feature_cols].values.astype(np.float32)
    y = df["production"].values.astype(np.float32)

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std == 0] = 1.0

    y_mean = float(y.mean())
    y_std = float(y.std())
    if y_std == 0:
        y_std = 1.0

    return x_mean, x_std, y_mean, y_std


def row_has_valid_values(df, row_idx, feature_cols):
    x_values = df.loc[row_idx, feature_cols].values.astype(np.float32)
    y_value = df.loc[row_idx, "production"]

    if not np.isfinite(x_values).all():
        return False

    if not np.isfinite(float(y_value)):
        return False

    return True


# ============================================================
# Construcción de samples
# ============================================================

def build_pv_only_samples(df, feature_cols, window=12):
    """
    Sample PV-only:
        X = feature_cols en las W filas anteriores
        y = production en la fila actual

    No exige imágenes. Esto es lo importante para la ablation.
    """
    samples = []

    for label_idx in range(window, len(df)):
        input_indices = list(range(label_idx - window, label_idx))

        valid_input = all(
            row_has_valid_values(df, idx, feature_cols)
            for idx in input_indices
        )
        valid_label = np.isfinite(float(df.loc[label_idx, "production"]))

        if valid_input and valid_label:
            samples.append(
                {
                    "label_idx": label_idx,
                    "input_indices": input_indices,
                    "image_indices": None,
                }
            )

    return samples


def build_img_samples(df, feature_cols, window=12, max_image_age_minutes=120):
    """
    Sample PV + imágenes:
        X sensores = feature_cols en timestamps de las W imágenes anteriores
        X imágenes = W imágenes anteriores existentes
        y = production actual

    Este modo mantiene la lógica antigua para comparar.
    """
    samples = []
    previous_image_indices = []
    max_image_age = pd.Timedelta(minutes=max_image_age_minutes)

    has_image = df["image_path"].apply(image_path_exists).values

    for label_idx in range(len(df)):
        image_indices = None

        if len(previous_image_indices) >= window:
            current_ts = df.loc[label_idx, "timestamp"]

            valid_previous = [
                idx for idx in previous_image_indices
                if current_ts - df.loc[idx, "timestamp"] <= max_image_age
            ]

            if len(valid_previous) >= window:
                image_indices = valid_previous[-window:]

        if image_indices is not None:
            valid_input = all(
                row_has_valid_values(df, idx, feature_cols)
                for idx in image_indices
            )
            valid_label = np.isfinite(float(df.loc[label_idx, "production"]))

            if valid_input and valid_label:
                samples.append(
                    {
                        "label_idx": label_idx,
                        "input_indices": image_indices,
                        "image_indices": image_indices,
                    }
                )

        # La imagen actual se añade después para no usar el mismo timestamp como entrada.
        if has_image[label_idx]:
            previous_image_indices.append(label_idx)

    return samples


def split_samples_by_label_time(samples, split_idx):
    train_samples = [s for s in samples if s["label_idx"] < split_idx]
    eval_samples = [s for s in samples if s["label_idx"] >= split_idx]
    return train_samples, eval_samples


# ============================================================
# Dataset
# ============================================================

class PVAblationDataset(Dataset):
    def __init__(
        self,
        df,
        feature_cols,
        samples,
        x_mean,
        x_std,
        y_mean,
        y_std,
        include_img=False,
        img_size=64,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.feature_cols = feature_cols
        self.samples = samples
        self.include_img = include_img

        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

        x = self.df[self.feature_cols].values.astype(np.float32)
        y = self.df["production"].values.astype(np.float32)

        self.x = (x - self.x_mean) / self.x_std
        self.y = (y - self.y_mean) / self.y_std

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        label_idx = sample["label_idx"]
        input_indices = sample["input_indices"]

        x_window = self.x[input_indices]
        y = self.y[label_idx]

        x_window = torch.tensor(x_window, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        if not self.include_img:
            return x_window, y

        image_indices = sample["image_indices"]
        image_paths = self.df.iloc[image_indices]["image_path"].tolist()

        imgs = []
        for p in image_paths:
            img = Image.open(p)
            img = rgb_to_sky_channel(img)
            img = self.transform(img)
            imgs.append(img)

        imgs = torch.stack(imgs, dim=0)
        # imgs: [W, 1, 64, 64]

        return x_window, imgs, y


# ============================================================
# Encoder serie temporal: Conv1D + Conv1D + LSTM + LSTM
# ============================================================

class SeriesEncoder(nn.Module):
    def __init__(self, num_features, emb_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        self.proj = nn.Sequential(
            nn.Linear(128, emb_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: [B, W, C]
        x = x.transpose(1, 2)
        # x: [B, C, W]

        x = self.conv(x)
        # x: [B, 128, W]

        x = x.transpose(1, 2)
        # x: [B, W, 128]

        out, _ = self.lstm(x)
        last = out[:, -1, :]

        return self.proj(last)


# ============================================================
# Encoder imágenes
# Solo se usa si INCLUDE_IMG=True
# ============================================================

class ImageEncoder(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        self.proj = nn.Sequential(
            nn.Linear(128, emb_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: [B, W, 1, 64, 64]
        b, t, c, h, w = x.shape

        x = x.reshape(b * t, c, h, w)
        x = self.cnn(x)
        x = x.reshape(b, t, 64)

        out, _ = self.lstm(x)
        last = out[:, -1, :]

        return self.proj(last)


# ============================================================
# Modelo con flag de ablation
# ============================================================

class PVAblationModel(nn.Module):
    def __init__(self, num_features, include_img=False, emb_dim=128):
        super().__init__()

        self.include_img = include_img

        self.series_encoder = SeriesEncoder(
            num_features=num_features,
            emb_dim=emb_dim,
        )

        if self.include_img:
            self.image_encoder = ImageEncoder(emb_dim=emb_dim)
            fusion_dim = emb_dim * 4
        else:
            self.image_encoder = None
            fusion_dim = emb_dim

        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(64, 1),
        )

    def forward(self, x_series, images=None):
        series_emb = self.series_encoder(x_series)

        if not self.include_img:
            fusion = series_emb
        else:
            if images is None:
                raise ValueError("images no puede ser None cuando include_img=True")

            image_emb = self.image_encoder(images)
            fusion = torch.cat(
                [
                    series_emb,
                    image_emb,
                    series_emb * image_emb,
                    torch.abs(series_emb - image_emb),
                ],
                dim=1,
            )

        y_hat = self.regressor(fusion).squeeze(1)
        return y_hat


# ============================================================
# Train / Eval
# ============================================================

def unpack_batch(batch, include_img):
    if include_img:
        x_series, images, y = batch
        return x_series, images, y

    x_series, y = batch
    return x_series, None, y


def train_one_epoch(model, loader, optimizer, criterion, include_img=False):
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(loader):
        if batch_idx % 10 == 0:
            print(f"  batch {batch_idx}/{len(loader)}", flush=True)

        x_series, images, y = unpack_batch(batch, include_img)

        x_series = x_series.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        if images is not None:
            images = images.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        pred = model(x_series, images)
        loss = criterion(pred, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * x_series.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, y_mean, y_std, include_img=False):
    model.eval()
    total_loss = 0.0

    preds_all = []
    y_all = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx % 10 == 0:
            print(f"  eval batch {batch_idx}/{len(loader)}", flush=True)

        x_series, images, y = unpack_batch(batch, include_img)

        x_series = x_series.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        if images is not None:
            images = images.to(DEVICE, non_blocking=True)

        pred = model(x_series, images)
        loss = criterion(pred, y)

        total_loss += loss.item() * x_series.size(0)

        pred_real = pred.detach().cpu().numpy() * y_std + y_mean
        y_real = y.detach().cpu().numpy() * y_std + y_mean

        preds_all.append(pred_real)
        y_all.append(y_real)

    preds_all = np.concatenate(preds_all)
    y_all = np.concatenate(y_all)

    errors = preds_all - y_all

    mae = np.mean(np.abs(errors))
    rmse = math.sqrt(np.mean(errors ** 2))

    mean_y_abs = np.mean(np.abs(y_all))

    if mean_y_abs > 1e-8:
        mae_rel_pct = 100.0 * mae / mean_y_abs
        rmse_rel_pct = 100.0 * rmse / mean_y_abs
    else:
        mae_rel_pct = float("nan")
        rmse_rel_pct = float("nan")

    mask = np.abs(y_all) > MAPE_MIN_Y

    if mask.sum() > 0:
        mape_pct = np.mean(
            np.abs((preds_all[mask] - y_all[mask]) / y_all[mask])
        ) * 100.0
    else:
        mape_pct = float("nan")

    return {
        "loss": total_loss / len(loader.dataset),
        "mae": mae,
        "rmse": rmse,
        "mae_rel_pct": mae_rel_pct,
        "rmse_rel_pct": rmse_rel_pct,
        "mape_pct": mape_pct,
        "mean_y_abs": mean_y_abs,
    }


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)

    print(f"Usando dispositivo: {DEVICE}")
    print(f"INCLUDE_IMG: {INCLUDE_IMG}")
    print(f"INPUT_MODE: {INPUT_MODE}")
    print(f"W: {W}")

    df = pd.read_csv(TSV_PATH, sep="\t")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if "image_path" not in df.columns:
        df["image_path"] = ""

    df["image_path"] = df["image_path"].fillna("").astype(str)

    feature_cols = get_feature_columns(df)

    print("Columnas usadas como entrada:")
    for col in feature_cols:
        print(f"  - {col}")

    split_idx = int(len(df) * TRAIN_RATIO)

    if INCLUDE_IMG:
        all_samples = build_img_samples(
            df=df,
            feature_cols=feature_cols,
            window=W,
            max_image_age_minutes=MAX_IMAGE_AGE_MINUTES,
        )
    else:
        all_samples = build_pv_only_samples(
            df=df,
            feature_cols=feature_cols,
            window=W,
        )

    train_samples, eval_samples = split_samples_by_label_time(
        samples=all_samples,
        split_idx=split_idx,
    )

    train_scaler_df = df.iloc[:split_idx].reset_index(drop=True)

    x_mean, x_std, y_mean, y_std = fit_scalers_from_train_rows(
        train_scaler_df,
        feature_cols,
    )

    train_dataset = PVAblationDataset(
        df=df,
        feature_cols=feature_cols,
        samples=train_samples,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        include_img=INCLUDE_IMG,
        img_size=IMG_SIZE,
    )

    eval_dataset = PVAblationDataset(
        df=df,
        feature_cols=feature_cols,
        samples=eval_samples,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        include_img=INCLUDE_IMG,
        img_size=IMG_SIZE,
    )

    print()
    print("Diagnóstico:")
    print(f"Filas totales: {len(df)}")
    print(f"Split index temporal: {split_idx}")
    print(f"Filas train: {split_idx}")
    print(f"Filas eval: {len(df) - split_idx}")
    print(f"Filas con image_path vacío: {(df['image_path'].str.strip() == '').sum()}")
    print(f"Filas con imagen existente: {df['image_path'].apply(image_path_exists).sum()}")
    print(f"Samples válidos totales: {len(all_samples)}")
    print(f"Samples train válidos: {len(train_dataset)}")
    print(f"Samples eval válidos: {len(eval_dataset)}")
    print(f"x_mean train: {x_mean}")
    print(f"x_std train: {x_std}")
    print(f"y_mean train: {y_mean:.4f}")
    print(f"y_std train: {y_std:.4f}")

    if len(train_dataset) > 0:
        s = train_dataset.samples[0]
        print()
        print("Primer sample train:")
        print(f"  Label idx: {s['label_idx']}")
        print(f"  Label timestamp: {df.loc[s['label_idx'], 'timestamp']}")
        print("  Índices entrada anteriores:")
        for idx in s["input_indices"]:
            age = df.loc[s["label_idx"], "timestamp"] - df.loc[idx, "timestamp"]
            print(
                f"    {idx} | "
                f"{df.loc[idx, 'timestamp']} | "
                f"age={age} | "
                f"production={df.loc[idx, 'production']:.2f}"
            )

    if len(eval_dataset) > 0:
        s = eval_dataset.samples[0]
        print()
        print("Primer sample eval:")
        print(f"  Label idx: {s['label_idx']}")
        print(f"  Label timestamp: {df.loc[s['label_idx'], 'timestamp']}")
        print("  Índices entrada anteriores:")
        for idx in s["input_indices"]:
            age = df.loc[s["label_idx"], "timestamp"] - df.loc[idx, "timestamp"]
            print(
                f"    {idx} | "
                f"{df.loc[idx, 'timestamp']} | "
                f"age={age} | "
                f"production={df.loc[idx, 'production']:.2f}"
            )

    if len(train_dataset) == 0:
        raise RuntimeError("No hay samples válidos de entrenamiento.")

    if len(eval_dataset) == 0:
        raise RuntimeError("No hay samples válidos de evaluación.")

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

    model = PVAblationModel(
        num_features=len(feature_cols),
        include_img=INCLUDE_IMG,
        emb_dim=128,
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
            include_img=INCLUDE_IMG,
        )

        print("Eval:")

        metrics = evaluate(
            model,
            eval_loader,
            criterion,
            y_mean=y_mean,
            y_std=y_std,
            include_img=INCLUDE_IMG,
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
                    "feature_cols": feature_cols,
                    "x_mean": x_mean,
                    "x_std": x_std,
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "window": W,
                    "img_size": IMG_SIZE,
                    "include_img": INCLUDE_IMG,
                    "input_mode": INPUT_MODE,
                    "max_image_age_minutes": MAX_IMAGE_AGE_MINUTES,
                    "sample_mode": (
                        "previous_W_rows_pv_only"
                        if not INCLUDE_IMG
                        else "previous_W_images_limited_age"
                    ),
                    "best_epoch": best_epoch,
                    "best_rmse": best_rmse,
                    "best_metrics": metrics,
                    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                    "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
                    "mape_min_y": MAPE_MIN_Y,
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
    print("Entrenamiento terminado.")
    print(f"Mejor epoch: {best_epoch}")
    print(f"Mejor RMSE eval: {best_rmse:.2f}")
    print(f"Modelo guardado en: {MODEL_OUT}")


if __name__ == "__main__":
    main()
