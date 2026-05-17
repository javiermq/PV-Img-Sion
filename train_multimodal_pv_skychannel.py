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

W = 5
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3

TRAIN_RATIO = 0.8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Si te daba CUDNN_STATUS_NOT_INITIALIZED con LSTM, deja esto en False.
# Mantiene CUDA activa, pero evita cuDNN para las LSTM.
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False

SEED = 42

MAX_IMAGE_AGE_MINUTES = 120

NUM_WORKERS = 2

MODEL_OUT = "best_multimodal_pv_skychannel_prev_images.pt"

# Early stopping
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 1e-4

# Para MAPE: ignora valores reales muy cercanos a 0.
# En PV esto es importante porque de noche o con producción muy baja el % explota.
MAPE_MIN_Y = 100.0

# Quitamos production porque es el label.
# Quitamos windspeed porque no quieres usarlo.
DROP_INPUT_COLUMNS = [
    "timestamp",
    "production",
    "windspeed",
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
# Conversión RGB -> canal azul/verdoso + blanco
# ============================================================

def rgb_to_sky_channel(img):
    """
    Convierte una imagen RGB a un único canal pensado para cielo/nubes.

    Combina:
    - brillo general
    - azul/verdoso frente a rojo
    - blancura, útil para nubes y zonas saturadas/brillantes

    Devuelve una imagen PIL de 1 canal.
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

    channel = (
        0.60 * brightness +
        0.30 * blue_green +
        0.10 * whiteness
    )

    channel = np.clip(channel, 0.0, 1.0)

    return Image.fromarray(
        (channel * 255).astype(np.uint8)
    )


# ============================================================
# Utilidades
# ============================================================

def image_path_exists(p):
    if not isinstance(p, str):
        return False

    p = p.strip()

    if p == "":
        return False

    return Path(p).exists()


def fit_scalers_from_train_rows(df, sensor_cols):
    sensors = df[sensor_cols].values.astype(np.float32)
    y = df["production"].values.astype(np.float32)

    sensor_mean = sensors.mean(axis=0)
    sensor_std = sensors.std(axis=0)
    sensor_std[sensor_std == 0] = 1.0

    y_mean = float(y.mean())
    y_std = float(y.std())

    if y_std == 0:
        y_std = 1.0

    return sensor_mean, sensor_std, y_mean, y_std


def row_has_valid_values(df, row_idx, sensor_cols):
    sensor_values = df.loc[row_idx, sensor_cols].values.astype(np.float32)
    y_value = df.loc[row_idx, "production"]

    if not np.isfinite(sensor_values).all():
        return False

    if not np.isfinite(float(y_value)):
        return False

    return True


def build_all_valid_samples(
    df,
    sensor_cols,
    window=5,
    max_image_age_minutes=120,
):
    samples = []
    previous_image_indices = []

    max_image_age = pd.Timedelta(minutes=max_image_age_minutes)

    has_image = df["image_path"].apply(image_path_exists).values

    for i in range(len(df)):
        image_indices = None

        if len(previous_image_indices) >= window:
            current_ts = df.loc[i, "timestamp"]

            valid_previous = [
                idx for idx in previous_image_indices
                if current_ts - df.loc[idx, "timestamp"] <= max_image_age
            ]

            if len(valid_previous) >= window:
                image_indices = valid_previous[-window:]

        if image_indices is not None:
            if row_has_valid_values(df, i, sensor_cols):
                samples.append(
                    {
                        "label_idx": i,
                        "image_indices": image_indices,
                    }
                )

        # La imagen de la fila actual se añade después,
        # para no usar imagen del mismo timestamp como entrada.
        if has_image[i]:
            previous_image_indices.append(i)

    return samples


# ============================================================
# Dataset
# ============================================================

class PVMultimodalPrevImageDataset(Dataset):
    """
    Dataset multimodal para nowcasting PV.

    Cada sample contiene:
        label_idx
        image_indices

    y = production[label_idx]
    X sensores = sensores de image_indices
    X imágenes = imágenes de image_indices
    """

    def __init__(
        self,
        df,
        sensor_cols,
        samples,
        sensor_mean,
        sensor_std,
        y_mean,
        y_std,
        img_size=64,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.sensor_cols = sensor_cols
        self.samples = samples

        self.sensor_mean = sensor_mean
        self.sensor_std = sensor_std
        self.y_mean = y_mean
        self.y_std = y_std

        sensors = self.df[self.sensor_cols].values.astype(np.float32)
        y = self.df["production"].values.astype(np.float32)

        self.sensors = (sensors - self.sensor_mean) / self.sensor_std
        self.y = (y - self.y_mean) / self.y_std

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5],
            ),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        label_idx = sample["label_idx"]
        image_indices = sample["image_indices"]

        sensor_window = self.sensors[image_indices]
        y = self.y[label_idx]

        image_paths = self.df.iloc[image_indices]["image_path"].tolist()

        imgs = []

        for p in image_paths:
            img = Image.open(p)
            img = rgb_to_sky_channel(img)
            img = self.transform(img)
            imgs.append(img)

        imgs = torch.stack(imgs, dim=0)
        # imgs: [W, 1, 64, 64]

        sensor_window = torch.tensor(sensor_window, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return sensor_window, imgs, y


# ============================================================
# Encoder sensores
# Conv1D + Conv1D + LSTM + LSTM
# ============================================================

class SensorEncoder(nn.Module):
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
        # out: [B, W, 128]

        last = out[:, -1, :]
        emb = self.proj(last)

        return emb


# ============================================================
# Encoder imágenes
# Conv2D + Conv2D + LSTM + LSTM
# ============================================================

class ImageEncoder(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            # 64 -> 32

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            # 32 -> 16

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
        # x: [B*W, 1, 64, 64]

        x = self.cnn(x)
        # x: [B*W, 64, 1, 1]

        x = x.reshape(b, t, 64)
        # x: [B, W, 64]

        out, _ = self.lstm(x)
        # out: [B, W, 128]

        last = out[:, -1, :]
        emb = self.proj(last)

        return emb


# ============================================================
# Modelo multimodal
# ============================================================

class PVMultimodalModel(nn.Module):
    def __init__(self, num_sensor_features, emb_dim=128):
        super().__init__()

        self.sensor_encoder = SensorEncoder(
            num_features=num_sensor_features,
            emb_dim=emb_dim,
        )

        self.image_encoder = ImageEncoder(
            emb_dim=emb_dim,
        )

        fusion_dim = emb_dim * 4

        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1),
        )

    def forward(self, sensors, images):
        sensor_emb = self.sensor_encoder(sensors)
        image_emb = self.image_encoder(images)

        fusion = torch.cat(
            [
                sensor_emb,
                image_emb,
                sensor_emb * image_emb,
                torch.abs(sensor_emb - image_emb),
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

    for batch_idx, (sensors, images, y) in enumerate(loader):
        if batch_idx % 10 == 0:
            print(f"  batch {batch_idx}/{len(loader)}", flush=True)

        sensors = sensors.to(DEVICE, non_blocking=True)
        images = images.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        pred = model(sensors, images)
        loss = criterion(pred, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        total_loss += loss.item() * sensors.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, y_mean, y_std):
    model.eval()

    total_loss = 0.0

    preds_all = []
    y_all = []

    for batch_idx, (sensors, images, y) in enumerate(loader):
        if batch_idx % 10 == 0:
            print(f"  eval batch {batch_idx}/{len(loader)}", flush=True)

        sensors = sensors.to(DEVICE, non_blocking=True)
        images = images.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        pred = model(sensors, images)
        loss = criterion(pred, y)

        total_loss += loss.item() * sensors.size(0)

        pred_real = pred.detach().cpu().numpy() * y_std + y_mean
        y_real = y.detach().cpu().numpy() * y_std + y_mean

        preds_all.append(pred_real)
        y_all.append(y_real)

    preds_all = np.concatenate(preds_all)
    y_all = np.concatenate(y_all)

    errors = preds_all - y_all

    mae = np.mean(np.abs(errors))
    rmse = math.sqrt(np.mean(errors ** 2))

    # Error relativo estable:
    # relativo respecto a la media absoluta de la producción real en eval.
    mean_y_abs = np.mean(np.abs(y_all))

    if mean_y_abs > 1e-8:
        mae_rel_pct = 100.0 * mae / mean_y_abs
        rmse_rel_pct = 100.0 * rmse / mean_y_abs
    else:
        mae_rel_pct = float("nan")
        rmse_rel_pct = float("nan")

    # MAPE con máscara para evitar división por valores muy pequeños.
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

    df = pd.read_csv(TSV_PATH, sep="\t")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    df["image_path"] = df["image_path"].fillna("").astype(str)

    sensor_cols = [
        col for col in df.columns
        if col not in DROP_INPUT_COLUMNS
    ]

    print("Columnas usadas como sensores:")
    for col in sensor_cols:
        print(f"  - {col}")

    all_samples = build_all_valid_samples(
        df=df,
        sensor_cols=sensor_cols,
        window=W,
        max_image_age_minutes=MAX_IMAGE_AGE_MINUTES,
    )

    sample_split_idx = int(len(all_samples) * TRAIN_RATIO)

    train_samples = all_samples[:sample_split_idx]
    eval_samples = all_samples[sample_split_idx:]

    train_label_indices = [s["label_idx"] for s in train_samples]
    train_scaler_df = df.iloc[train_label_indices].reset_index(drop=True)

    sensor_mean, sensor_std, y_mean, y_std = fit_scalers_from_train_rows(
        train_scaler_df,
        sensor_cols,
    )

    train_dataset = PVMultimodalPrevImageDataset(
        df=df,
        sensor_cols=sensor_cols,
        samples=train_samples,
        sensor_mean=sensor_mean,
        sensor_std=sensor_std,
        y_mean=y_mean,
        y_std=y_std,
        img_size=IMG_SIZE,
    )

    eval_dataset = PVMultimodalPrevImageDataset(
        df=df,
        sensor_cols=sensor_cols,
        samples=eval_samples,
        sensor_mean=sensor_mean,
        sensor_std=sensor_std,
        y_mean=y_mean,
        y_std=y_std,
        img_size=IMG_SIZE,
    )

    print()
    print("Diagnóstico:")
    print(f"Filas totales: {len(df)}")
    print(f"Filas con image_path vacío: {(df['image_path'].str.strip() == '').sum()}")
    print(f"Filas con imagen existente: {df['image_path'].apply(image_path_exists).sum()}")
    print(f"MAX_IMAGE_AGE_MINUTES: {MAX_IMAGE_AGE_MINUTES}")
    print(f"Samples válidos totales: {len(all_samples)}")
    print(f"Samples train válidos: {len(train_dataset)}")
    print(f"Samples eval válidos: {len(eval_dataset)}")
    print(f"y_mean train: {y_mean:.4f}")
    print(f"y_std train: {y_std:.4f}")

    if len(train_dataset) > 0:
        s = train_dataset.samples[0]
        print()
        print("Primer sample train:")
        print(f"  Label idx: {s['label_idx']}")
        print(f"  Label timestamp: {df.loc[s['label_idx'], 'timestamp']}")
        print("  Imágenes anteriores:")
        for idx in s["image_indices"]:
            age = df.loc[s["label_idx"], "timestamp"] - df.loc[idx, "timestamp"]
            print(
                f"    {idx} | "
                f"{df.loc[idx, 'timestamp']} | "
                f"age={age} | "
                f"{df.loc[idx, 'image_path']}"
            )

    if len(eval_dataset) > 0:
        s = eval_dataset.samples[0]
        print()
        print("Primer sample eval:")
        print(f"  Label idx: {s['label_idx']}")
        print(f"  Label timestamp: {df.loc[s['label_idx'], 'timestamp']}")
        print("  Imágenes anteriores:")
        for idx in s["image_indices"]:
            age = df.loc[s["label_idx"], "timestamp"] - df.loc[idx, "timestamp"]
            print(
                f"    {idx} | "
                f"{df.loc[idx, 'timestamp']} | "
                f"age={age} | "
                f"{df.loc[idx, 'image_path']}"
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

    model = PVMultimodalModel(
        num_sensor_features=len(sensor_cols),
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
        )

        print("Eval:")

        metrics = evaluate(
            model,
            eval_loader,
            criterion,
            y_mean=y_mean,
            y_std=y_std,
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
                    "sensor_cols": sensor_cols,
                    "sensor_mean": sensor_mean,
                    "sensor_std": sensor_std,
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "window": W,
                    "img_size": IMG_SIZE,
                    "max_image_age_minutes": MAX_IMAGE_AGE_MINUTES,
                    "image_mode": "sky_channel_bluegreen_white",
                    "sample_mode": "previous_W_rows_with_existing_images_limited_age",
                    "drop_input_columns": DROP_INPUT_COLUMNS,
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
