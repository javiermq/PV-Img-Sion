from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch


def print_progress(i, total, start_time):
    pct = 100.0 * i / total
    elapsed = time.time() - start_time

    if i > 0:
        rate = i / elapsed
        remaining = (total - i) / rate if rate > 0 else 0
    else:
        rate = 0
        remaining = 0

    print(
        f"[{i:6d}/{total:6d}] "
        f"{pct:6.2f}% | "
        f"{rate:7.2f} img/s | "
        f"ETA {remaining/60:6.1f} min",
        flush=True,
    )


def rgb_to_bluewhite_channel(img_rgb_uint8):
    """
    Convierte RGB uint8 [H, W, 3] a 1 canal [H, W].

    Canal aproximado:
      - alto en cielo azul/verdoso
      - alto en nubes/blancos
      - bajo en zonas oscuras/rojizas

    Devuelve uint8 [H, W] en rango 0..255.
    """
    x = img_rgb_uint8.astype(np.float32) / 255.0

    r = x[:, :, 0]
    g = x[:, :, 1]
    b = x[:, :, 2]

    # Componente azul/verdosa
    blue_green = 0.55 * b + 0.35 * g - 0.25 * r
    blue_green = np.clip(blue_green, 0.0, 1.0)

    # Componente blanca: píxeles brillantes y poco saturados
    brightness = (r + g + b) / 3.0
    chroma = np.max(x, axis=2) - np.min(x, axis=2)

    white = brightness * (1.0 - chroma)
    white = np.clip(white, 0.0, 1.0)

    # Canal final: cielo azul/verdoso o blanco
    channel = np.maximum(blue_green, white)
    channel = np.clip(channel * 255.0, 0, 255).astype(np.uint8)

    return channel


def load_image(path, image_size, mode):
    with Image.open(path) as im:
        im = im.convert("RGB")

        if im.size != (image_size, image_size):
            im = im.resize((image_size, image_size), Image.BILINEAR)

        arr = np.asarray(im, dtype=np.uint8)

    if mode == "rgb":
        # [H, W, 3] -> [3, H, W]
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr.copy())

    if mode == "bluewhite":
        # [H, W, 3] -> [H, W] -> [1, H, W]
        ch = rgb_to_bluewhite_channel(arr)
        ch = ch[None, :, :]
        return torch.from_numpy(ch.copy())

    raise ValueError(f"Modo no soportado: {mode}")


def save_bluewhite_visual_image(
    path,
    rel_path,
    image_size,
    out_dir,
    base_prefix="",
    ext="png",
):
    """
    Guarda una imagen visual en escala de grises 0..255 del filtro azul/verdoso+blanco.

    Ejemplo:
      rel_path = data/procesadas/07/01/img.jpg
      base_prefix = data/procesadas
      out_dir = data/procesadas_bluewhite

    Salida:
      data/procesadas_bluewhite/07/01/img.png
    """
    with Image.open(path) as im:
        im = im.convert("RGB")

        if im.size != (image_size, image_size):
            im = im.resize((image_size, image_size), Image.BILINEAR)

        arr = np.asarray(im, dtype=np.uint8)

    ch = rgb_to_bluewhite_channel(arr)

    rel = Path(rel_path)

    if base_prefix != "":
        base = Path(base_prefix)
        try:
            rel = rel.relative_to(base)
        except ValueError:
            # Si la ruta no empieza por base_prefix, mantenemos la ruta relativa original.
            pass

    out_path = Path(out_dir) / rel
    out_path = out_path.with_suffix("." + ext.lower())
    out_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(ch, mode="L").save(out_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tsv",
        type=str,
        default="data/weather_with_images.tsv",
        help="TSV con la columna image_path.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="data/images_cache_bluewhite.pt",
        help="Fichero de salida .pt.",
    )

    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Raíz desde la que resolver rutas relativas.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="bluewhite",
        choices=["rgb", "bluewhite"],
        help="rgb = 3 canales; bluewhite = 1 canal azul/verdoso+blanco.",
    )

    parser.add_argument(
        "--visualize-bluewhite-dir",
        type=str,
        default="data/procesadas_bluewhite",
        help=(
            "Si se indica, guarda una carpeta espejo con imágenes en escala de grises "
            "0..255 representando el filtro azul/verdoso+blanco."
        ),
    )

    parser.add_argument(
        "--visualize-bluewhite-base",
        type=str,
        default="data/procesadas",
        help=(
            "Prefijo de ruta a eliminar para crear la carpeta espejo. "
            "Ejemplo: data/procesadas"
        ),
    )

    parser.add_argument(
        "--visualize-bluewhite-ext",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Extensión de las imágenes visuales generadas.",
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    tsv_path = Path(args.tsv)
    out_path = Path(args.out)
    root = Path(args.root)

    print(f"Leyendo TSV: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")

    if "image_path" not in df.columns:
        raise RuntimeError("El TSV no tiene columna image_path.")

    paths = (
        df["image_path"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    paths = [p for p in paths.tolist() if p != ""]
    paths = sorted(set(paths))

    print(f"Rutas image_path únicas no vacías: {len(paths)}")
    print(f"Modo imagen: {args.mode}")
    print(f"Image size: {args.image_size}x{args.image_size}")

    if args.visualize_bluewhite_dir != "":
        print(f"Visualización bluewhite activada: {args.visualize_bluewhite_dir}")
        if args.visualize_bluewhite_base != "":
            print(f"Base espejo visual: {args.visualize_bluewhite_base}")
        print(f"Extensión visual: {args.visualize_bluewhite_ext}")

    existing_paths = []
    missing_paths = []

    print("Comprobando existencia de imágenes...")

    for p in paths:
        full_path = root / p

        if full_path.exists():
            existing_paths.append(p)
        else:
            missing_paths.append(p)

    print(f"Imágenes existentes: {len(existing_paths)}")
    print(f"Imágenes no encontradas: {len(missing_paths)}")

    if len(existing_paths) == 0:
        raise RuntimeError("No se encontró ninguna imagen. Revisa --root y las rutas image_path.")

    if len(missing_paths) > 0:
        print()
        print("Primeras imágenes no encontradas:")
        for p in missing_paths[:10]:
            print(f"  - {p}")

    channels = 3 if args.mode == "rgb" else 1
    n = len(existing_paths)

    images = torch.empty(
        (n, channels, args.image_size, args.image_size),
        dtype=torch.uint8,
    )

    print()
    print("Cargando y convirtiendo imágenes...")
    start_time = time.time()

    good_paths = []
    failed = []

    write_idx = 0

    for i, p in enumerate(existing_paths, start=1):
        full_path = root / p

        try:
            img_tensor = load_image(
                path=full_path,
                image_size=args.image_size,
                mode=args.mode,
            )

            images[write_idx] = img_tensor
            good_paths.append(p)
            write_idx += 1

            if args.visualize_bluewhite_dir != "":
                save_bluewhite_visual_image(
                    path=full_path,
                    rel_path=p,
                    image_size=args.image_size,
                    out_dir=args.visualize_bluewhite_dir,
                    base_prefix=args.visualize_bluewhite_base,
                    ext=args.visualize_bluewhite_ext,
                )

        except Exception as e:
            failed.append((p, str(e)))

        if i == 1 or i % args.progress_every == 0 or i == n:
            print_progress(i, n, start_time)

    images = images[:write_idx].contiguous()

    path_to_idx = {
        p: i for i, p in enumerate(good_paths)
    }

    print()
    print(f"Imágenes cargadas correctamente: {len(good_paths)}")
    print(f"Fallos leyendo/convirtiendo imágenes: {len(failed)}")

    if len(failed) > 0:
        print()
        print("Primeros fallos:")
        for p, err in failed[:10]:
            print(f"  - {p}: {err}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache = {
        "images": images,
        "paths": good_paths,
        "path_to_idx": path_to_idx,
        "image_size": args.image_size,
        "channels": channels,
        "mode": args.mode,
        "dtype": "uint8",
    }

    print()
    print(f"Guardando cache en: {out_path}")
    torch.save(cache, out_path)

    size_mb = out_path.stat().st_size / (1024 ** 2)

    print()
    print("Cache creada correctamente.")
    print(f"Fichero: {out_path}")
    print(f"Tamaño: {size_mb:.2f} MB")
    print(f"Tensor images: {tuple(images.shape)}")
    print(f"dtype: {images.dtype}")

    if args.visualize_bluewhite_dir != "":
        print()
        print("Carpeta espejo visual creada correctamente.")
        print(f"Directorio: {args.visualize_bluewhite_dir}")


if __name__ == "__main__":
    main()