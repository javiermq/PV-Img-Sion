import cv2
import numpy as np
from pathlib import Path
import argparse


def crop_with_padding(img, x1, y1, x2, y2):
    h, w = img.shape[:2]

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        img = cv2.copyMakeBorder(
            img,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

    x1 += pad_left
    x2 += pad_left
    y1 += pad_top
    y2 += pad_top

    return img[y1:y2, x1:x2]


def crop_mode(img, cx, cy, r, out_size=None, alpha=True):
    """
    Recorta un cuadrado de lado 2r alrededor del círculo.
    Opcionalmente deja fuera del círculo transparencia.
    """
    x1 = int(round(cx - r))
    y1 = int(round(cy - r))
    x2 = int(round(cx + r))
    y2 = int(round(cy + r))

    crop = crop_with_padding(img, x1, y1, x2, y2)

    # asegurar cuadrado
    h, w = crop.shape[:2]
    side = max(h, w)

    if h != w:
        tmp = np.zeros((side, side, 3), dtype=np.uint8)
        yoff = (side - h) // 2
        xoff = (side - w) // 2
        tmp[yoff:yoff+h, xoff:xoff+w] = crop
        crop = tmp

    if out_size is not None and out_size > 0:
        crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

    if alpha:
        side = crop.shape[0]
        mask = np.zeros((side, side), dtype=np.uint8)
        cv2.circle(mask, (side // 2, side // 2), side // 2 - 1, 255, -1)

        out = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        out[:, :, 3] = mask
        return out

    return crop


def circle_to_square_map(img, cx, cy, r, out_size=1024):
    """
    Remapea el disco circular a un cuadrado completo.
    Esto NO es una homografía pura, pero suele ser justo lo que se quiere
    cuando se desea pasar un círculo a una imagen cuadrada útil.
    """
    # coordenadas del cuadrado destino en [-1, 1]
    lin = np.linspace(-1.0, 1.0, out_size, dtype=np.float32)
    u, v = np.meshgrid(lin, lin)

    # Mapeo square -> disk (FG-squircular / aproximado)
    x = u * np.sqrt(np.clip(1.0 - (v * v) / 2.0, 0.0, 1.0))
    y = v * np.sqrt(np.clip(1.0 - (u * u) / 2.0, 0.0, 1.0))

    map_x = (cx + r * x).astype(np.float32)
    map_y = (cy + r * y).astype(np.float32)

    warped = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    return warped


def process_image(img_path, out_dir, mode, cx, cy, r, size, alpha):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {img_path}")

    if mode == "crop":
        result = crop_mode(img, cx, cy, r, out_size=size, alpha=alpha)
        ext = ".png" if alpha else ".jpg"
        out_path = out_dir / f"{img_path.stem}_crop{ext}"

    elif mode == "squaremap":
        if size is None:
            size = int(2 * r)
        result = circle_to_square_map(img, cx, cy, r, out_size=size)
        out_path = out_dir / f"{img_path.stem}_squaremap.jpg"

    else:
        raise ValueError("Modo no válido")

    cv2.imwrite(str(out_path), result)
    return out_path


def process_folder(input_dir, output_dir, mode, cx, cy, r, size, alpha):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    files = []
    for ext in exts:
        files.extend(input_dir.glob(ext))

    if not files:
        print("No se encontraron imágenes.")
        return

    for img_path in files:
        try:
            out_path = process_image(img_path, output_dir, mode, cx, cy, r, size, alpha)
            print(f"OK: {img_path.name} -> {out_path.name}")
        except Exception as e:
            print(f"ERROR en {img_path.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Procesa imágenes de ojo de pez con centro/radio manuales."
    )

    parser.add_argument("--input", required=True, help="Carpeta de entrada")
    parser.add_argument("--output", required=True, help="Carpeta de salida")

    parser.add_argument("--cx", type=float, required=True, help="Centro X del círculo")
    parser.add_argument("--cy", type=float, required=True, help="Centro Y del círculo")
    parser.add_argument("--r", type=float, required=True, help="Radio del círculo")

    parser.add_argument(
        "--mode",
        choices=["crop", "squaremap"],
        default="crop",
        help="crop = recorte cuadrado; squaremap = disco a cuadrado"
    )

    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Tamaño final de salida (ej. 512, 1024). En squaremap, si no se pone, usa 2r."
    )

    parser.add_argument(
        "--no_alpha",
        action="store_true",
        help="En modo crop, guarda sin transparencia"
    )

    args = parser.parse_args()

    process_folder(
        input_dir=args.input,
        output_dir=args.output,
        mode=args.mode,
        cx=args.cx,
        cy=args.cy,
        r=args.r,
        size=args.size,
        alpha=not args.no_alpha
    )