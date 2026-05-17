import cv2
import numpy as np
from pathlib import Path
import argparse


def crop_circle_region(img, cx, cy, radius, pad=0):
    """
    Recorta una región cuadrada centrada en (cx, cy) con lado 2*(radius+pad).
    Si se sale de la imagen, añade padding negro.

    Devuelve:
        crop: imagen recortada cuadrada
    """
    r = int(round(radius + pad))
    cx = int(round(cx))
    cy = int(round(cy))

    x1 = cx - r
    y1 = cy - r
    x2 = cx + r
    y2 = cy + r

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

    crop = img[y1:y2, x1:x2]

    # Asegurar que es cuadrado
    hh, ww = crop.shape[:2]
    if hh != ww:
        side = max(hh, ww)
        square = np.zeros((side, side, 3), dtype=crop.dtype)
        yoff = (side - hh) // 2
        xoff = (side - ww) // 2
        square[yoff:yoff+hh, xoff:xoff+ww] = crop
        crop = square

    return crop


def add_circle_alpha(crop):
    """
    Añade canal alpha dejando transparente lo que queda fuera del círculo inscrito.
    """
    h, w = crop.shape[:2]
    side = min(h, w)
    cx = w / 2.0
    cy = h / 2.0
    r = side / 2.0

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r ** 2)
    alpha = np.zeros((h, w), dtype=np.uint8)
    alpha[mask] = 255

    out = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    out[:, :, 3] = alpha
    return out


def fisheye_crop_to_square(crop, output_size=1024, projection="equidistant", background_value=0):
    """
    Toma un recorte cuadrado con el círculo centrado y lo proyecta a imagen cuadrada.

    projection:
        - equidistant
        - stereographic
        - orthographic
        - squircle
    """
    h, w = crop.shape[:2]
    cx = w / 2.0
    cy = h / 2.0
    radius = min(w, h) / 2.0

    yy, xx = np.meshgrid(
        np.linspace(-1, 1, output_size, dtype=np.float32),
        np.linspace(-1, 1, output_size, dtype=np.float32),
        indexing="ij"
    )

    r_out = np.sqrt(xx ** 2 + yy ** 2)
    phi = np.arctan2(yy, xx)

    if projection == "squircle":
        # El círculo ocupa todo el cuadrado
        src_x_norm = xx * np.sqrt(np.clip(1 - (yy ** 2) / 2, 0, 1))
        src_y_norm = yy * np.sqrt(np.clip(1 - (xx ** 2) / 2, 0, 1))
        valid = np.ones_like(src_x_norm, dtype=bool)

    else:
        theta_max = np.pi / 2
        valid = r_out <= 1.0

        if projection == "equidistant":
            src_r_norm = r_out

        elif projection == "stereographic":
            theta = r_out * theta_max
            src_r_norm = np.tan(theta / 2) / np.tan(theta_max / 2)

        elif projection == "orthographic":
            theta = r_out * theta_max
            src_r_norm = np.sin(theta) / np.sin(theta_max)

        else:
            raise ValueError(
                "projection debe ser: 'equidistant', 'stereographic', "
                "'orthographic' o 'squircle'"
            )

        src_x_norm = src_r_norm * np.cos(phi)
        src_y_norm = src_r_norm * np.sin(phi)

    map_x = (cx + src_x_norm * radius).astype(np.float32)
    map_y = (cy + src_y_norm * radius).astype(np.float32)

    output = cv2.remap(
        crop,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(background_value, background_value, background_value)
    )

    if projection != "squircle":
        output[~valid] = background_value

    return output


def draw_debug_circle(img, cx, cy, r):
    debug = img.copy()
    cv2.circle(debug, (int(round(cx)), int(round(cy))), int(round(r)), (0, 255, 0), 4)
    cv2.circle(debug, (int(round(cx)), int(round(cy))), 8, (0, 0, 255), -1)
    return debug


def process_image(
    image_path,
    output_dir,
    cx,
    cy,
    r,
    crop_pad=0,
    crop_size=None,
    projection="squircle",
    projected_size=1024,
    save_crop=True,
    crop_alpha=False,
    debug=False
):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"No se pudo leer: {image_path}")
        return

    # 1) Recorte
    crop = crop_circle_region(img, cx, cy, r, pad=crop_pad)

    crop_to_save = crop.copy()
    if crop_size is not None:
        crop_to_save = cv2.resize(crop_to_save, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

    # 2) Guardar recorte
    if save_crop:
        if crop_alpha:
            crop_alpha_img = add_circle_alpha(crop_to_save)
            crop_path = output_dir / f"{image_path.stem}_crop.png"
            cv2.imwrite(str(crop_path), crop_alpha_img)
        else:
            crop_path = output_dir / f"{image_path.stem}_crop.jpg"
            cv2.imwrite(str(crop_path), crop_to_save)

    # 3) Proyección del recorte
    projected = fisheye_crop_to_square(
        crop,
        output_size=projected_size,
        projection=projection,
        background_value=0
    )

    proj_path = output_dir / f"{image_path.stem}_projected_{projection}.png"
    cv2.imwrite(str(proj_path), projected)

    # 4) Debug opcional
    if debug:
        dbg = draw_debug_circle(img, cx, cy, r)
        dbg_path = output_dir / f"{image_path.stem}_debug_circle.jpg"
        cv2.imwrite(str(dbg_path), dbg)

    print(f"OK: {image_path.name}")


def process_folder(
    input_dir,
    output_dir,
    cx,
    cy,
    r,
    crop_pad=0,
    crop_size=None,
    projection="squircle",
    projected_size=1024,
    save_crop=True,
    crop_alpha=False,
    debug=False
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    files = []
    for ext in exts:
        files.extend(input_dir.glob(ext))

    files = sorted(files)

    if not files:
        print("No se encontraron imágenes.")
        return

    for image_path in files:
        process_image(
            image_path=image_path,
            output_dir=output_dir,
            cx=cx,
            cy=cy,
            r=r,
            crop_pad=crop_pad,
            crop_size=crop_size,
            projection=projection,
            projected_size=projected_size,
            save_crop=save_crop,
            crop_alpha=crop_alpha,
            debug=debug
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recorta un círculo manual y luego proyecta ese recorte a cuadrado."
    )

    parser.add_argument("--input", required=True, help="Imagen o carpeta de entrada")
    parser.add_argument("--output", required=True, help="Carpeta de salida")

    parser.add_argument("--cx", type=float, required=True, help="Centro X del círculo")
    parser.add_argument("--cy", type=float, required=True, help="Centro Y del círculo")
    parser.add_argument("--r", type=float, required=True, help="Radio del círculo")

    parser.add_argument("--crop_pad", type=float, default=0, help="Padding extra en el recorte")
    parser.add_argument("--crop_size", type=int, default=None, help="Tamaño opcional para guardar el recorte")
    parser.add_argument("--projected_size", type=int, default=1024, help="Tamaño de la proyección cuadrada")

    parser.add_argument(
        "--projection",
        choices=["equidistant", "stereographic", "orthographic", "squircle"],
        default="squircle",
        help="Tipo de proyección"
    )

    parser.add_argument("--no_crop_save", action="store_true", help="No guardar el recorte")
    parser.add_argument("--crop_alpha", action="store_true", help="Guardar el recorte con transparencia fuera del círculo")
    parser.add_argument("--debug", action="store_true", help="Guardar imagen de debug con el círculo dibujado")

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        process_image(
            image_path=input_path,
            output_dir=args.output,
            cx=args.cx,
            cy=args.cy,
            r=args.r,
            crop_pad=args.crop_pad,
            crop_size=args.crop_size,
            projection=args.projection,
            projected_size=args.projected_size,
            save_crop=not args.no_crop_save,
            crop_alpha=args.crop_alpha,
            debug=args.debug
        )

    elif input_path.is_dir():
        process_folder(
            input_dir=input_path,
            output_dir=args.output,
            cx=args.cx,
            cy=args.cy,
            r=args.r,
            crop_pad=args.crop_pad,
            crop_size=args.crop_size,
            projection=args.projection,
            projected_size=args.projected_size,
            save_crop=not args.no_crop_save,
            crop_alpha=args.crop_alpha,
            debug=args.debug
        )

    else:
        raise FileNotFoundError(f"No existe: {input_path}")