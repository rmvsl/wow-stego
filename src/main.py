import os
import sys
import numpy as np
from PIL import Image
from io import BytesIO
import imageio.v3 as iio
from matplotlib import pyplot as plt

from args import args
from cost_map import compute_rho
from stego import embed, embed_ones, extract



command = args.command
image_path = args.image_path
out_path = args.out_path
image_name = os.path.splitext(image_path)[0]
w, h = args.width, args.height


if command == "embed":
    message = sys.stdin.buffer.read()

    stego_img, distortion = embed(image_path, message, w, h)

    print(f"Distortion: {distortion:.2f}", file=sys.stderr)

    if out_path:
        iio.imwrite(out_path, stego_img, extension='.png')
    else:
        buf = BytesIO()
        iio.imwrite(buf, stego_img, extension='.png')
        sys.stdout.buffer.write(buf.getvalue())


elif command == "extract":
    message = extract(image_path, w, h)

    if out_path:
        with open(out_path, 'wb') as out:
            out.write(message)
    else:
        sys.stdout.buffer.write(message)


# Show how much data can fit into the given image
elif command == 'info':
    image = iio.imread(image_path)

    if len(image.shape) == 2:
        n = image.shape[0] * image.shape[1]
    elif image.shape[2] in (3, 4):
        n = image.shape[0] * image.shape[1] * 3

    blocks = n // w

    print(f"w = {w}")
    print(f"h = {h}")
    print(f"Payload: {1/w:.3f} bpp")
    print(f"Cover length: {n} bits")
    print(f"Max secret message length: {blocks - 64} bits = {blocks // 8 - 8} bytes")


# Visualize the cost map
elif command == 'cost_map':
    cover_m = iio.imread(image_path).astype(np.uint8)

    if len(cover_m.shape) == 2:
        plt.imshow(np.log2(compute_rho(cover_m)), cmap=args.cmap)
        plt.axis('off')

    elif cover_m.shape[2] in (3, 4):
        if cover_m.shape[0] >= cover_m.shape[1]:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        else:
            fig, axes = plt.subplots(3, 1, figsize=(5, 15))

        for i in range(3):
            axes[i].axis('off')
            axes[i].imshow(np.log2(compute_rho(cover_m[:, :, i])), cmap=args.cmap)

    else:
        raise RuntimeError("Only grayscale and RGB images are supported")

    plt.show()


# Will be used later
elif command == 'cost_map_luma':
    cover_m =  np.array(Image.open(image_path).convert("L")).astype(np.uint8)

    rho = np.log2(compute_rho(cover_m))
    rho = rho - np.min(rho)
    rho = (rho / np.max(rho) * 255).astype(np.uint8)

    if out_path:
        iio.imwrite(out_path, rho, extension='.png')
    else:
        buf = BytesIO()
        iio.imwrite(buf, rho, extension='.png')
        sys.stdout.buffer.write(buf.getvalue())


# Difference between the cover and stego images when embedding the given message
# Black — pixel wasn't changed
# White — pixel was increased/decreased by 1
elif command in ('xy_diff', 'xy_diff_full'):
    cover_m = iio.imread(image_path).astype(np.uint8)

    if command == 'xy_diff':
        message = sys.stdin.buffer.read()
        stego_m, distortion = embed(image_path, message, w, h)
    else:
        stego_m, distortion = embed_ones(image_path, w, h)

    print(f"Distortion: {distortion:.2f}")

    if len(cover_m.shape) == 2:
        plt.imshow(abs(stego_m.astype(np.int8) - cover_m.astype(np.int8)), cmap=args.cmap)

    elif cover_m.shape[2] in (3, 4):
        if cover_m.shape[0] >= cover_m.shape[1]:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        else:
            fig, axes = plt.subplots(3, 1, figsize=(5, 15))

        for i in range(3):
            axes[i].axis('off')
            axes[i].imshow(np.abs(stego_m[:, :, i].astype(np.int8) - cover_m[:, :, i].astype(np.int8)), cmap=args.cmap)

    else:
        raise RuntimeError("Only grayscale and RGB images are supported")

    plt.show()
