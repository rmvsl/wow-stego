import os
import sys
import numpy as np
import imageio.v3 as iio
from matplotlib import pyplot as plt

from args import args
from stego import embed, embed_ones, extract
from cost_map import compute_rho



command = args.command
image_path = args.image_path
image_name = os.path.splitext(image_path)[0]
w, h = args.width, args.height


if command == "embed":
    message = sys.stdin.buffer.read()

    stego_img, distortion = embed(image_path, message, w, h)

    print(f"Distortion: {distortion:.2f}")

    stego_path = image_name + ".stego.png"
    iio.imwrite(stego_path, stego_img)


elif command == "extract":
    message = extract(image_path, w, h)

    sys.stdout.buffer.write(message)


# Show how much data can fit into the given image
elif command == 'info':
    image = iio.imread(image_path)

    if len(image.shape) == 2:
        blocks = (image.shape[0] * image.shape[1]) // w

        print(f"w = {w}")
        print(f"h = {h}")
        print(f"Payload: {1/w:.3f} bpp")
        print(f"Cover length: {image.shape[0] * image.shape[1]} bits")
        print(f"Max secret message length: {blocks} bits = {blocks // 8} bytes")

    elif image.shape[2] in (3, 4):
        blocks = (image.shape[0] * image.shape[1] * 3) // w

        print(f"w = {w}")
        print(f"h = {h}")
        print(f"Payload: {1/w:.3f} bpp")
        print(f"Cover length: {image.shape[0] * image.shape[1] * 3} bits")
        print(f"Max secret message length: {blocks} bits = {blocks // 8} bytes")


# Visualize the cost map
elif command == 'cost_map':
    cover_m = iio.imread(image_path).astype(np.uint8)

    if len(cover_m.shape) == 2:
        if args.save:
            plt.imsave(image_name + ".cost_map.png", np.log2(compute_rho(cover_m)), cmap=args.cmap)
        else:
            plt.imshow(np.log2(compute_rho(cover_m)), cmap=args.cmap)
            plt.axis('off')
            plt.show()

    elif cover_m.shape[2] in (3, 4):
        if args.save:
            for i, ch in enumerate('rgb'):
                plt.imsave(image_name + f".cost_map.{ch}.png", np.log2(compute_rho(cover_m[:, :, i])), cmap=args.cmap)

        else:
            if cover_m.shape[0] >= cover_m.shape[1]:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            else:
                fig, axes = plt.subplots(3, 1, figsize=(5, 15))

            for i in range(3):
                axes[i].axis('off')
                axes[i].imshow(np.log2(compute_rho(cover_m[:, :, i])), cmap=args.cmap)

            plt.show()

    else:
        raise RuntimeError("Only grayscale and RGB images are supported")


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
        if args.save:
            plt.imsave(image_name + f".{command}.png", abs(stego_m.astype(np.int8) - cover_m.astype(np.int8)), cmap=args.cmap)
        else:
            plt.imshow(abs(stego_m.astype(np.int8) - cover_m.astype(np.int8)), cmap=args.cmap)
            plt.show()

    elif cover_m.shape[2] in (3, 4):
        if args.save:
            for i, ch in enumerate('rgb'):
                plt.imsave(image_name + f".{command}.{ch}.png", np.abs(stego_m[:, :, i].astype(np.int8) - cover_m[:, :, i].astype(np.int8)), cmap=args.cmap)
        else:
            if cover_m.shape[0] >= cover_m.shape[1]:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            else:
                fig, axes = plt.subplots(3, 1, figsize=(5, 15))

            for i in range(3):
                axes[i].axis('off')
                axes[i].imshow(np.abs(stego_m[:, :, i].astype(np.int8) - cover_m[:, :, i].astype(np.int8)), cmap=args.cmap)

            plt.show()

    else:
        raise RuntimeError("Only grayscale and RGB images are supported")
