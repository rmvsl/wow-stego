import os
import sys
import numpy as np
import imageio.v3 as iio
from matplotlib import pyplot as plt

from args import args
from stego import embed, extract
from cost_map import compute_rho



command = args.command
image_path = args.image_path
w, h = args.width, args.height


if command == "embed":
    message = sys.stdin.buffer.read()

    stego_img, distortion = embed(image_path, message, w, h)

    print(f"Distortion: {distortion:.2f}")

    stego_path = os.path.splitext(image_path)[0] + ".stego.png"
    iio.imwrite(stego_path, stego_img)


elif command == "extract":
    message = extract(image_path, w, h)

    sys.stdout.buffer.write(message)


# Show how much data can fit into the given image
elif command == "info":
    image = iio.imread(image_path)

    if len(image.shape) == 2:
        blocks = (image.shape[0] * image.shape[1]) // w

        print(f"w = {w}")
        print(f"h = {h}")
        print(f"Payload: {1/w:.3f} bpp")
        print(f"Cover length: {image.shape[0] * image.shape[1]} bits")
        print(f"Max secret message length: {blocks} bits = {blocks // 8} bytes")

    elif image.shape[2] == 3 or image.shape[2] == 4:
        blocks = (image.shape[0] * image.shape[1] * 3) // w

        print(f"w = {w}")
        print(f"h = {h}")
        print(f"Payload: {1/w:.3f} bpp")
        print(f"Cover length: {image.shape[0] * image.shape[1] * 3} bits")
        print(f"Max secret message length: {blocks} bits = {blocks // 8} bytes")


# Visualize the cost map
elif command == "cost_map":
    cover_m = iio.imread(image_path).astype(np.uint8)

    rho = compute_rho(cover_m)

    plt.imshow(np.log(rho), cmap='magma')
    plt.colorbar()
    plt.show()


# Difference between the cover and stego images when embedding the given message
# Gray — pixel wasn't changed
# Black — pixel was increased by 1
# White — pixel was decreased by 1
elif command == "xy_diff":
    message = sys.stdin.buffer.read()

    cover_m = iio.imread(image_path).astype(np.uint8)
    stego_m, distortion = embed(image_path, message, w, h)

    print(f"Distortion: {distortion:.2f}")

    if len(cover_m.shape) == 2:
        plt.imshow(stego_m.astype(np.int8) - cover_m.astype(np.int8), cmap='gray')

    elif cover_m.shape[2] == 3 or cover_m.shape[2] == 4:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(stego_m[:, :, 0].astype(np.int8) - cover_m[:, :, 0].astype(np.int8), cmap='gray')
        axes[1].imshow(stego_m[:, :, 1].astype(np.int8) - cover_m[:, :, 1].astype(np.int8), cmap='gray')
        axes[2].imshow(stego_m[:, :, 2].astype(np.int8) - cover_m[:, :, 2].astype(np.int8), cmap='gray')

    else:
        raise RuntimeError("Only grayscale and RGB images are supported")

    plt.show()


# # # Doesn't work yet
# # Difference between cover and stego for a maximum-length message
# elif command == "xy_diff_full":
#     cover_m, _, _ = prepare_image(image_path)
#     stego_m, distortion = embed_ones(image_path, w, h)

#     print(f"Distortion: {distortion:.2f}")

#     plt.imshow(stego_m.astype(np.int8) - cover_m.astype(np.int8), cmap='gray')
#     plt.show()
