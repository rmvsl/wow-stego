import os
import sys
import numpy as np
import imageio as io
from matplotlib import pyplot as plt

from args import args
from utils import prepare_image
from stc.mats import get_matrix
from stego import embed, embed_ones, extract
from cost_map import compute_rho



command = args.command
image_path = args.image_path
w, h = args.width, args.height

H_HAT = get_matrix(w, h)


if command == "embed":
    message = sys.stdin.buffer.read()

    stego_img, distortion = embed(image_path, message, w, h)

    print(f"Distortion: {distortion:.2f}")

    stego_path = os.path.splitext(image_path)[0] + ".stego.png"
    io.imwrite(stego_path, stego_img.astype(np.uint8))


elif command == "extract":
    message = extract(image_path, w, h)

    sys.stdout.buffer.write(message)


# Show how much data can fit into the given image
elif command == "info":
    _, image, _ = prepare_image(image_path)

    blocks = len(image) // w

    print(f"w = {w}")
    print(f"h = {h}")
    print(f"Payload: {1/w:.3f} bpp")
    print(f"Cover length: {len(image)} bits")
    print(f"Max secret message length: {blocks} bits = {blocks // 8} bytes")


# Visualize the cost map
elif command == "cost_map":
    cover_m, _, _ = prepare_image(image_path)

    rho = compute_rho(cover_m, -1)

    plt.imshow(np.log(rho), cmap='magma')
    plt.colorbar()
    plt.show()


# Difference between the cover and stego images when embedding the given message
# Gray — pixel wasn't changed
# Black — pixel was increased by 1
# White — pixel was decreased by 1
elif command == "xy_diff":
    message = sys.stdin.buffer.read()

    cover_m, _, _ = prepare_image(image_path)
    stego_img, distortion = embed(image_path, message, w, h)

    print(f"Distortion: {distortion:.2f}")

    plt.imshow(stego_img.astype(np.int8) - cover_m.astype(np.int8), cmap='gray')
    plt.show()


# Difference between cover and stego for a maximum-length message
elif command == "xy_diff_full":
    cover_m, _, _ = prepare_image(image_path)
    stego_m, distortion = embed_ones(image_path, w, h)

    print(f"Distortion: {distortion:.2f}")

    plt.imshow(stego_m.astype(np.int8) - cover_m.astype(np.int8), cmap='gray')
    plt.show()
