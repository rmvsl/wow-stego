import os
import cv2
import sys
import numpy as np
import imageio as io
from matplotlib import pyplot as plt

from args import args
from wow import compute_rho
from mats import get_matrix
from stc import stc_embed, stc_extract



# Little-endian
def u32_to_bits(n):
    if n < 0 or n > 2 ** 32 - 1:
        raise ValueError("Not u32")

    B = []
    for _ in range(32):
        b = n % 2
        n //= 2
        B.append(b)
    return B

def u32_from_bits(B):
    if len(B) != 32:
        raise ValueError("Not u32")

    n = 0
    for i in range(32):
        n += int(B[i]) * 2 ** i
    return n


def prepare_image(image_path):
    image_m = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    
    image = image_m.ravel()
    x = image % 2
    
    return image_m, image, x


command = args.command
image_path = args.image_path
w, h = args.width, args.height

H_HAT = get_matrix(w, h)


if command == "embed":
    cover_m, cover, x = prepare_image(image_path)

    blocks = len(x) // w
    
    rho = compute_rho(cover_m.astype(np.float64), -1).ravel()

    mbits = np.unpackbits(np.frombuffer(sys.stdin.buffer.read(), dtype=np.uint8))
    l = len(mbits)

    m = np.concatenate((u32_to_bits(l), mbits))
    k = len(m)

    if k > blocks:
        print("Your message is too long for this image")
        print(f"Max capacity is {blocks // 8} bytes")
        exit()

    y = x
    y[:k * w], embedding_cost = stc_embed(x[:k * w], rho[:k * w], m, H_HAT, h)

    print("Embedding cost:", embedding_cost)

    stego = (cover >> 1 << 1) + y
    stego_img = np.reshape(stego, cover_m.shape)

    stego_path = os.path.splitext(image_path)[0] + ".stego.png"
    io.imwrite(stego_path, stego_img.astype(np.uint8))


elif command == "extract":
    stego_m, stego, y = prepare_image(image_path)

    m = stc_extract(y, H_HAT, h)
    l = u32_from_bits(m[:32])

    m_bytes = np.packbits(m[32:32 + l]).tobytes()

    sys.stdout.buffer.write(m_bytes)


# Show how much data can fit into the given image
elif command == "info":
    _, image, _ = prepare_image(image_path)

    blocks = len(image) // w

    print(f"w = {w}")
    print(f"h = {h}")
    print(f"Payload: {1/w} bpp")
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
    cover_m, cover, x = prepare_image(image_path)

    rho = compute_rho(cover_m.astype(np.float64), -1)
    rho = rho.ravel()

    blocks = len(x) // w

    mbits = np.unpackbits(np.frombuffer(sys.stdin.buffer.read(), dtype=np.uint8))
    l = len(mbits)

    m = np.concatenate((u32_to_bits(l), mbits))
    k = len(m)

    if k > blocks:
        print("Your message is too long for this image")
        print(f"Max capacity is {blocks // 8} bytes")
        exit()

    y = x.copy()
    y[:k * w], _ = stc_embed(x[:k * w], rho[:k * w], m, H_HAT, h)

    plt.imshow(np.reshape(y, cover_m.shape).astype(np.int8) - np.reshape(x, cover_m.shape).astype(np.int8), cmap='gray')
    plt.show()


# Difference between cover and stego for a maximum-length message
elif command == "xy_diff_full":
    cover_m, cover, x = prepare_image(image_path)

    rho = compute_rho(cover_m.astype(np.float64), -1).ravel()

    blocks = len(x) // w
    m = np.ones(blocks, dtype=np.uint8)

    y, _ = stc_embed(x, rho, m, H_HAT, h)

    plt.imshow(np.reshape(y - x, cover_m.shape), cmap='gray')
    plt.show()
