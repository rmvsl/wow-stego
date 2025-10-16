import cv2
import numpy as np



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


def prepare_image(path):
    image_m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image_m is None:
        raise FileNotFoundError

    image_m = image_m.astype(np.uint8)
    
    image = image_m.ravel()
    x = image % 2
    
    return image_m, image, x