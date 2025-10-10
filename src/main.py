import os
import cv2
import sys
import glob
import skimage
import numpy as np
import imageio as io
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

from stc import stc_embed, stc_extract
from wow import compute_rho



# Сначала маленькие биты
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

argv = sys.argv
arg_command = argv[1]
arg_path = argv[2]


# H_hat для 0.1bpp
H_HAT = np.array([[1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
                    [1, 0, 0, 1, 0, 1, 1, 0, 1, 1]], dtype=np.uint8)
H_HAT_EMBED = [0b11, 0b00, 0b01, 0b10, 0b01, 0b11, 0b10, 0b01, 0b10, 0b11]
WAVELET = "db8"
h, w = np.size(H_HAT, 0), np.size(H_HAT, 1)



if arg_command == "embed":
    cover_m, cover, x = prepare_image(arg_path)

    blocks = len(x) // w
    
    rho = compute_rho(cover_m.astype(np.float64), -1).ravel()

    mbits = np.array([int(bit) for byte in sys.stdin.buffer.read() for bit in format(byte, "08b")])
    l = len(mbits)

    m = np.concatenate((u32_to_bits(l), mbits))
    k = len(m)

    if k > blocks:
        print("Your message is too long for this image")
        print(f"Max capacity is {blocks // 8} characters|bytes")
        exit()

    y = x
    y[:k * w] = stc_embed(x[:k * w], rho[:k * w], m, H_HAT_EMBED, w, h)

    stego = (cover >> 1 << 1) + y
    stego_img = np.reshape(stego, cover_m.shape)

    stego_path = arg_path[:arg_path.rfind('.')] + ".stego.png"
    io.imwrite(stego_path, stego_img.astype(np.uint8))


elif arg_command == "extract":
    stego_m, stego, y = prepare_image(arg_path)

    m = stc_extract(y, H_HAT)
    l = u32_from_bits(m[:32])

    bytes_list = [int("".join(map(str, m[32:32 + l][i:i+8])), 2) for i in range(0, l, 8)]

    s = bytes(bytes_list)

    sys.stdout.buffer.write(s)


# Показать сколько можно уместить в картинку с текущими параметрами
elif arg_command == "info":
    _, image, _ = prepare_image(arg_path)

    blocks = len(image) // w

    print(f"w = {w}")
    print(f"h = {h}")
    print(f"Payload: {1/w} bpp")
    print(f"Cover length: {len(image)} bits")
    print(f"Blocks amount / max secret message length: {blocks} bits / {blocks // 8} bytes")


# Показать карту стоимостей для изображения
# log нужен чтобы избавиться от огромного разрыва в значениях, который мешает визуализации
elif arg_command == "cost_map":
    from matplotlib import pyplot as plt

    cover_m, _, _ = prepare_image(arg_path)

    rho = compute_rho(cover_m, -1)

    plt.imshow(np.log(rho), cmap='magma')
    plt.colorbar()
    plt.show()


# Разница между cover и stego при встраивании конкретного сообщения
# Серый — пиксель не изменился
# Черный — пиксель уменьшился на 1
# Белый — пиксель увеличился на 1
elif arg_command == "xy_diff":
    cover_m, cover, x = prepare_image(arg_path)

    rho = compute_rho(cover_m.astype(np.float64), -1)
    rho = rho.ravel()

    blocks = len(x) // w

    mbits = np.array([int(bit) for byte in input("Enter your message: ").encode("utf-8") for bit in format(byte, "08b")])
    l = len(mbits)

    m = np.concatenate((u32_to_bits(l), mbits))
    k = len(m)

    if k > blocks:
        print("Your message is too long for this image")
        print(f"Max capacity is {blocks // 8} bytes or characters")

    y = x.copy()
    y[:k * w] = stc_embed(x[:k * w], rho[:k * w], m, H_HAT_EMBED, w, h)

    plt.imshow(np.reshape(y, cover_m.shape).astype(np.int8) - np.reshape(x, cover_m.shape).astype(np.int8), cmap='gray')
    plt.show()


# Разница между cover и stego, но когда m имеет максимальную длину
# То есть k = blocks
elif arg_command == "xy_diff_full":
    cover_m, cover, x = prepare_image(arg_path)

    rho = compute_rho(cover_m.astype(np.float64), -1).ravel()

    blocks = len(x) // w
    m = np.ones(blocks, dtype=np.uint8)

    y = stc_embed(x, rho, m, H_HAT_EMBED, w, h)

    plt.imshow(np.reshape(y - x, cover_m.shape), cmap='gray')
    plt.show()




#########################################
### Экспериментальные способы встраивания

H_HAT = np.array([[1, 0],
            [1, 1]], dtype=np.uint8)
H_HAT_EMBED = [0b11, 0b10]
h, w = np.size(H_HAT, 0), np.size(H_HAT, 1)


if arg_command == "fancy_embed":
    cover_m, cover, x = prepare_image(arg_path)
    
    # Карта стоимостей без учета последних битов, которые используются 
    rho = compute_rho(cover_m.astype(np.float64), -1).ravel()
    rho_tilde = compute_rho((cover_m & 0b11111110).astype(np.float64), -1).ravel()


    # Сортировка битов по стоимости
    order = np.argsort(rho_tilde)
    x_sorted = x[order]
    
    blocks = len(x_sorted) // w

    # Вводим сообщение и превращаем в биты
    mbits = np.array([int(bit) for byte in input("Enter your message: ").encode("utf-8") for bit in format(byte, "08b")])
    l = len(mbits)

    if 32 + l > blocks:
        print("Your message is too long for this image")
        print(f"Max capacity is {blocks // 8} bytes")
        exit()

    m = np.concatenate((u32_to_bits(l), mbits))

    # Встраивание в отсортированные биты
    # В первую очередь дешевые
    y_sorted = x_sorted.copy()
    y_sorted[:(32 + l) * w] = stc_embed(x_sorted[:(32 + l) * w], rho[order][:(32 + l) * w], m, H_HAT_EMBED, w, h)
    
    # Восстановление исходного порядка
    y = np.zeros_like(x)
    y[order] = y_sorted

    stego = (cover >> 1 << 1) + y
    stego_img = np.reshape(stego, cover_m.shape)

    stego_path = arg_path[:arg_path.rfind('.')] + ".stego.png"
    io.imwrite(stego_path, stego_img.astype(np.uint8))


elif arg_command == "fancy_extract":
    stego_m, stego, y = prepare_image(arg_path)

    rho = compute_rho(stego_m.astype(np.float64), -1).ravel()
    rho_tilde = compute_rho((stego_m & 0b11111110).astype(np.float64), -1).ravel()
    
    # Та же сортировка, что и при встраивании
    order = np.argsort(rho_tilde)
    y_sorted = y[order]

    m = stc_extract(y_sorted, H_HAT)
    l = u32_from_bits(m[:32])

    bytes_list = [int("".join(map(str, m[32:32 + l][i:i+8])), 2) for i in range(0, l, 8)]
    s = bytes(bytes_list).decode("utf-8")
    print(s)