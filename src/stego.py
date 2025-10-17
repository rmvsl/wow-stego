import numpy as np
import imageio.v3 as iio
from stc.mats import get_matrix
from cost_map import compute_rho
from stc.stc import stc_embed, stc_extract
from utils import u32_from_bits, u32_to_bits



def embed(path: str, message: bytes, w: int, h: int):
    img = iio.imread(path).astype(np.uint8)

    if len(img.shape) == 2:
        return embed_gray(img, message, w, h)
    elif img.shape[2] == 3:
        return embed_rgb(img, message, w, h)
    elif img.shape[2] == 4:
        stego_img, distortion = embed_rgb(img[:, :, :3], message, w, h)
        return np.dstack((stego_img, img[:, :, 3])), distortion
    else:
        raise RuntimeError("Only grayscale and RGB images are supported")


def embed_gray(cover_m, message: bytes, w: int, h: int):
    H_hat = get_matrix(w, h)

    cover = cover_m.ravel()
    x = cover % 2

    blocks = len(x) // w
    
    rho = compute_rho(cover_m.astype(np.float64)).ravel()

    mbits = np.unpackbits(np.frombuffer(message, dtype=np.uint8))
    l = len(mbits)

    m = np.concatenate((u32_to_bits(l), mbits))
    k = len(m)

    if k > blocks:
        raise RuntimeError('Message is too long')

    y = x
    y[:k * w], embedding_cost = stc_embed(x[:k * w], rho[:k * w], m, H_hat, h)

    stego = (cover >> 1 << 1) + y
    stego_img = np.reshape(stego, cover_m.shape)

    return stego_img, embedding_cost


def embed_rgb(Crgb, message: bytes, w: int, h: int):
    H_hat = get_matrix(w, h)

    cover = np.empty(Crgb.shape[0] * Crgb.shape[1] * 3, dtype=np.uint8)
    rho = np.empty(Crgb.shape[0] * Crgb.shape[1] * 3)

    for i in range(3):
        cover[i::3] = Crgb[:, :, i].ravel()
        rho[i::3] = compute_rho(Crgb[:, :, i].astype(np.float64)).ravel()

    x = cover % 2

    blocks = len(x) // w

    mbits = np.unpackbits(np.frombuffer(message, dtype=np.uint8))
    l = len(mbits)

    m = np.concatenate((u32_to_bits(l), mbits))
    k = len(m)

    if k > blocks:
        raise RuntimeError('Message is too long')

    y = x
    y[:k * w], embedding_cost = stc_embed(x[:k * w], rho[:k * w], m, H_hat, h)

    stego = (cover >> 1 << 1) + y

    Srgb = np.empty(Crgb.shape, dtype=np.uint8)

    for i in range(3):
        Srgb[:, :, i] = np.reshape(stego[i::3], (Srgb.shape[0], Srgb.shape[1]))

    return Srgb, embedding_cost


def extract(path: str, w: int, h: int):
    img = iio.imread(path)

    if len(img.shape) == 2:
        return extract_gray(img, w, h)
    elif img.shape[2] == 3:
        return extract_rgb(img, w, h)
    elif img.shape[2] == 4:
        return extract_rgb(img[:, :, :3], w, h)
    else:
        raise RuntimeError("Only grayscale and RGB images are supported")


def extract_gray(stego_m, w: int, h: int):
    H_hat = get_matrix(w, h)

    stego = stego_m.ravel()
    y = stego % 2

    m = stc_extract(y, H_hat, h)
    l = u32_from_bits(m[:32])

    message = np.packbits(m[32:32 + l]).tobytes()

    return message


def extract_rgb(Srgb, w: int, h: int):
    H_hat = get_matrix(w, h)

    stego = np.empty(Srgb.shape[0] * Srgb.shape[1] * 3, dtype=np.uint8)

    for i in range(3):
        stego[i::3] = Srgb[:, :, i].ravel()

    y = stego % 2

    m = stc_extract(y, H_hat, h)
    l = u32_from_bits(m[:32])

    message = np.packbits(m[32:32 + l]).tobytes()

    return message
