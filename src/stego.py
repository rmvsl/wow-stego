import numpy as np
from utils import prepare_image
from stc.mats import get_matrix
from cost_map import compute_rho
from stc.stc import stc_embed, stc_extract
from utils import u32_from_bits, u32_to_bits



def embed(path: str, message: bytes, w: int, h: int):
    H_hat = get_matrix(w, h)

    cover_m, cover, x = prepare_image(path)
    blocks = len(x) // w
    
    rho = compute_rho(cover_m.astype(np.float64), -1).ravel()

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


def embed_ones(path: str, w: int, h: int):
    H_hat = get_matrix(w, h)

    cover_m, cover, x = prepare_image(path)
    blocks = len(x) // w
    
    rho = compute_rho(cover_m.astype(np.float64), -1).ravel()

    m = np.ones(blocks, dtype=np.uint8)

    y, embedding_cost = stc_embed(x, rho, m, H_hat, h)

    stego = (cover >> 1 << 1) + y
    stego_img = np.reshape(stego, cover_m.shape)

    return stego_img, embedding_cost


def extract(path: str, w: int, h: int):
    H_hat = get_matrix(w, h)

    stego_m, stego, y = prepare_image(path)

    m = stc_extract(y, H_hat, h)
    l = u32_from_bits(m[:32])

    message = np.packbits(m[32:32 + l]).tobytes()

    return message