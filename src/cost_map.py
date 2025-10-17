import numpy as np
from pywt import Wavelet
from scipy.signal import fftconvolve



wt = Wavelet("db8")

h = wt.dec_lo
g = wt.dec_hi

lh = np.outer(h, g)
hl = np.outer(g, h)
hh = np.outer(g, g)

F = [lh, hl, hh]


def fftconvolve_mirror_padded(cover, kernel):
    pz = np.size(kernel[0], 0) // 2
    cover_padded = np.pad(cover, pz, 'reflect')
    result_padded = fftconvolve(cover_padded, kernel, mode='same')
    result = result_padded[pz:-pz, pz:-pz]
    return result


def compute_rho(cover, p = -1):
    wetCost = 1e10

    XI = []

    for f in F:
        r = fftconvolve_mirror_padded(cover, f)

        xi = fftconvolve_mirror_padded(np.abs(r), np.rot90(np.abs(f), 2))

        # fftconvolve may sometimes return small negative values, even though mathematically they shouldn't be there
        xi[xi < 0] = 0

        if np.size(f, 0) % 2 == 0:
            xi = np.roll(xi, 1, axis=0)
        if np.size(f, 1) % 2 == 0:
            xi = np.roll(xi, 1, axis=1)

        XI.append(xi)

    with np.errstate(divide='ignore'):
        rho = (XI[0] ** p + XI[1] ** p + XI[2] ** p) ** (-1 / p)

    rho[rho > wetCost] = wetCost
    rho[np.isnan(rho)] = wetCost

    return rho
