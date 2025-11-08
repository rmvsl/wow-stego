import numpy as np
from numba import njit



@njit
def stc_embed(x, rho, m, H_hat, h):
    w = len(H_hat)
    n = len(x)

    wght = np.array([0.0] + [np.inf] * (2**h - 1))

    path = np.zeros((n, 2**h), dtype=np.uint8)

    indx = 0
    indm = 0

    newwght = np.empty(2**h, dtype=np.float64)
    for i in range(n // w):
        for j in range(w):
            newwght.fill(np.inf)

            for k in range(2**h):
                w0 = wght[k] + x[indx] * rho[indx]
                w1 = wght[k ^ H_hat[j]] + (1-x[indx])*rho[indx]

                path[indx][k] = np.uint8(w1 < w0)
                newwght[k] = min(w0, w1)

            wght, newwght = newwght, wght
            indx += 1

        for j in range(2**(h-1)):
            wght[j] = wght[2*j + m[indm]]
        wght[2**(h-1):2**h] = np.inf

        indm += 1


    embedding_cost = wght[0]
    state = 0
    indx -= 1
    indm -= 1

    y = np.array([0] * n)

    for i in range(n // w):
        state = 2 * state + m[indm]
        state &= (1 << h) - 1
        indm -= 1
        for j in range(w - 1, -1, -1):
            y[indx] = path[indx][state]
            state ^= y[indx] * H_hat[j]
            indx -= 1

    return y, embedding_cost


@njit
def stc_extract(y, H_hat, h):
    w = len(H_hat)
    blocks = len(y) // w

    m = np.zeros(blocks, dtype=np.uint8)
    syndrome = 0

    for b in range(blocks):
        for j in range(w):
            if y[b * w + j] == 1:
                syndrome ^= H_hat[j]
        
        m[b] = syndrome & 1
        
        syndrome >>= 1

    return m