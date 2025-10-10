import numpy as np
from numba import njit



@njit
def stc_embed(x, rho, m, H_hat, w, h):
    n = len(x)
    num_states = 2**h

    wght = np.array([0.0] + [np.inf] * (2**h - 1))

    path = np.zeros((n, num_states), dtype=np.uint8)

    indx = 0
    indm = 0

    newwght = np.empty(num_states, dtype=np.float64)
    for i in range(n // w):
        for j in range(w):
            newwght.fill(np.inf)

            for k in range(num_states):
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


    ## embedding_cost была в псевдокоде из оригинальной статьи и возвращалась вместе с y
    ## После выполнения backward pass она показывает суммарную цену изменений.
    ## Вообще, если у нас все картинки 512x512 как например принимает SRNet,
    ## то имеет смысл взглянуть на корреляцию embedding_cost и того, насколько успешно SRNet распознала встраивание.
    ## Однако при обычном использовании она не несет никакой полезной информации,
    ## т.к. зависит намного больше от размеров изображения, чем от эффективности встраивания.

    # embedding_cost = wght[0]

    state = 0
    indx -= 1
    indm -= 1

    y = np.array([0] * n)

    for i in range(n // w):
        state = 2 * state + m[indm]
        indm -= 1
        for j in range(w - 1, -1, -1):
            y[indx] = path[indx][state]
            state ^= y[indx] * H_hat[j]
            indx -= 1

    return y


def stc_extract(y, H_hat):
    y = np.array(y, dtype=np.uint8)
    h, w = np.size(H_hat, 0), np.size(H_hat, 1)
    blocks = len(y) // w

    syndrome = np.zeros(blocks + h, dtype=np.uint8)

    for b in range(blocks):
        col_start = b * w
        col_end   = (b + 1) * w
        y_block = y[col_start:col_end]

        local = (H_hat @ y_block) % 2
        syndrome[b:b+h] ^= np.uint8(local)

    return syndrome[:blocks]
