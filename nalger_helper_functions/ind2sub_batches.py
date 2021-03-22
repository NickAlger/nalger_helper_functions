import numpy as np


def ind2sub_batches(ind, batch_lengths):
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/blob/master/jupyter_notebooks/ind2sub_batches.ipynb
    if (ind < 0) or (ind >= np.sum(batch_lengths)):
        raise RuntimeError('ind is not in any batch')

    cs = np.cumsum(batch_lengths)
    b = np.argwhere(ind < cs).reshape(-1)[0]
    zcs = np.concatenate([np.array([0]), cs])
    k = ind - zcs[b]
    return b, k


def sub2ind_batches(b, k, batch_lengths):
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/blob/master/jupyter_notebooks/ind2sub_batches.ipynb
    if b >= len(batch_lengths):
        raise RuntimeError('batch ind too big')
    if k >= batch_lengths[b]:
        raise RuntimeError('k too big for batch')

    return np.sum(batch_lengths[:b]) + k
