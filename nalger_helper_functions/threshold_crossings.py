import numpy as np
import numpy.typing as npt

def threshold_crossings(y: npt.ArrayLike, thresholds: npt.ArrayLike) -> np.ndarray:
    '''out[ii]=k where k is first index at which y[k] < threshold[ii].
    If no no indicies satisfy this condition, then out[ii]=-1

    In:
        threshold_crossings([1000, 100, 10, 1, 0.1],
                            [2000, 1000, 999, 333, 8, 0.5, 0.001])
    Out:
        array([ 0,  1,  1,  1,  3,  4, -1])
    '''
    y = np.array(y)
    thresholds = np.array(thresholds)
    assert(len(y.shape) == 1)
    assert(len(thresholds.shape) == 1)
    crossings = -1 * np.ones(len(thresholds), dtype=int)
    for ii, threshold in enumerate(thresholds):
        good_inds = np.argwhere(y < threshold).reshape(-1)
        if len(good_inds) > 0:
            crossing = np.min(good_inds)
            crossings[ii] = crossing
    return crossings