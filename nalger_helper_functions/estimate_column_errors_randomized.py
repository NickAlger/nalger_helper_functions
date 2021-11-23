import numpy as np
from tqdm.auto import tqdm


def estimate_column_errors_randomized(apply_A_true_numpy, apply_A_numpy, ncol_A, n_random_error_matvecs):
    Y_true = np.zeros((ncol_A, n_random_error_matvecs))
    Y = np.zeros((ncol_A, n_random_error_matvecs))
    for k in tqdm(range(n_random_error_matvecs)):
        omega = np.random.randn(ncol_A)
        Y_true[:, k] = apply_A_true_numpy(omega)
        Y[:, k] = apply_A_numpy(omega)

    norm_A = np.linalg.norm(Y_true) / np.sqrt(n_random_error_matvecs)
    norm_A_err = np.linalg.norm(Y_true - Y) / np.sqrt(n_random_error_matvecs)

    relative_error_overall = norm_A_err / norm_A

    norm_of_each_column_of_A_true = np.linalg.norm(Y_true, axis=1) / np.sqrt(n_random_error_matvecs)
    norm_of_each_column_of_error = np.linalg.norm(Y_true - Y, axis=1) / np.sqrt(n_random_error_matvecs)
    relative_error_of_each_column = norm_of_each_column_of_error / norm_of_each_column_of_A_true

    return relative_error_of_each_column, norm_of_each_column_of_A_true, norm_of_each_column_of_error, relative_error_overall