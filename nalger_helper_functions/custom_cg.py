import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla


def custom_cg(A, b, *args, display=True, track_residuals=True, x_true=None, iters_to_save=None, **kwargs):
    '''Wrapper for scipy's CG that adds ability to track errors and residuals as the iterations progress,
    and save user-selected iterates. Note: tracking the residuals requires applying the coefficient matrix twice
    each iteration instead of once.

    :param A: numpy array or numpy matrix or scipy sparse matrix or LinearOperator. Coefficient matrix in Ax=b. shape=(n,n)
    :param b: numpy array. right hand side vector. shape=(n,)
    :param *args: other arguments to pass to scipy's CG function
    :param display: bool. if true, we display convergence info as iterations progress
    :param track_residuals: bool. if true, we compute the residual ||A*xk-b||_2 at each iteration
    :param x_true: numpy array. true solution to Ax=b. len=n
        If supplied, we compute the error ||xk-x_true||_2/||x_true||_2 at each iteration.
    :param iters_to_save: list of iteration numbers to save and return
    :param **kwargs: other keyword arguments to pass to scipy's CG function
    :return:
    x: numpy array. CG solution to Ax=b
    info: int. convergence info. 0: converged. >0: did not converge to tolerance in maxiter. <0: failure within CG
    residuals: list of floats. relative residuals at each iteration, ||A*xk-b||_2/||b||_2
    errors: list of floats. relative errors at each iteration, ||xk-x_true||_2/||x_true||_2
    kk_saved: list of ints. iterations that xk was saved at
    xx_saved: list of numpy arrays. xx_saved[i] is x at iteration kk_saved[i]
    '''
    if track_residuals:
        residuals = list()

    if x_true is not None:
        errors = list()

    if iters_to_save is not None:
        iters_to_save = list(iters_to_save)
        xx_saved = list()
        kk_saved = list()

    if hasattr(A, 'matvec'):
        A_linop = A
    elif sps.issparse(A):
        A_linop = spla.LinearOperator(A.shape, matvec=lambda u: A * u)
    else:
        A_linop = spla.LinearOperator(A.shape, matvec=lambda u: np.dot(A, u))

    counter = [0,]

    def callback(xk):
        k = counter[0]
        counter[0] = k+1
        display_str = 'k=' + str(k)

        if iters_to_save is not None:
            if k in iters_to_save:
                xx_saved.append(xk.copy())
                kk_saved.append(k)

        if track_residuals:
            res = np.linalg.norm(A_linop.matvec(xk) - b) / np.linalg.norm(b)
            residuals.append(res)
            display_str += ', res=' + str(res)

        if x_true is not None:
            err = np.linalg.norm(x_true - xk) / np.linalg.norm(x_true)
            errors.append(err)
            display_str += ', err=' + str(err)

        if display:
            print(display_str)

    x, info = spla.cg(A, b, *args, callback=callback, **kwargs)

    return_tuple = (x, info)

    if track_residuals:
        return_tuple = return_tuple + (residuals,)

    if x_true is not None:
        return_tuple = return_tuple + (errors,)

    if iters_to_save is not None:
        return_tuple = return_tuple + (kk_saved, xx_saved)

    return return_tuple
