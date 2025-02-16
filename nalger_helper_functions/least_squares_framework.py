import typing as typ

import nalger_helper_functions.tree_linalg as tla

__all__ = [
    'misfit',
    'misfit_gradient',
    'misfit_gauss_newton_hessian_matvec',
    'objective',
    'gradient',
    'gauss_newton_hessian_matvec',
]


Param          = typ.TypeVar('Param')
ParamTangent   = typ.TypeVar('ParamTangent')
ParamCoTangent = typ.TypeVar('ParamCoTangent')

Inputs = typ.TypeVar('Inputs')

State           = typ.TypeVar('State')
StateTangent    = typ.TypeVar('StateTangent')
StateCoTangent  = typ.TypeVar('StateCoTangent')

ForwardAux      = typ.TypeVar('ForwardAux')
ForwardJvpAux   = typ.TypeVar('ForwardJvpAux')
ForwardVjpAux   = typ.TypeVar('ForwardVjpAux')
LossAux         = typ.TypeVar('LossAux')

Outputs             = typ.TypeVar('Outputs')
OutputsTangent      = typ.TypeVar('OutputsTangent')
OutputsCoTangent    = typ.TypeVar('OutputsCoTangent')

Scalar  = typ.TypeVar('Scalar')


default_observe = lambda u: u
default_observe_jvp = lambda u, du: du
default_observe_vjp = lambda u, dy: dy

def default_loss(y: Outputs, y_true: Outputs) -> typ.Tuple[Scalar, LossAux]:
    all_num_squared = tla.tree_normsquared_leaves(tla.tree_sub(y_true, y))
    all_den_squared = tla.tree_normsquared_leaves(y_true)
    all_relative_squared_errors = tla.tree_div(all_num_squared, all_den_squared)
    J = 0.5 * tla.tree_sum(all_num_squared)
    return J, all_relative_squared_errors

def default_loss_vjp(y: Outputs, y_true: Outputs, dJ: Scalar) -> OutputsCoTangent:
    return tla.tree_scale(tla.tree_sub(y, y_true), dJ)


def default_loss_gnhvp(y: Outputs, dy: OutputsTangent) -> OutputsCoTangent:
    return dy

default_regularization = lambda U: 0.5 * tla.tree_normsquared(U)
default_regularization_gradient = lambda m: m
default_regularization_gnhvp = lambda m, dm: dm


def misfit(
        m:              Param,
        x:              Inputs,
        y_true:         Outputs, # true observations
        forward_map:    typ.Callable[[Param, Inputs],       typ.Tuple[State,    ForwardAux]],
        observe:        typ.Callable[[State],               Outputs]                        = default_observe,
        loss:   typ.Callable[[Outputs, Outputs],    typ.Tuple[Scalar,   LossAux]]   = default_loss, # true_obs, obs -> J, J_aux
) -> typ.Tuple[
    Scalar, # misfit Jd
    typ.Tuple[
        ForwardAux,  # Auxiliary data computed when computing the forward map
        State,
        Outputs,  # predicted outputs y based on the model m
        LossAux,
    ]
]:
    u, f_aux    = forward_map(m, x)
    y           = observe(u)
    J, J_aux    = loss(y, y_true)
    return J, (f_aux, u, y, J_aux)


def misfit_gradient(
        m:      Param,
        x:      Inputs,
        u:      State,
        f_aux:  ForwardAux,
        y:      Outputs,
        y_true: Outputs,
        forward_map_vjp:    typ.Callable[[Param, Inputs, StateCoTangent, ForwardAux],   typ.Tuple[ParamCoTangent, ForwardVjpAux]],
        observe_vjp:        typ.Callable[[State, OutputsCoTangent],                     StateCoTangent]   = default_observe_vjp,
        loss_vjp:   typ.Callable[[Outputs, Outputs, Scalar],                    OutputsCoTangent] = default_loss_vjp,
) -> typ.Tuple[
    ParamCoTangent, # gradient g
    ForwardVjpAux,
]:
    z1: OutputsCoTangent    = loss_vjp(y, y_true, 1.0)
    z2: StateCoTangent      = observe_vjp(u, z1)
    g, fvjp_aux = forward_map_vjp(m, x,  z2, f_aux)
    return g, fvjp_aux


def misfit_gauss_newton_hessian_matvec(
        dm: ParamTangent,
        m:  Param,
        x:  Inputs,
        u:  State,
        f_aux: ForwardAux,
        y: Outputs,
        forward_map_jvp:    typ.Callable[[Param, Inputs, ParamTangent, ForwardAux],   typ.Tuple[StateTangent, ForwardJvpAux]],
        forward_map_vjp:    typ.Callable[[Param, Inputs, StateCoTangent, ForwardAux], typ.Tuple[ParamCoTangent, ForwardVjpAux]],
        observe_jvp:        typ.Callable[[State, StateTangent],     OutputsTangent] = default_observe_jvp,
        observe_vjp:        typ.Callable[[State, OutputsCoTangent], StateCoTangent] = default_observe_vjp,
        loss_gnhvp: typ.Callable[[Outputs, OutputsTangent], OutputsCoTangent] = default_loss_gnhvp,
) -> typ.Tuple[
    ParamCoTangent, # H @ dm, Gauss-Newton Hessian vector product
    typ.Tuple[
        ForwardJvpAux,
        ForwardVjpAux,
    ]
]:
    du, fjvp_aux = forward_map_jvp(m, x, dm, f_aux)
    dy = observe_jvp(u, du)
    dy2 = loss_gnhvp(y, dy)
    du2 = observe_vjp(u, dy2)
    Hdm, fvjp_aux = forward_map_vjp(m, x, du2, f_aux)
    return Hdm, (fjvp_aux, fvjp_aux)


def objective(
        m:      Param,
        x:      Inputs,
        y_true: Outputs,
        forward_map:    typ.Callable[[Param, Inputs], typ.Tuple[State, ForwardAux]],
        a_reg: Scalar,  # regularization parameter
        observe:        typ.Callable[[State],               Outputs]                    = default_observe,
        loss:   typ.Callable[[Outputs, Outputs],    typ.Tuple[Scalar, LossAux]] = default_loss, # true_obs, obs -> J, J_aux
        regularization_function: typ.Callable[[Param], Scalar] = default_regularization,
) -> typ.Tuple[
    Scalar, # J, objective
    typ.Tuple[
        Scalar, # Jd, misfit component
        Scalar, # Jr, regularization component
        ForwardAux,
        State,
        Outputs, # outputs y
        LossAux,
    ]
]:
    Jd, (f_aux, u, y, Jd_aux) = misfit(m, x, y_true, forward_map, observe, loss)
    Jr0 = regularization_function(m)
    Jr = tla.tree_scale(Jr0, a_reg)
    J = tla.tree_add(Jd, Jr)
    return J, (Jd, Jr, f_aux, u, y, Jd_aux)


def gradient(
        m: Param,
        x: Inputs,
        u: State,
        f_aux: ForwardAux,
        y:      Outputs,
        y_true: Outputs,
        a_reg: Scalar,
        forward_map_vjp:    typ.Callable[[Param, Inputs, StateCoTangent, ForwardAux],   typ.Tuple[ParamCoTangent, ForwardVjpAux]],
        observe_vjp:        typ.Callable[[State, OutputsCoTangent],     StateCoTangent]   = default_observe_vjp,
        loss_vjp:   typ.Callable[[Outputs, Outputs, Scalar],    OutputsCoTangent] = default_loss_vjp,
        regularization_gradient: typ.Callable[[Param], ParamCoTangent] = default_regularization_gradient,
) -> typ.Tuple[
    ParamCoTangent, # gradient g
    typ.Tuple[
        ParamCoTangent, # gd, misfit component
        ParamCoTangent, # gr, # regularization component
        ForwardVjpAux,
    ]
]:
    gd, fvjp_aux = misfit_gradient(m, x, u, f_aux, y, y_true, forward_map_vjp, observe_vjp, loss_vjp)
    gr0 = regularization_gradient(m)
    gr = tla.tree_scale(gr0, a_reg)
    g = tla.tree_add(gd, gr)
    return g, (gd, gr, fvjp_aux)


def gauss_newton_hessian_matvec(
        dm: ParamTangent,
        m: Param,
        x: Inputs,
        u: State,
        f_aux: ForwardAux,
        y: Outputs,
        forward_map_jvp: typ.Callable[[Param, Inputs, ParamTangent, ForwardAux], typ.Tuple[StateTangent, ForwardJvpAux]],
        forward_map_vjp: typ.Callable[[Param, Inputs, StateCoTangent, ForwardAux], typ.Tuple[ParamCoTangent, ForwardVjpAux]],
        a_reg: Scalar,
        observe_jvp: typ.Callable[[State, StateTangent], OutputsTangent]             = default_observe_jvp,
        observe_vjp: typ.Callable[[State, OutputsCoTangent], StateCoTangent]         = default_observe_vjp,
        loss_gnhvp: typ.Callable[[Outputs, OutputsTangent], OutputsCoTangent] = default_loss_gnhvp,
        regularization_gnhvp: typ.Callable[[Param, ParamTangent], ParamCoTangent] = default_regularization_gnhvp,
) -> typ.Tuple[
    ParamCoTangent, # H @ dm, Gauss-Newton Hessian matvec
    typ.Tuple[
        ParamCoTangent, # Hd @ dm, misfit component
        ParamCoTangent, # Hr @ dm, misfit component
        ForwardJvpAux,
        ForwardVjpAux,
    ]
]:
    Hd_dm, (fjvp_aux, fvjp_aux) = misfit_gauss_newton_hessian_matvec(
        dm, m, x, u, f_aux, y,
        forward_map_jvp, forward_map_vjp,
        observe_jvp, observe_vjp,
        loss_gnhvp,
    )
    Hr_dm0 = regularization_gnhvp(m, dm)
    Hr_dm = tla.tree_scale(Hr_dm0, a_reg)
    H_dm = tla.tree_add(Hd_dm, Hr_dm)
    return H_dm, (Hd_dm, Hr_dm, fjvp_aux, fvjp_aux)

