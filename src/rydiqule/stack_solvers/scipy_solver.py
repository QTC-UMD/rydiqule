import numpy as np

from scipy.integrate import solve_ivp

from typing import Sequence, Callable, Literal
from ..sensor_utils import TimeFunc
from ..exceptions import RydiquleError

def scipy_solve(eoms_base: np.ndarray, const: np.ndarray,
                eom_time_r: np.ndarray, const_r: np.ndarray,
                eom_time_i: np.ndarray, const_i: np.ndarray,
                time_inputs: Sequence[TimeFunc],
                t_eval: np.ndarray, init_cond: np.ndarray,
                eqns: Literal["loop", "comp"] = "loop",
                **kwargs
                ) -> np.ndarray:
    """
    Solve a set of Optical Bloch Equations (OBEs) with rydiqule's time solving convention
    using scipy's `solve_ivp`. 

    Uses matrix components of the equations of motion provided by the methods of a :meth:`~.Sensor`.
    Designed to be used as a wrapped function within :func:`~.timesolvers.solve_time`.
    Builds and solves equations of motion according rydiqule's time solving conventions.
    Sets up and solves dx/dt = A(t)x + b(t)

    Args
    ----
    eoms_base: numpy.ndarray
        The matrix of shape `(*l,n,n)` representing the non time-varying portion of the matrix A
        in the equations of motion.
    const: numpy.ndarray
        The array of shape `(*l, n)` representing the non time-varying portion of the vector b in the
        equations of motion.
    eoms_time_r: numpy.ndarraynumpy
        The matrix of shape `(n_t, *l, n, n)` representing the real time-varying portion of the matrix A,
        where n_t is the length of `time_inputs`.
        The ith slice along the first axis should be multiplied by the real part
        of the ith entry in `time_inputs`.
    const_r: numpy.nd_array
        The matrix of shape `(n_t, *l, n)` representing the real time-varying portion of the vector b,
        where n_t is the length of `time_inputs`.
        The ith slice along the first axis should be multiplied by the real part
        of the ith entry in `time_inputs`.
    eoms_time_i: numpy.ndarray
        The matrix of shape `(n_t, *l, n, n)` representing the imaginary time-varying portion of the matrix A,
        where n_t is the length of `time_inputs`.
        The ith slice along the first axis should be multiplied by the imaginary part
        of the ith entry in `time_inputs`.
    const_i: numpy.nd_array
        The matrix of shape `(n_t, *l, n)` representing the imaginary time-varying portion of the vector b,
        where n_t is the length of `time_inputs`.
        The ith slice along the first axis should be multiplied by the imaginary part
        of the ith entry in `time_inputs`.
    time_inputs: list(callable)
        List of callable functions of length `n_t`.
        The functions should take a single floating point
        as an input representing the time in microseconds,
        and return a real or complex floating point value represent an
        electric field in V/m at that time.
    t_eval: numpy.ndarray
        Array of times to sample the integration at.
    init_cond: (numpy.ndarray)
        Matrix of shape `(*l, n)` representing the initial state of the system.
    eqns: {"loop", "comp"}
        Function used of generating equations of motion. One of "loop" or "comp", corresponding
        to defining time-dependent equations of motion as a loop over time-dependent components
        or with a list comprehension. List comprehensions are preferred for longer solves and
        loops are preferred for shorter solves.
    **kwargs: dict
        Additional keyword arguments passed to the nbkode solver constructor.

    Returns
    -------
    numpy.ndarray
        The matrix solution of shape `(*l,n,n_t)`
        representing the density matrix of the system at each time t.
    """
    
    _derEqns = {"loop": _derEqns_loop, "comp": _derEqn_comp}

    sol_shape = eoms_base.shape[:-1]

    try:
        equations = _derEqns[eqns](eoms_base, const,
                                   eom_time_r, const_r,
                                   eom_time_i, const_i,
                                   time_inputs)
    except KeyError as err:
        raise RydiquleError("\'eqns\' must be one of \'loop\' or \'comp\'.") from err
    
    init_cond = init_cond.ravel()

    method = kwargs.pop("method", "RK45")
    rtol = kwargs.pop("rtol", 1e-6)

    sol_flat = solve_ivp(equations, (t_eval[0], t_eval[-1]),
                         init_cond, t_eval=t_eval,
                         method=method, rtol=rtol,
                         **kwargs).y

    return sol_flat.reshape(sol_shape+(sol_flat.shape[-1],))


def _derEqns_loop(obes_base: np.ndarray, const_base: np.ndarray,
             obes_time_r: np.ndarray, const_r: np.ndarray,
             obes_time_i: np.ndarray,  const_i: np.ndarray,
             time_inputs: Sequence[TimeFunc],
             ) -> Callable[[float, np.ndarray], np.ndarray]:
    """
    Function to build the callable passed to scipy's solve_ivp in :func:`~.scipy_solve`.

    Uses the base and time matrix components of the eoms to build a function of
    vector and scalar time that has the expected input/output of functions passed
    to `scipy.integrate.solve_ivp()`
    """

    t_func_num = obes_time_r.shape[0]
    input_shape = obes_base.shape[:-1]

    def func(t: float, A_flat: np.ndarray) -> np.ndarray:

        # create OBEs at time t
        obe_total = obes_base.copy()
        const_total = const_base.copy()

        for idx in range(t_func_num):
            ti = time_inputs[idx](t)
            obe_total += ti.real*obes_time_r[idx] + ti.imag*obes_time_i[idx]
            const_total += ti.real*const_r[idx] + ti.imag*const_i[idx]

        # Unflatten A for matrix broadcasting
        A_stack = A_flat.reshape(input_shape+(1,))
        result = np.matmul(obe_total, A_stack).squeeze(-1) + const_total
        # Flatten result to match input
        return np.ravel(result)

    return func


def _derEqn_comp(obes_base: np.ndarray, const: np.ndarray,
            obes_time_r: np.ndarray, const_r: np.ndarray,
            obes_time_i: np.ndarray,  const_i: np.ndarray,
            time_inputs: Sequence[TimeFunc],
            ) -> Callable[[float, np.ndarray], np.ndarray]:
    """
    Function to build the callable passed to scipy's solve_ivp in :func:`~.scipy_solve`.

    Uses the base and time matrix components of the eoms to build a function of vector and
    scalar time that has the expected input/output of functions passed to
    `scipy.integrate.solve_ivp()`
    """
    input_shape = obes_base.shape[:-1]
    l = obes_time_r.shape[0]

    def func(t: float, A_flat: np.ndarray) -> np.ndarray:

        obe_total = obes_base + np.sum([time_inputs[idx](t).real*obes_time_r[idx]
                                        + time_inputs[idx](t).imag*obes_time_i[idx]
                                        for idx in range(l)], axis=0)
        const_total = const + np.sum([time_inputs[idx](t).real*const_r[idx]
                                      + time_inputs[idx](t).imag*const_i[idx]
                                      for idx in range(l)], axis=0)

        # Unflatten A for matrix broadcasting
        A_stack = A_flat.reshape(input_shape+(1,))
        result = np.matmul(obe_total, A_stack).squeeze(-1) + const_total
        # Flatten result to match input
        return np.ravel(result)
 
    return func
