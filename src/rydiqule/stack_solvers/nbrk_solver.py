import warnings

import numpy as np

try:
    import numba as nb
    from CyRK import nbrk_ode
    nbrk_available = True
except ImportError as e:
    nbrk_available = False
    nbrk_import_error = e

from typing import Sequence, Callable
from rydiqule.sensor_utils import TimeFunc


def nbrk_solve(eoms_base: np.ndarray, const_base: np.ndarray,
               eom_time_r: np.ndarray, const_r: np.ndarray,
               eom_time_i: np.ndarray, const_i: np.ndarray,
               time_inputs: Sequence[TimeFunc],
               t_eval: np.ndarray, init_cond: np.ndarray, **kwargs
               ) -> np.ndarray:
    """
    Solve a set of Optical Bloch Equations (OBEs) with rydiqule's time solving convention
    using CyRK's `nbrk_ode`.

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
        Return type of each function must be the same for all inputs t.
    t_eval: numpy.ndarray
        Array of times to sample the integration at.
        This array must have dtype of float64.
    init_cond: (numpy.ndarray)
        Matrix of shape `(*l, n)` representing the initial state of the system.
    **kwargs: dict
        Additional keyword arguments passed to nbrk_ode.

    Returns
    -------
    numpy.ndarray
        The matrix solution of shape `(*l,n,n_t)`
        representing the density matrix of the system at each time t.
    """

    if not nbrk_available:
        raise ImportError('CyRK backend not installed') from nbrk_import_error

    to_compile = [not nb.extending.is_jitted(f) for f in time_inputs]
    complex_out = [isinstance(f(0.0), complex) for f in time_inputs]
    time_inputs_compiled = tuple(nb.njit("c16(f8)", cache=True)(f) if t
                                 else nb.njit("f8(f8)", cache=True)(f)
                                 if c else f
                                 for c,t,f in zip(to_compile,complex_out,time_inputs))

    equations = _derEqns(eoms_base, const_base,
                         eom_time_r, const_r,
                         eom_time_i, const_i,
                         time_inputs_compiled
                         )

    sol_shape = eoms_base.shape[:-1]
    with warnings.catch_warnings():
        # ignore first-class function warning
        warnings.simplefilter("ignore", category=nb.NumbaExperimentalFeatureWarning)
        nb_times, nb_sols, nb_success, nb_message = nbrk_ode(equations, (t_eval[0], t_eval[-1]),
                                                             init_cond.ravel(),
                                                             t_eval=t_eval,
                                                             **kwargs)

    return nb_sols.reshape(sol_shape+(nb_sols.shape[-1],))


def _derEqns(obes_base: np.ndarray, const_base: np.ndarray,
             obes_time_r: np.ndarray, const_r: np.ndarray,
             obes_time_i: np.ndarray,  const_i: np.ndarray,
             time_inputs: Sequence[TimeFunc]
             ) -> Callable[[float, np.ndarray], np.ndarray]:
    """
    Function to build the callable passed to CyRK's nbrk_ode numba solver.

    Note that `time_inputs` functions must be njit compiled.

    Uses the base and time matrix components of the eoms to build
    a function of vector and scalar time
    that has the expected input/output of functions passed to `cyrk.nbrk_ode()`
    """

    t_func_num = obes_time_r.shape[0]
    input_shape = obes_base.shape[:-1]
    stack_shape = obes_base.shape[:-2]

    @nb.njit
    def func(t: float, A_flat: np.ndarray):

        # create OBEs at time t
        obe_total = obes_base.copy()
        const_total = const_base.copy()

        for idx in range(t_func_num):
            ti = time_inputs[idx](t)
            obe_total += ti.real*obes_time_r[idx] + ti.imag*obes_time_i[idx]
            const_total += ti.real*const_r[idx] + ti.imag*const_i[idx]

        # reshape input to stack shape
        A_stack = A_flat.reshape(input_shape)
        result = np.empty_like(A_stack)
        # matrix multiply obes with input for each parameter of stack
        for sidx in np.ndindex(stack_shape):
            A = A_stack[sidx]
            result[sidx] = np.dot(obe_total[sidx], A)
        # add const values, note: uses broadcasting to handle doppler axis
        result += const_total
        # flatten output
        return np.ravel(result)

    return func
