import warnings

import numpy as np
import psutil

from typing import Sequence, Callable, Literal
from ..sensor_utils import TimeFunc
from ..exceptions import RydiquleError


def cyrk_solve(eoms_base: np.ndarray, const_base: np.ndarray,
               eom_time_r: np.ndarray, const_r: np.ndarray,
               eom_time_i: np.ndarray, const_i: np.ndarray,
               time_inputs: Sequence[TimeFunc],
               t_eval: np.ndarray, init_cond: np.ndarray,
               eqns: Literal["orig", "flat"] = "orig",
               **kwargs
               ) -> np.ndarray:
    """
    Solve a set of Optical Bloch Equations (OBEs) with rydiqule's time solving convention
    using CyRK's `pysolve_ivp`.

    Uses matrix components of the equations of motion provided by the methods of a :meth:`~.Sensor`.
    Designed to be used as a wrapped function within :func:`~.timesolvers.solve_time`.
    Builds and solves equations of motion according rydiqule's time solving conventions.
    Sets up and solves dx/dt = A(t)x + b(t)

    For larger solve systems, `max_ram_MB` kwarg for `pysolve_ivp` will likely need to be increased
    from its default of 2000.

    Args
    ----
    eoms_base: numpy.ndarray
        The matrix of shape `(*l,n,n)` representing the non time-varying portion of the matrix A
        in the equations of motion.
    const_base: numpy.ndarray
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
    init_cond: numpy.ndarray
        Matrix of shape `(*l, n)` representing the initial state of the system.
    eqns: {"orig", "flat"}
        Method used to generate the derivative equations.
        Options are orig (which uses a numpy reshaping approach)
        and flat (which uses flat array indexing).
    **kwargs: dict:
        Additional keyword arguments passed to `pysolve_ivp`.

    Returns
    -------
    numpy.ndarray
        The matrix solution of shape `(*l,n,n_t)`
        representing the density matrix of the system at each time t.

    Raises
    ------
    RydiquleError
        If system size exceeds cyrk backend limit of 65535 equations.
        If we see this error a lot, consider getting CyRK project to increase it
        by changing type of `y_size` from unisgned short.
    """

    try:
        import numba as nb
        from CyRK import pysolve_ivp
    except ImportError as e:
        raise RydiquleError('CyRK backend not installed') from e
        
    try:
        fns = _eqnsGen[eqns]
    except KeyError as err:
        raise RydiquleError("\'eqns\' must be one of \'orig\' or \'flat\'") from err

    to_compile = [not nb.extending.is_jitted(f) for f in time_inputs]
    complex_out = [isinstance(f(0.0), complex) for f in time_inputs]
    time_inputs_compiled = tuple(nb.njit("c16(f8)", cache=True)(f) if t and c
                                 else nb.njit("f8(f8)", cache=True)(f)
                                 if c else f
                                 for c,t,f in zip(to_compile,complex_out,time_inputs))
    with warnings.catch_warnings():
        # ignore first-class function warning
        warnings.simplefilter("ignore", category=nb.NumbaExperimentalFeatureWarning)
        equations = fns(eoms_base, const_base,
                        eom_time_r, const_r,
                        eom_time_i, const_i,
                        time_inputs_compiled
                        )

    # enforce default arguments consistent with scipy solver
    method = kwargs.pop("method", "RK45")
    rtol = kwargs.pop("rtol", 1e-6)
    max_ram_MB = kwargs.pop("max_ram_MB", max(psutil.virtual_memory().available/(1024**2)/10, 2_000))
    
    result = pysolve_ivp(equations, (t_eval[0], t_eval[-1]),
                         init_cond.ravel(),
                         t_eval=t_eval,
                         method=method, rtol=rtol,
                         max_ram_MB=max_ram_MB,
                         pass_dy_as_arg=True,
                         **kwargs)
    
    if not result.success:
        result.print_diagnostics()
        raise RydiquleError(f"Integration failed ({result.error_code}): {result.message}")

    sol_shape = eoms_base.shape[:-1]
    return result.y.reshape(sol_shape + (result.y.shape[-1],))

def _derEqns(obes_base: np.ndarray, const_base: np.ndarray,
             obes_time_r: np.ndarray, const_r: np.ndarray,
             obes_time_i: np.ndarray, const_i: np.ndarray,
             time_inputs: Sequence[TimeFunc]
             ) -> Callable[[float, np.ndarray, np.ndarray], None]:
    """
    Function to build the callable passed to CyRK's cyrk_ode cython solver.

    Note that `time_inputs` functions must be njit compiled.

    Uses the base and time matrix components of the eoms to build
    a function of vector and scalar time
    that has the expected input/output of functions passed to `cyrk.cyrk_ode()`
    """
    import numba as nb

    t_func_num = obes_time_r.shape[0]
    input_shape = obes_base.shape[:-1]
    stack_shape = obes_base.shape[:-2]

    @nb.njit("void(f8[::1], f8, f8[::1])")
    def func(result_out: np.ndarray, t: float, A_flat: np.ndarray):

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
            result[sidx] = np.dot(obe_total[sidx], A_stack[sidx])
        # add const values, note: uses broadcasting to handle doppler axis
        result += const_total
        # load stacked result into flat output array
        for i, v in enumerate(result.flat):
            result_out[i] = v

    return func

def _derEqns_flat(obes_base: np.ndarray, const_base: np.ndarray,
                  obes_time_r: np.ndarray, const_r: np.ndarray,
                  obes_time_i: np.ndarray, const_i: np.ndarray,
                  time_inputs: Sequence[TimeFunc]
                  ) -> Callable[[float, np.ndarray, np.ndarray], None]:
    """
    Function to build the callable passed to CyRK's pysolve_ivp cython solver.

    Note that `time_inputs` functions must be njit compiled.

    Uses the base and time matrix components of the eoms to build
    a function of vector and scalar time
    that has the expected input/output of functions passed to `cyrk.pysolve_ivp()`

    This implementation is explicitly flat and avoids extra array allocations.
    """
    import numba as nb

    if obes_base.shape[:-1] != const_base.shape:
        raise RydiquleError("CyRK flat solver incompatible with doppler solves")

    # basis dimension size
    b = obes_base.shape[-1]
    b2 = b**2
    # time function dimension size
    t_func_num = obes_time_r.shape[0]
    # flatten eqns arrays
    obes_base = obes_base.reshape(-1)
    const_base = const_base.reshape(-1)
    obes_time_r = obes_time_r.reshape((t_func_num, -1))
    obes_time_i = obes_time_i.reshape((t_func_num, -1))
    const_r = const_r.reshape((t_func_num, -1))
    const_i = const_i.reshape((t_func_num, -1))

    @nb.njit("void(f8[::1], f8, f8[::1])")
    def func(result_out: np.ndarray, t: float, A_flat: np.ndarray):

        # calculate time inputs at time t
        ts = np.zeros(t_func_num, dtype=np.complex128)
        for idx in range(t_func_num):
            ts[idx] = time_inputs[idx](t)

        for i in range(result_out.size):
            # start result with time-independent constant part
            result_out[i] = const_base[i]
            # define idx for this loop separately
            const_time_idx = i%b
            for idx in range(t_func_num):
                # add time-dependent const part
                result_out[i] += ts[idx].real*const_r[idx, const_time_idx] + ts[idx].imag*const_i[idx, const_time_idx]
            
            for j in range(b):
                # define indeces for this step
                obe_idx = i*b+j
                obe_time_idx = obe_idx%b2
                A_idx = (i//b)*b+j
                # add time-independent obe part
                # implements einsum('...ij,...j', obes, A)
                result_out[i] += obes_base[obe_idx] * A_flat[A_idx]
                for idx in range(t_func_num):
                    # add time-dependent obe part
                    result_out[i] += (ts[idx].real*obes_time_r[idx, obe_time_idx]
                                      + ts[idx].imag*obes_time_i[idx, obe_time_idx]) * A_flat[A_idx]

    return func

_eqnsGen = {"orig": _derEqns, "flat": _derEqns_flat}