"""
Solvers for time domain analysis with an arbitrary RF field
"""

import numpy as np
from importlib.metadata import version
from copy import deepcopy

from .sensor_utils import _hamiltonian_term, generate_eom, make_real, TimeFunc, check_positive_semi_definite, _squeeze_dims
from .sensor_solution import Solution
from .sensor import Sensor  # only needing for type hinting
from .slicing.slicing import matrix_slice, get_slice_num_t
from .exceptions import RydiquleError

from rydiqule.solvers import solve_steady_state
from rydiqule.doppler_utils import *
from rydiqule.stack_solvers.scipy_solver import scipy_solve
from rydiqule.stack_solvers.cyrk_solver import cyrk_solve

from typing import Optional, Tuple, List, Callable, Literal, Union, Dict

stack_solvers: Dict[str, Callable] = {"scipy": scipy_solve,
                                      "cyrk": cyrk_solve,
                                      }

solver_type = Union[Callable, Literal['scipy', 'cyrk']]

def solve_time(sensor: Sensor, end_time: float, num_pts: int,
               init_cond: Optional[np.ndarray] = None, doppler: bool = False,
               doppler_mesh_method: Optional[MeshMethod] = None, sum_doppler: bool = True,
               weight_doppler: bool = True,
               n_slices: Union[int, None] = None, solver: solver_type = "scipy",
               **kwargs) -> Solution:
    """
    Solves the response of the optical sensor in the time domain given the its time-dependent inputs
    
    If insuffucent system memory is available to solve the system all at once, 
    system is broken into "slices" of manageable memory footprint which are solved indivudually. 
    This slicing behavior does not affect the result.
    All couplings that include a "time_dependence" argument will be solved in the time domain. 
    
    A number of solver backends work with rydiqule, but the default `"scipy"` ivp solver is the
    is recommended backend in almost all cases, as it is the most fully-featured and
    documented. Advanced users have the ablity to define their own solver backends
    by creating a function that follows the call signature for rydiqule timesolver
    backends. Additional arguments to the solver backend can be supplied with `**kwargs`.
    
    Parameters
    -----------
    sensor : :class:`~.Sensor`
        The sensor object representing the atomic/laser arrangement of the system.
    end_time : float  
        Amount of time, in microseconds, for which to simulate the system
    num_pts : int
        The number of points along the range `(0, end_time)` for which
        the solution is evaluated. This does not affect the number of funtion
        evaluations during the solve, rather the spacing of the points in 
        the reported solution.
    init_cond : numpy.ndarray or `None`, optional 
        Density matrix representing the initial state of the system. 
        If specified, the shape should be either `(n)` in the case of a single
        initial condition for all parameter values, or should be of shape `(*l, n)`
        matching the output shape of a steady state solve if the initial condition
        may be different for different combinations of parameters. 
        If `None`, will solve the problem in the steady state with all time-dependent fields
        at their :math:`t=0` value and use the solution as the initial condition.
        Other possible manual options might include a matrix populated by zeros representing the
        entire population in the ground state. Defaults to `None`.
    doppler : bool, optional
        Whether to account for doppler shift among moving atoms in
        the gas. If True, the solver will implicitly define a velocity distribution
        for particles in the cell, solve the problem for each velocity class,
        and return a weighted average of the results. Note that solving in this
        manner carries a substantial performance penalty, as each doppler velocity class
        is solved as its own problem. If solved with doppler, only axis specified by a `"kvec"'
        argument in one of the sensor couplings will be average over. The time solver currently 
        supports doppler averaging in any number of spatial dimensions, up to the limit of 
        3 imposed by the macroscopic physical world. Defaults to `False`.
    doppler_mesh_method : dict, optional
        Dictionary that controls the doppler meshing method. Exact details of this are
        found in the documentation of :func:`~doppler_classes`. Ignored if 
        `doppler=False`. Default is `None`.
    sum_doppler : bool, optional
        Whether to average over doppler classes after the solve
        is complete. Setting to false will not perform the sum, allowing viewing
        of the weighted results of the solve for each doppler class. Ignored
        if `doppler=False`. Default is `True`.
    weight_doppler : bool
        Whether to apply weights to doppler solution to perform
        averaging. If `False`, will **not** apply weights or perform a doppler_average,
        regardless of the value of `sum_doppler`. Changing from default intended
        only for internal use. Ignored if `doppler=False` or `sum_doppler=False`. 
        Default is `True`.
    n_slices : int or None, optional
        How many sets of equations to break the full equations into.
        The actual number of slices will be the largest between this value and the minumum
        number of slices to solve the system without a memory error. If `None`, solver uses the
        minimum number of slices required to solve without a `memoryError`. Defaults to None.
    solver : {"scipy", "cyrk"} or callable
        The backend solver used to solve the ivp generated by the sensor.
        All string values correspond to backend solvers built in to rydiqule.
        Valid string values are:

            - "scipy": Solves equations with :func:`scipy:scipy.integrate.solve_ivp`.
              The default, most stable, and well-supported option.
            - "cyrk": Solves jit-compiled equations with a cython compiled RK solver from `CyRK`.
              Due to some jit compilation, only faster for moderate length problems
              (ie problems with a moderate number of required time steps).

        Additionally, can be specified with a callable that matches
        rydiqule's time-solver convention,
        enabling using a custom solver backend.

        .. note::

            Unless otherwise noted, backends other than scipy are considered experimental.
            Issues with their use are considered features not fully implemented rather than
            bugs.

    **kwargs: Additional keyword arguments passed to the backend solver.
        See documentation of the relevant solver (i.e. :func:`scipy:scipy.integrate.solve_ivp`)
        for details and supported arguments.

    Returns
    -------
    :class:`~.Solution`
        An object contining the solution and related information.
        Timesolver-specific defined attributes are `t`  and `init_cond`,
        corresponding respectively to the times at which the solution is sampled and
        the initial conditions used for the solve.
    
    Examples
    --------
    A basic solve for a 3-level system would have a "density matrix" solution of size 8 (3^2-1).
    Here we use a trivial time dependence for demonstration purposes, but in practice the time
    dependence is likely more complicated. Below the most basic use of `solve_time` is demonstrated
    
    >>> s = rq.Sensor(3)
    >>> td = lambda t: 1
    >>> s.add_coupling((0,1), detuning = 1, rabi_frequency=1)
    >>> s.add_coupling((1,2), detuning = 2, rabi_frequency=2, time_dependence=td)
    >>> s.add_transit_broadening(0.1)
    >>> end_time = 10 #microseconds
    >>> n_pts = 1000 #interpoleted points in solution
    >>> sol = rq.solve_time(s, end_time, n_pts)
    >>> print(type(sol))
    <class 'rydiqule.sensor_solution.Solution'>
    >>> print(type(sol.rho))
    <class 'numpy.ndarray'>
    >>> print(sol.rho.shape)
    (1000, 8)
    
    Defining an array-like parameter will automatically calculate the density matrix solution
    for every value. Here we use 11 values, resulting in 11 density matrices. The `axis_labels`
    attribute of the solution can clarify which axes are which.
    
    >>> s = rq.Sensor(3)
    >>> td = lambda t: 1
    >>> det = np.linspace(-1,1,11)
    >>> s.add_coupling((0,1), detuning = det, rabi_frequency=1)
    >>> s.add_coupling((1,2), detuning = 2, rabi_frequency=2, time_dependence=td)
    >>> s.add_transit_broadening(0.1)
    >>> sol = rq.solve_time(s, end_time, n_pts)
    >>> print(sol.rho.shape)
    (11, 1000, 8)
    >>> print(sol.axis_labels)
    ['(0,1)_detuning', 'time', 'density_matrix']
        
    As expected, multiple axes of scanned parameters are handled the same way as they are in the
    steady-state case, with the expected additions from the time solver.

    >>> s = rq.Sensor(3)
    >>> td = lambda t: 1
    >>> det = np.linspace(-1,1,11)
    >>> s.add_coupling((0,1), detuning = det, rabi_frequency=1)
    >>> s.add_coupling((1,2), detuning = det, rabi_frequency=2, time_dependence=td)
    >>> s.add_transit_broadening(0.1)
    >>> sol = rq.solve_time(s, end_time, n_pts)
    >>> print(sol.rho.shape)
    (11, 11, 1000, 8)
    >>> print(sol.axis_labels)
    ['(0,1)_detuning', '(1,2)_detuning', 'time', 'density_matrix']
    
    If the solve uses doppler broadening, all doppler classes will be computed a weighted average
    will be taken over the doppler axis, and the shape of the solution will not change. While this
    is the desired behavior in most situations, the `sum_doppler` argument can be used to override
    this behavior, leave (or more) solution axis corresponding to different doppler classes.
    
    >>> s = rq.Sensor(3, vP=1)
    >>> td = lambda t: 1
    >>> det = np.linspace(-1,1,11)
    >>> s.add_coupling((0,1), detuning = det, rabi_frequency=1, kvec=(4,0,0))
    >>> s.add_coupling((1,2), detuning = 2, rabi_frequency=2, kvec=(-4,0,0), time_dependence=td)
    >>> s.add_transit_broadening(0.1)
    >>> end_time = 10 #microseconds
    >>> n_pts = 1000 #interpoleted points in solution
    >>> sol_dop = rq.solve_time(s, end_time, n_pts, doppler=True)
    >>> sol_dop_nosum = rq.solve_time(s, end_time, n_pts, doppler=True, sum_doppler=False)
    >>> print(sol_dop.rho.shape)
    (11, 1000, 8)
    >>> print(sol_dop_nosum.rho.shape)
    (561, 11, 1000, 8)
    >>> print(sol_dop.axis_labels)
    ['(0,1)_detuning', 'time', 'density_matrix']
    >>> print(sol_dop_nosum.axis_labels)
    ['doppler_0', '(0,1)_detuning', 'time', 'density_matrix']

    """
    if len(sensor.couplings_with("time_dependence")) == 0:
        raise RydiquleError("At least one time-dependent coupling is required")

    if isinstance(solver, str):
        try:
            solver = stack_solvers[solver]
        except KeyError as err:
            raise RydiquleError(
                f"{solver} is not a built-in solver."
                f" Supported built-in solvers are {list(stack_solvers.keys())}") from err
    
    if not callable(solver):
        raise RydiquleError("Solvers must be callable functions")
    
    solution = Solution()

    # relevant sensor-related quantities
    stack_shape = sensor._stack_shape()
    basis_size = sensor.basis_size
    spatial_dim = sensor.spatial_dim()

    # set up time parameters
    time_range = (0.0, end_time)
    t_eval = np.linspace(*time_range, num=num_pts, dtype=np.float64)

    # initialize doppler-related quantities
    doppler_axis_shape: Tuple[int, ...] = ()
    dop_classes = None
    doppler_shifts = None
    out_doppler_axes: Tuple[slice, ...] = ()
    doppler_axes: Tuple[slice, ...] = ()

    # update doppler-related values
    if doppler:
        dop_classes = doppler_classes(method=doppler_mesh_method)
        doppler_shifts = sensor.get_doppler_shifts()
        doppler_axis_shape = tuple(len(dop_classes) for _ in range(spatial_dim))
        doppler_axes = tuple(slice(None) for _ in range(spatial_dim))

        if not sum_doppler:
            out_doppler_axes = doppler_axes

    if init_cond is None:
        init_cond = solve_steady_state(
            sensor, doppler=doppler, weight_doppler=False, sum_doppler=False,
            doppler_mesh_method=doppler_mesh_method
            ).rho
    else:
        # check that user provided init_cond are physical
        check_positive_semi_definite(init_cond)

    # use available memory to figure out how to slice the hamiltonian
    n_slices, out_sol_shape = get_slice_num_t(basis_size, stack_shape, doppler_axis_shape,
                                                  num_pts, sum_doppler, weight_doppler, n_slices)

    if n_slices > 1:
        print(f"Breaking equations of motion into {n_slices} sets of equations...")

    # allocate arrays
    hamiltonians = sensor.get_hamiltonian()
    hamiltonians_time, hamiltonians_time_i = sensor.get_time_hamiltonian_components()
    n_time = len(hamiltonians_time)
    time_functions = sensor.get_time_dependence()
    gamma = sensor.decoherence_matrix()
    sols = np.zeros(out_sol_shape, dtype="float64")
    
    n_slices_true = sum(1 for _ in matrix_slice(gamma, n_slices=n_slices))

    for i, (idx, H, G, *time_hams) in enumerate(matrix_slice(hamiltonians, gamma,
                                                             *hamiltonians_time,
                                                             *hamiltonians_time_i,
                                                             n_slices=n_slices)):
        
        if n_slices_true > 1:
            print(f"Solving equation slice {i+1}/{n_slices_true}", end='\r')
            
        Ht = np.array(time_hams[:n_time])
        Ht_i = np.array(time_hams[n_time:])
        
        full_idx = (*out_doppler_axes, *idx, slice(None))
        sols[full_idx] = _solve_hamiltonian_stack(H, Ht, Ht_i,
                                                  G, time_functions, t_eval,
                                                  init_cond[(*doppler_axes, *idx)], solver,
                                                  doppler=doppler,
                                                  dop_classes=dop_classes, sum_doppler=sum_doppler,
                                                  weight_doppler=weight_doppler,
                                                  doppler_shifts=doppler_shifts,
                                                  spatial_dim=spatial_dim, **kwargs)

    # save results to Solution object
    solution.rho = sols

    # specific to observable calculations
    solution._eta = sensor.eta
    solution._kappa = sensor.kappa
    solution._cell_length = sensor.cell_length
    solution._beam_area = sensor.beam_area
    solution._probe_freq = sensor.probe_freq
    solution._probe_tuple = sensor.probe_tuple

    sensor._expand_dims()
    solution.couplings = deepcopy(sensor.couplings)
    _squeeze_dims(sensor.couplings)

    solution.axis_labels = ([f'doppler_{i:d}' for i in range(spatial_dim) if not sum_doppler]
                            + sensor.axis_labels()
                            + ["time", "density_matrix"])
    solution.axis_values = ([dop_classes for i in range(spatial_dim) if not sum_doppler]
                            + [val for _,_,val,_ in sensor.variable_parameters()]
                            + [t_eval, sensor.dm_basis()])
    solution.dm_basis = sensor.dm_basis()
    solution.rq_version = version("rydiqule")
    solution.doppler_classes = dop_classes
    
    # time solver specific
    solution.t = t_eval
    solution.init_cond = init_cond

    return solution


def _solve_hamiltonian_stack(hamiltonians_base: np.ndarray, hamiltonians_time: np.ndarray,
                             hamiltonians_time_i: np.ndarray, gamma_matrix: np.ndarray,
                             time_functions: List[TimeFunc], t_eval: np.ndarray,
                             init_cond: np.ndarray, solver, doppler: bool = False,
                             dop_classes: Optional[np.ndarray] = None, sum_doppler: bool = True,
                             weight_doppler: bool = True,
                             doppler_shifts: Optional[np.ndarray] = None,
                             spatial_dim: int = 0, **kwargs
                             ) -> np.ndarray:

    """
    Internal funtions which solve the equations of a given hamiltonian stack 
    with the given parameters.
    """
    eom_base, const = generate_eom(hamiltonians_base, gamma_matrix,
                                   remove_ground_state=True, real_eom=True)
    eom_time_r, const_r = generate_eom_time(hamiltonians_time)
    eom_time_i, const_i = generate_eom_time(hamiltonians_time_i)

    if doppler:
        assert dop_classes is not None and doppler_shifts is not None
        dop_velocities, dop_volumes = doppler_mesh(dop_classes, spatial_dim)
        eom_base = get_doppler_equations(eom_base, doppler_shifts, dop_velocities)

        sols = solve_eom_stack(eom_base, const, eom_time_r, const_r,
                               eom_time_i, const_i, time_functions,
                               t_eval, init_cond, solver, **kwargs)

        if weight_doppler:
            sols_weighted = apply_doppler_weights(sols, dop_velocities, dop_volumes)
            if sum_doppler:
                sum_axes = tuple(np.arange(spatial_dim))
                sols_final = np.sum(sols_weighted, axis=sum_axes)
            else:
                sols_final = sols_weighted
        else:
            sols_final = sols

    else:
        sols_final = solve_eom_stack(eom_base, const, eom_time_r, const_r,
                                     eom_time_i, const_i, time_functions, t_eval,
                                     init_cond, solver, **kwargs)

    return sols_final.swapaxes(-1,-2)


def solve_eom_stack(eoms_base: np.ndarray, const: np.ndarray,
                    eom_time_r: np.ndarray, const_r: np.ndarray,
                    eom_time_i: np.ndarray, const_i: np.ndarray,
                    time_inputs: List[TimeFunc],
                    t_eval: np.ndarray, init_cond: np.ndarray, solver, **kwargs
                    ) -> np.ndarray:

    """
    Solve a stack of equations of motion with shape `(*l, n, n)` in the time domain.
    
    Companion function to :func:`~.timesolvers.solve_time`, but can be invoked on its
    for equations already formatted. 

    Parameters
    ----------
    eoms_base : numpy.ndarray
        Array of shape `(*l, n, n)` represnting the part of
        equations of motion of the system which do not respond to external fields.
    const : numpy.ndarray
        constant term of shape (n,) added in differential equations. Typically
        generated by :func:`~.sensor_utils.generate_eom`.
    eoms_time_r : list[numpy.ndarray]
        list of arrays of shape `(basis_size^2-1, basis_size^2-1)`
        representing the parts of the OBEs with a real-valued time-dependence. In the solver,
        this array will be multiplied by a time-dependent rabi frequency. Typically a matrix of
        mostly zeros, with non-zero terms corresponding to a particular time-dependent coupling
    const_r : numpy.ndarray
        Constant term of shape (n,) added in a real time-dependent portion of
        differential equations. Typically generated by :func:`~.sensor_utils.generate_eom_time`.
    eoms_time_i : numpy.ndarray
        list of arrays of shape `(basis_size^2-1, basis_size^2-1)`
        representing the parts of the OBEs with an imaginary-valued time-dependence. In the solver,
        this array will be multiplied by a time-dependent rabi frequency.
    const_i : numpy.ndarray
        constant term of shape (n,) added in an imaginary time-dependent portion of
        differential equations. Typically generated by :func:`~.sensor_utils.generate_eom_time`.
    t_eval : numpy.ndarray
        1-D array of times, in microseconds, at which to
        evaluate the solution. Does not affect evaluations in the solve.
    time_inputs : list[function float->float]
        List of functions which represent
        the rabi frequency of a field as a function of time. list length should
        be identical to the length of obes_time. In the solver, the *i* th time
        input will be evaluated at time *t* and multiplied by the *i* th entry of
        obes_time.
    time_range tuple(float):
        Pair of values represent the start and end time, in microseconds, of the simulation.
    init_cond : numpy.ndarray or `None`, optional 
        Density matrix representing the initial state of the system. 
        If specified, the shape should be either `(n)` in the case of a single
        initial condition for all parameter values, or should be of shape `(*l, n)`
        matching the output shape of a steady state solve if the initial condition
        may be different for different combinations of parameters. 
        If `None`, will solve the problem in the steady state with all time-dependent fields
        "off" and use the solution as the initial condition for the time behavior. Other
        possible manual options might include a matrix populated by zeros representing the
        entire population in the ground state. Defaults to `None`.

    Returns
    -------
    numpy.ndarray
        Flattened solution array corresponding to time points.
        
    """
    stack_shape = eoms_base.shape[:-2]
    solution_shape = stack_shape + (eoms_base.shape[-1], len(t_eval))

    if init_cond.shape == eoms_base.shape[-1:]:
        init_cond = np.broadcast_to(init_cond, solution_shape[:-1])

    elif init_cond.shape != solution_shape[:-1]:
        msg = f"""Inital condition shape {init_cond.shape} does not match expected
        soulution shape {solution_shape[:-1]}"""
        raise RydiquleError(msg)
    
    solutions = solver(eoms_base, const, eom_time_r, const_r, eom_time_i, const_i,
                       time_inputs, t_eval, init_cond, **kwargs)

    return solutions


def generate_eom_time(hamiltonians_time: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates the Optical Bloch Equations for just the rf terms. 
    Uses the convention of the hamiltonian_rf return of the get_time_hamiltonian function.
    The equations of motion returned are assumed to be used in conjunction with an electric field.

    Parameters
    ----------
    hamiltonians_time : numpy.ndarray
        A matrix of shape (basis_size, basis_size), where the off-diagonal
        terms (i,j) are the dipole matrix elements in e a_b of the transition
        coupling state i to state j.

    Returns
    -------
    numpy.ndarray: Part of the Optical Bloch Equations corresponding to time_dependent couplings.
        To produce equations to solve, these values must be multiplied by an electric
        field in V/m.
    numpy.ndarray: Constant term of the time-dependent portion of the equations
        of motion. Same units as the equations themselves.
        
    """
    eom_time = _hamiltonian_term(hamiltonians_time)
    eqns, const = remove_ground(eom_time)
    eom_time, const = make_real(eqns,const)

    return eom_time.real, const
