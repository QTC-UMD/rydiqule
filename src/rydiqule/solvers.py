"""
Steady-state solvers of the Optical Bloch Equations.
"""
import numpy as np
from importlib.metadata import version
from copy import deepcopy

from .sensor import Sensor
from .sensor_utils import *
from .sensor_utils import _squeeze_dims
from .doppler_utils import *
from .slicing.slicing import *
from .sensor_solution import Solution


from typing import Optional, Iterable, Union


def solve_steady_state(
        sensor: Sensor, doppler: bool = False, doppler_mesh_method: Optional[MeshMethod] = None,
        sum_doppler: bool = True, weight_doppler: bool = True,
        n_slices: Union[int, None] = None) -> Solution:
    """
    Finds the steady state solution for a system characterized by a sensor.

    If insuffucent system memory is available to solve the system in a single call,
    system is broken into "slices" of manageable memory footprint which are solved indivudually.
    This slicing behavior does not affect the result.
    Can be performed with or without doppler averging.

    Parameters
    ----------
    sensor : :class:`~.Sensor`
        The sensor for which the solution will be calculated.
    doppler : bool, optional
        Whether to calculate the solution for a doppler-broadened
        gas. If `True`, only uses dopper brodening defined by `kvec` parameters
        for couplings in the `sensoe`, so setting this `True` without `kvec` definitions
        will have no effect. Default is `False`.
    doppler_mesh_method (dict,optional):
        If not `None`, should be a dictionary of meshing parameters to be passed
        to :func:`~.doppler_classes`. See :func:`~.doppler_classes` for more
        information on supported methods and arguments. If `None, uses the
        default doppler meshing. Default is `None`.
    sum_doppler : bool
        Whether to average over doppler classes after the solve
        is complete. Setting to `False` will not perform the sum, allowing viewing
        of the weighted results of the solve for each doppler class. In this case,
        an axis will be prepended to the solution for each axis along which doppler
        broadening is computed. Ignored if `doppler=False`. Default is `True`.
    weight_doppler : bool
        Whether to apply weights to doppler solution to perform
        averaging. If `False`, will **not** apply weights or perform a doppler_average,
        regardless of the value of `sum_doppler`. Changing from default intended
        only for internal use. Ignored if `doppler=False` or `sum_doppler=False`.
        Default is `True`.
    n_slices : int or None, optional
        How many sets of equations to break the full equations into.
        The actual number of slices will be the largest between this value and the minumum
        number of slices to solve the system without a memory error. If `None`, uses the minimum
        number of slices to solve the system without a memory error. Detailed information about
        slicing behavior can be found in :func:`~.slicing.slicing.matrix_slice`. Default is `None`.

    Notes
    -----
    .. note::
        If decoherence values are not sufficiently populated in the sensor, the resulting
        equations may be singular, resulting in an error in `numpy.linalg`. This error is not
        caught for flexibility, but is likely the culprit for `numpy.linalg` errors encountered
        in steady-state solves.

    .. note::
        The solution produced by this function will be expressed using rydiqule's convention
        of converting a density matrix into the real basis and removing the ground state to
        improve numerical stability.

    .. note::
        If the sensor contains couplings with `time_dependence`, this solver will add those
        couplings at their :math:`t=0` value to the steady-state hamiltonian to solve.

    Returns
    -------
    :class:`~.Solution`
        An object contining the solution and related information.

    Examples
    --------
    A basic solve for a 3-level system would have a "density matrix" solution of size 8 (3^2-1)

    >>> s = rq.Sensor(3)
    >>> s.add_coupling((0,1), detuning = 1, rabi_frequency=1)
    >>> s.add_coupling((1,2), detuning = 2, rabi_frequency=2)
    >>> s.add_transit_broadening(0.1)
    >>> sol = rq.solve_steady_state(s)
    >>> print(type(sol))
    <class 'rydiqule.sensor_solution.Solution'>
    >>> print(type(sol.rho))
    <class 'numpy.ndarray'>
    >>> print(sol.rho.shape)
    (8,)

    Defining an array-like parameter will automatically calculate the density matrix solution
    for every value. Here we use 11 values, resulting in 11 density matrices. The `axis_labels`
    attribute of the solution can clarify which axes are which.

    >>> s = rq.Sensor(3)
    >>> det = np.linspace(-1,1,11)
    >>> s.add_coupling((0,1), detuning = det, rabi_frequency=1)
    >>> s.add_coupling((1,2), detuning = 2, rabi_frequency=2)
    >>> s.add_transit_broadening(0.1)
    >>> sol = rq.solve_steady_state(s)
    >>> print(sol.rho.shape)
    (11, 8)
    >>> print(sol.axis_labels)
    ['(0,1)_detuning', 'density_matrix']

    >>> s = rq.Sensor(3)
    >>> det = np.linspace(-1,1,11)
    >>> s.add_coupling((0,1), detuning = det, rabi_frequency=1)
    >>> s.add_coupling((1,2), detuning = det, rabi_frequency=2)
    >>> s.add_transit_broadening(0.1)
    >>> sol = rq.solve_steady_state(s)
    >>> print(sol.rho.shape)
    (11, 11, 8)
    >>> print(sol.axis_labels)
    ['(0,1)_detuning', '(1,2)_detuning', 'density_matrix']

    If the solve uses doppler broadening, but not averaging for doppler is specified,
    there will be a solution axis corresponding to doppler classes.
    
    >>> s = rq.Sensor(3, vP=1)
    >>> det = np.linspace(-1,1,11)
    >>> s.add_coupling((0,1), detuning = det, rabi_frequency=1)
    >>> s.add_coupling((1,2), detuning = 2, rabi_frequency=2, kvec=(4,0,0))
    >>> s.add_transit_broadening(0.1)
    >>> sol = rq.solve_steady_state(s, doppler=True, sum_doppler=False)
    >>> print(sol.rho.shape)
    (561, 11, 8)
    >>> print(sol.axis_labels)
    ['doppler_0', '(0,1)_detuning', 'density_matrix']

    """

    solution = Solution()

    # relevant sensor-related quantities
    stack_shape = sensor._stack_shape()
    basis_size = sensor.basis_size
    spatial_dim = sensor.spatial_dim()

    # initialize doppler-related quantities
    doppler_axis_shape: Tuple[int, ...] = ()
    dop_classes = None
    doppler_shifts = None
    doppler_axes: Iterable[slice] = ()

    # update doppler-related values
    if doppler:
        dop_classes = doppler_classes(method=doppler_mesh_method)
        doppler_shifts = sensor.get_doppler_shifts()
        doppler_axis_shape = tuple(len(dop_classes) for _ in range(spatial_dim))

        if not sum_doppler:
            doppler_axes = tuple(slice(None) for _ in range(spatial_dim))

    n_slices, out_sol_shape = get_slice_num(basis_size, stack_shape, doppler_axis_shape,
                                                sum_doppler, weight_doppler, n_slices)

    if n_slices > 1:
        print(f"Breaking equations of motion into {n_slices} sets of equations...")

    # get steady-state hamiltonians, assume time-dependent parts have t=0 value
    hamiltonians_total = sensor.get_time_hamiltonian(t=0)
    # get decoherence matrix
    gamma = sensor.decoherence_matrix()
    # allocate solution array
    sols = np.zeros(out_sol_shape)

    # loop over individual slices of hamiltonian
    n_slices_true = sum(1 for _ in matrix_slice(gamma, n_slices=n_slices))

    for i, (idx, H, G) in enumerate(matrix_slice(hamiltonians_total, gamma, n_slices=n_slices)):

        if n_slices_true > 1:
            print(f"Solving equation slice {i+1}/{n_slices_true}", end='\r')

        full_idx = (*doppler_axes, *idx)
        sols[full_idx] = _solve_hamiltonian_stack(
            H, G, doppler=doppler, dop_classes=dop_classes, sum_doppler=sum_doppler,
            weight_doppler=weight_doppler, doppler_shifts=doppler_shifts,
            spatial_dim=spatial_dim
            )

    # save results to Solution object
    solution.rho = sols

    # specific to observable calculations
    solution._eta = sensor.eta
    solution._kappa = sensor.kappa
    solution._cell_length = sensor.cell_length
    solution._beam_area = sensor.beam_area
    solution._probe_freq = sensor.probe_freq
    solution._probe_tuple = sensor.probe_tuple

    #store the graph with fully_expanded dimensionality
    sensor._expand_dims()
    solution.couplings = deepcopy(sensor.couplings)
    _squeeze_dims(sensor.couplings)

    solution.axis_labels = ([f'doppler_{i:d}' for i in range(spatial_dim) if not sum_doppler]
                            + sensor.axis_labels()
                            + ["density_matrix"])
    solution.axis_values = ([dop_classes for i in range(spatial_dim) if not sum_doppler]
                            + [val for _,_,val,_ in sensor.variable_parameters()]
                            + [sensor.dm_basis()])
    solution.dm_basis = sensor.dm_basis()
    solution.rq_version = version("rydiqule")
    solution.doppler_classes = dop_classes

    return solution


def _solve_hamiltonian_stack(
        hamiltonians: np.ndarray, gamma_matrix: np.ndarray,
        doppler: bool = False, dop_classes: Optional[np.ndarray] = None,
        sum_doppler: bool = True, weight_doppler: bool = True,
        doppler_shifts: Optional[np.ndarray] = None, spatial_dim: int = 0
        ) -> np.ndarray:
    """
    Solves a the equations of motion corresponding to the given set of hamiltonians.

    Typically used as an auxillary function for :meth:`~.solve_steady_state`. Hamiltonian and
    gamma matrices must be of broadcastable shapes.
    """
    eom, const = generate_eom(hamiltonians, gamma_matrix)

    if doppler:
        assert dop_classes is not None and doppler_shifts is not None
        dop_velocities, dop_volumes = doppler_mesh(dop_classes, spatial_dim)

        eom = get_doppler_equations(eom, doppler_shifts, dop_velocities)

        # this is required for linalg.solve boadcasting to work
        const = np.expand_dims(const, tuple(np.arange(spatial_dim)))
        sols_full = steady_state_solve_stack(eom, const)

        if weight_doppler:
            sols_weighted = apply_doppler_weights(sols_full, dop_velocities, dop_volumes)
            if sum_doppler:
                sum_axes = tuple(np.arange(spatial_dim))
                sols = np.sum(sols_weighted, axis=sum_axes)
            else:
                sols = sols_weighted
        else:
            sols = sols_full

    else:
        sols = steady_state_solve_stack(eom, const)

    return sols


def steady_state_solve_stack(eom: np.ndarray, const: np.ndarray) -> np.ndarray:
    """
    Helper function which returns the solution to the given equations of motion

    Solves an equation of the form :math:`\\dot{x} = Ax + b`, or a set of such equations
    arranged into stacks.
    Essentially just wraps numpy.linalg.solve(), but included as its own
    function for modularity if another solver is found to be worth invesitigating.

    Parameters
    ----------
    eom : numpy.ndarray
        An square array of shape `(*l,n,n)` representing the differential
        equations to be solved. The matrix (or matrices) A in the above formula.
    const : numpy.ndarray
        An array or shape `(*l,n)` representing the constant in the matrix form
        of the differential equation. The constant b in the above formula. Stack shape
        `*l` must be consistent with that in the `eom` argument

    Returns
    -------
    numpy.ndarray
        A 1xn array representing the steady-state solution
        of the differential equation
    """

    # const broadcasting hack to retain np.linalg.solve behavior from np v1 with np v2
    # https://numpy.org/doc/stable/release/2.0.0-notes.html#removed-ambiguity-when-broadcasting-in-np-solve
    sol = np.linalg.solve(eom, -const[..., None])[..., 0]
    return sol
