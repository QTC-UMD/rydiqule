"""
Steady-state solver for analytical Doppler averaging

Code in this module implements analytical doppler averaging techniques described in
Omar Nagib and Thad G. Walker, *Exact steady state of perturbed open quantum systems*,
arXiv 2501.06134 (2025) http://arxiv.org/abs/2501.06134

This solver computes the doppler averaged steady state of a sensor.
In 1 spatial dimension problems, it averages analytically.
In 2 or 3 spatial dimension problems, it averages one axis analytically
and the remaining numerically.
"""

import warnings
import numpy as np
from importlib.metadata import version
from copy import deepcopy
from typing import Optional, Set

from scipy.special import erfcx

from .sensor import Sensor
from .sensor_utils import _hamiltonian_term, generate_eom, make_real, _squeeze_dims
from .sensor_solution import Solution
from .doppler_utils import doppler_classes, doppler_mesh, apply_doppler_weights, MeshMethod, get_doppler_equations
from .slicing.slicing import matrix_slice, get_slice_num_hybrid
from .exceptions import RydiquleError, RydiquleWarning, PopulationNotConservedWarning


def _doppler_eigvec_array(lamdas: np.ndarray, rtol: float = 1e-5, atol: float = 1e-9) -> np.ndarray:
    """
    Helper function for vectorizing the analytical summation of eigenvalues.

    Note that because rydiqule provides L_v as containing sqrt(2)*sigma_v,
    that term is rolled into the definition of lamdas here.

    Parameters
    ----------
    lamdas: numpy.ndarray
        Array of complex eigenvalues
    rtol: float, optional
        Relative tolerance parameter for checking 0-eigenvalues when calculating the doppler prefactor.
        Passed to :external+numpy:func:`~numpy.isclose`.
        Defaults to 1e-5.
    atol: float, optional
        Absolute tolerance parameter for checking 0-eigenvalues when calculating the doppler prefactor.
        Passed to :external+numpy:func:`~numpy.isclose`.
        Defaults to 1e-9.

    Returns
    -------
    numpy.ndarray
        Analytical Doppler broadening function evaluated at the given eigenvalues.
        Values are complex.
    """

    doppler_array = np.ones_like(lamdas) # for 0-eigenvalues, set value to 1
    idx = np.where(~np.isclose(lamdas, 0, rtol, atol))

    if len(idx[0]) > 0:
        # split the calculation into two lines for readability, using the complementary error function to avoid 
        # catastrophic cancellation when |lambda|~0
        p1 = np.sqrt(np.pi)/(np.sqrt(-1*lamdas[idx]**2))
        p2 = erfcx(-1*np.sqrt(-1*lamdas[idx]**2)/(lamdas[idx]**2))
        doppler_array[idx] = p1*p2

    return doppler_array

def _get_rho0(L0: np.ndarray) -> np.ndarray:
    """
    Helper function for computing rho0 (null vector) of a stack of equations of motion
    via the inverse power method.

    Parameters
    ----------
    L0: np.ndarray
        Stack of steady state Liouvillians
    
    Returns
    -------
    numpy.ndarray
        Stack of steady state (vectorized) density matrices
    """

    stack_shape = L0.shape[0:-2]
    n = int(np.sqrt(L0.shape[-1]))
    rho0 = np.random.rand(*stack_shape, n**2)[..., np.newaxis]
    rho0 /= np.linalg.norm(rho0, axis=-1, keepdims=True)

    I = np.eye(n**2)
    converged_flags = np.zeros(stack_shape, dtype=bool)
        
    # Compute rho0 by finding the null vector of L0 via the shifted inverse power method
    for iteration in range(50):
        if np.all(converged_flags):
            break
        
        remaining_flags_index = np.where(~converged_flags)
        
        current_L0 = L0[remaining_flags_index]
        current_shifted_L0 = current_L0 + 1e-14 * I

        current_rho0 = rho0[remaining_flags_index]
            
        z = np.linalg.solve(current_shifted_L0, current_rho0) # compute (L0 + 1e-14)^-1 * rho0
        rho0_new = z / (np.linalg.norm(z, axis=-2, keepdims=True) + np.finfo(float).eps)
        
        # Estimate magnitude of eigenvalues by Rayleigh quotient
        L0rho0 = np.einsum('...ij,...j->...i', current_L0, rho0_new[..., 0])
        numerator = np.sum(rho0_new[..., 0] * L0rho0, axis=-1)
        denominator = np.sum(rho0_new[..., 0] * rho0_new[..., 0], axis=-1)
        current_eigenvalues = np.abs(numerator / denominator)

        rho_0_diff = np.linalg.norm(current_rho0 - rho0_new, axis=-1)

        converged_flags[remaining_flags_index] = (rho_0_diff < 1e-15).any(axis=-1) | (current_eigenvalues < 1e-14)
                
        rho0[remaining_flags_index] = rho0_new

    rho0 = rho0.squeeze(axis=-1)

    rho0 *= np.sign(rho0[...,0])[...,None]  # remove arbitrary sign from null-vector so all pops are positive
    pops = np.sum(rho0[...,::n+1], axis=-1)  # calculate trace of each vector
    rho0 /= pops[..., None]  # normalize vectors by trace

    return rho0


def solve_doppler_analytic(sensor: Sensor, doppler_mesh_method: Optional[MeshMethod] = None, 
                                analytic_axis: Optional[int] = None, n_slices: Optional[int] = None, rtol: float = 1e-5, 
                                atol: float = 1e-9) -> Solution:
    """
    Solves a sensor in steady state in the presence of doppler broadening,
    with one dimension analytically averaged. 
    
    If the broadening is 1 dimensional, this function will solve analytically. 
    If the broadening is 2 or 3 dimensional, 
    this function will average analytically over the specified axis 
    and numerically over the remaining axes.

    This function uses the method outlined in Ref [1]_ for the analytic dimension. 

    This solver is considered more accurate than :func:`~.solve_steady_state`
    since it replaces direct sampling and solving of the velocity classes
    with a few tensor inversions and calculation of the numerical prefactor.
    This also leads to faster solves,
    approximately dictated by the ratio of samples along the doppler axis
    relative to the other parameter dimensions. 
    Additionally, in sensors with 2 or 3 dimensional doppler broadening, 
    this solver effectively reduces the dimension to 1 or 2, respectively, 
    leading to faster solves.

    If insuffucent system memory is available to solve the system in a single call,
    system is broken into "slices" of manageable memory footprint which are solved indivudually.
    This slicing behavior does not affect the result.
     
    Parameters
    ----------
    sensor : :class:`~.Sensor`
        The sensor for which the solution will be calculated.
        It must define 1 and only 1 dimension of doppler shifts
        (ie one or more couplings with `kvec` with non-zero values on the same dimension).
    doppler_mesh_method: dict, optional
        If not `None`, should be a dictionary of meshing parameters to be passed
        to :func:`~.doppler_classes`. See :func:`~.doppler_classes` for more
        information on supported methods and arguments. If `None, uses the
        default doppler meshing. Default is `None`.
    analytic_axis: int, optional
        Specifies over which axis the solver will average analytically.
        Defaults to the first nonzero axis.
    n_slices : int or None, optional
        How many sets of equations to break the full equations into.
        The actual number of slices will be the largest between this value and the minumum
        number of slices to solve the system without a memory error. If `None`, uses the minimum
        number of slices to solve the system without a memory error. Detailed information about
        slicing behavior can be found in :func:`~.slicing.slicing.matrix_slice`. Default is `None`.
    rtol : float, optional
        Relative tolerance parameter for checking 0-eigenvalues when calculating the doppler prefactor.
        Passed to :external+numpy:func:`~numpy.isclose`.
        Defaults to 1e-5.
    atol : float, optional
        Absolute tolerance parameter for checking 0-eigenvalues when calculating the doppler prefactor.
        Passed to :external+numpy:func:`~numpy.isclose`.
        Defaults to 1e-9.

    Returns
    -------
    :class:`~.Solution`
        An object containing the solution and related information.
    
    References
    ----------
    .. [1] Omar Nagib and Thad G. Walker,
        Exact steady state of perturbed open quantum systems,
        arXiv 2501.06134 (2025)
        http://arxiv.org/abs/2501.06134v3
    """

    spatial_dim = sensor.spatial_dim()

    if spatial_dim == 0:
        raise RydiquleError("System must have at least 1 spatial dimension of Doppler shifts")

    # Create a sorted array of nonzero axes
    kvecs = sensor.get_value_dictionary('kvec')
    nonzero: Set[int] = set()
    for _, v in kvecs.items():
        current_nonzero_axes = np.where(~np.isclose(v, 0.0))[0]
        nonzero.update(current_nonzero_axes)
    nonzero_axes = np.array(sorted(list(nonzero)))

    # Set the default analytic axis to the first nonzero axis
    if analytic_axis is None:
        analytic_axis: int = nonzero_axes[0]
    
    # Account for analytic axis set to a zero axis, e.g., kvec of (1,1,0) and inputted analytic axis of 2
    elif analytic_axis not in nonzero_axes:
        raise RydiquleError(f"analytic_axis={analytic_axis} has no doppler shifts, "
                            + f" valid options are: {nonzero_axes}")

    # Remap the physical axes to the reduced problem space, e.g., kvec of (0,0,1) and inputted analytic axis of 2
    analytic_axis = np.where(nonzero_axes == analytic_axis)[0][0]

    n = sensor.basis_size

    # 1. Get base Liouvillian (L0) and Doppler shifts
    L0, dummy_const = generate_eom(sensor.get_hamiltonian(), sensor.decoherence_matrix(),
                                   remove_ground_state=False, real_eom=True)
    all_shifts = sensor.get_doppler_shifts()

    # 2. Compute perturbation and velocity mesh
    numeric_axes = [ax for ax in range(spatial_dim) if ax != analytic_axis]
    num_numeric_dims = len(numeric_axes)
    dopp_classes = doppler_classes(method = doppler_mesh_method)
    dopp_velocities, dopp_volumes = doppler_mesh(dopp_classes, num_numeric_dims)

    analytic_shift = np.take(all_shifts, analytic_axis, axis=0)
    numeric_shifts = np.delete(all_shifts, analytic_axis, axis=0)
    L_pert_complex = _hamiltonian_term(analytic_shift.squeeze())
    L_pert, _ = make_real(L_pert_complex, dummy_const, ground_removed=False)

    n_vel_points = len(dopp_classes)
    numeric_doppler_shape = (n_vel_points,) * num_numeric_dims
    param_stack_shape = L0.shape[:-2]

    # 3. Compute number of slices and loop over each slice computing rho0, L0m, the analytic integral, 
    #    and the numeric weighting/summing
    n_slices, out_sol_shape = get_slice_num_hybrid(n, param_stack_shape, numeric_doppler_shape, 
                                                   n_slices=n_slices)

    if n_slices > 1:
        print(f"Breaking parameter stack into {n_slices} slices...")

    sols = np.zeros(out_sol_shape, dtype = np.complex128)

    for i, (idx, L0_slice) in enumerate(matrix_slice(L0, n_slices=n_slices)):
        if n_slices > 1:
            print(f"Solving slice {i+1}/{n_slices}", end='\r')
        
        if num_numeric_dims == 0:
            L_base_slice = L0_slice
        else:
            L_base_slice = get_doppler_equations(L0_slice, numeric_shifts, dopp_velocities, ground_removed=False)
        rho0_slice = _get_rho0(L_base_slice)

        vec1 = np.eye(n).flatten()
        L0m_slice = (np.linalg.inv(L_base_slice + rho0_slice[..., np.newaxis] * vec1[np.newaxis, :])
                        - rho0_slice[..., np.newaxis] * vec1[np.newaxis, :])
        lamdas, r_eigvecs = np.linalg.eig(L0m_slice @ L_pert)

        prefix = _doppler_eigvec_array(lamdas)
        suffix = np.linalg.solve(r_eigvecs, rho0_slice[..., np.newaxis]).squeeze(axis=-1)
        rho_dopp_slice = np.einsum('...j,...ij,...j->...i', prefix, r_eigvecs, suffix)

        sols_weighted = apply_doppler_weights(rho_dopp_slice, dopp_velocities, dopp_volumes)
        axes_to_sum = tuple(range(num_numeric_dims))
        sols_slice = np.sum(sols_weighted, axis=axes_to_sum)

        sols[idx] = sols_slice

    # 4. Postprocess solution
    imag_tol = 10000
    sols_real = np.real_if_close(sols, tol=imag_tol)
    if np.iscomplexobj(sols_real):
        rho_dopp_imag = np.abs(sols.imag)
        count = np.count_nonzero(rho_dopp_imag > np.finfo(float).eps*imag_tol)
        warnings.warn('Doppler-averaged solution has complex parts outside of tolerance, solution is suspect. ' +
                      f'{count:d} of {sols.size:d} elments larger than cutoff {np.finfo(float).eps*imag_tol:.3g}. ' +
                      f'Max Abs(Imag): {sols.max():.3g}, Std Abs(Imag): {np.std(sols):.3g}',
                      RydiquleWarning)
    pops_dopp = np.sum(sols_real[...,::n+1], axis=-1)
    if not np.isclose(pops_dopp, 1.0, rtol=.01).all():
        warnings.warn('Doppler-averaged solution has populations not conserved, solution is suspect. ' +
                    f'{np.count_nonzero(~np.isclose(pops_dopp, 1.0)):d} of {pops_dopp.size:d} have non-unit trace. ' +
                    f'Min trace is {pops_dopp.min():f}',
                    PopulationNotConservedWarning)

    # 5. Package into a solution object
    solution = Solution()
    solution.rho = sols_real[...,1:]
    
    solution._eta = sensor.eta
    solution._kappa = sensor.kappa
    solution._cell_length = sensor.cell_length
    solution._beam_area = sensor.beam_area
    solution._probe_freq = sensor.probe_freq
    solution._probe_tuple = sensor.probe_tuple

    sensor._expand_dims()
    solution.couplings = deepcopy(sensor.couplings)
    _squeeze_dims(sensor.couplings)

    solution.axis_labels = (sensor.axis_labels() + ["density_matrix"])
    solution.axis_values = ([val for _,_,val,_ in sensor.variable_parameters()]
                            + [sensor.dm_basis()])
    solution.dm_basis = sensor.dm_basis()
    solution.rq_version = version("rydiqule")

    return solution
