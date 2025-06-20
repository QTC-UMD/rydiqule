"""
Steady-state solver for analytical Doppler averaging

Code in this module implements analytical doppler averaging techniques described in
Omar Nagib and Thad G. Walker, *Exact steady state of perturbed open quantum systems*,
arXiv 2501.06134 (2025) http://arxiv.org/abs/2501.06134

Solvers in this module are considered experimental.
While we encourage raising issues encountered,
issues with their use are considered features not fully implemented rather than bugs.

At this point, only 1D doppler-averaged steady-state solutions are supported via
:func:`doppler_1d_exact`.
"""

import warnings
import numpy as np
from importlib.metadata import version
from copy import deepcopy

from scipy.special import erf

from .sensor import Sensor
from .sensor_utils import _hamiltonian_term, generate_eom, make_real, _squeeze_dims
from .sensor_solution import Solution
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
        # split the calculation into three lines for readability
        p1 = np.sqrt(np.pi)/(np.sqrt(-1*lamdas[idx]**2))
        p2 = np.exp(-1/(lamdas[idx]**2))
        p3 = 1+erf(np.sqrt(-1*lamdas[idx]**2)/(lamdas[idx]**2))
        doppler_array[idx] = p1*p2*p3

    return doppler_array

def _get_rho0(L0: np.ndarray) -> np.ndarray:
    """
    Helper function for computing rho0 (null vector) of a stack of equations of motion
    via the inverse power method.

    Paramaters
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

def doppler_1d_exact(sensor: Sensor, rtol: float = 1e-5, atol: float = 1e-9) -> Solution:
    """
    Analytically solves a sensor in steady-state in the presence of 1 dimensional
    Doppler broadening.

    Uses the method outlined in Ref [1]_.
    In particular, it uses Eq. 14 to analytically evaluate the Doppler average in 1D.

    This solver is considered more accurate than :func:`~.solve_steady_state`
    since it replaces direct sampling and solving of the velocity classes
    with a few tensor inversions and calculation of the numerical prefactor.
    This also leads to faster solves,
    approximately dictated by the ratio of samples along the doppler axis
    relative to the other parameter dimensions.

    Parameters
    ----------
    sensor : :class:`~.Sensor`
        The sensor for which the solution will be calculated.
        It must define 1 and only 1 dimension of doppler shifts
        (ie one or more couplings with `kvec` with non-zero values on the same dimension).
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
    :class:`~.Solution`
        An object containing the solution and related information.

    Raises
    ------
    RydiquleError
        If the `sensor` does not have exactly 1 dimension of doppler shifts to average over.
    AssertionError
        If the initial rho0 calculation results in an unphysical result.

    Warns
    -----
    RydiquleWarning
        If the averaged result is not real within tolerances.
        While rydiqule's computational basis is real,
        the method employed here involves complex number floating point calculations.
        If all is well, the complex parts should cancel to return a real result,
        but imprecision in floating point operations can occur.
    PopulationNotConservedWarning
        Before removing the ground state in the final solution,
        population conservation is confirmed.
        If the resulting density matrices do not preserve trace, this warning is raised
        indicating an issue in the calculation.

    References
    ----------
    .. [1] Omar Nagib and Thad G. Walker,
        Exact steady state of perturbed open quantum systems,
        arXiv 2501.06134 (2025)
        http://arxiv.org/abs/2501.06134v3
    """

    if sensor.spatial_dim() != 1:
        raise RydiquleError(f'Sensor must have 1 spatial dimension of Doppler shifts, found {sensor.spatial_dim():d}')

    n = sensor.basis_size
    # Liouvillian superoperator for the non-doppler-broadened components
    L0, dummy_const = generate_eom(sensor.get_hamiltonian(), sensor.decoherence_matrix(),
                      remove_ground_state=False,
                      real_eom=True)

    rho0 = _get_rho0(L0)
    
    vec1 = np.eye(n).flatten() #Initialize vectorized identity
    L0m = (np.linalg.inv(L0 + rho0[..., np.newaxis] * vec1[np.newaxis, :])
           - rho0[..., np.newaxis] * vec1[np.newaxis, :]
    )
    ### Liouvillian superoperator for doppler only
    # these are already multiplied by sqrt(2)*sigma_v by rydiqule
    # as such, lambdas are redefined as sqrt(2)*sigma_v*lambdas of Eq14 in the paper
    dopp = sensor.get_doppler_shifts().squeeze()
    Lv_complex = _hamiltonian_term(dopp)
    Lv, _ = make_real(Lv_complex, dummy_const, ground_removed=False)

    ### Calculate doppler averaged steady-state density matrix from propagator
    lamdas, r_eigvecs = np.linalg.eig(L0m@Lv)

    # calculate Eq 14
    prefix = _doppler_eigvec_array(lamdas)
    suffix = np.linalg.solve(r_eigvecs, rho0[..., np.newaxis]).squeeze(axis=-1)
    rho_dopp_complex = np.einsum('...j,...ij,...j->...i', prefix, r_eigvecs, suffix)

    # confirm that result is approximately real
    imag_tol = 10000
    rho_dopp = np.real_if_close(rho_dopp_complex, tol=imag_tol)  # chop complex parts if all are smaller than 10000*f64_eps
    if np.iscomplexobj(rho_dopp):
        rho_dopp_imag = np.abs(rho_dopp_complex.imag)
        count = np.count_nonzero(rho_dopp_imag > np.finfo(float).eps*imag_tol)
        warnings.warn('Doppler-averaged solution has complex parts outside of tolerance, solution is suspect. ' +
                      f'{count:d} of {rho_dopp.size:d} elments larger than cutoff {np.finfo(float).eps*imag_tol:.3g}. ' +
                      f'Max Abs(Imag): {rho_dopp_imag.max():.3g}, Std Abs(Imag): {np.std(rho_dopp_imag):.3g}',
                      RydiquleWarning)
    # ensure trace is preserved before dropping ground state, which will implicitely enforce it
    pops_dopp = np.sum(rho_dopp[...,::n+1], axis=-1)
    if not np.isclose(pops_dopp, 1.0).all():
        warnings.warn('Doppler-averaged solution has populations not conserved, solution is suspect. ' +
                      f'{np.count_nonzero(~np.isclose(pops_dopp, 1.0)):d} of {pops_dopp.size:d} have non-unit trace. ' +
                      f'Min trace is {pops_dopp.min():f}',
                      PopulationNotConservedWarning)

    ### make into a Solution object
    solution = Solution()
    # match rydiqule convention for return (ie no ground population)
    solution.rho = rho_dopp[...,1:]
    
    # specific to observable calculations
    solution._eta = sensor.eta
    solution._kappa = sensor.kappa
    solution._cell_length = sensor.cell_length
    solution._beam_area = sensor.beam_area
    solution._probe_freq = sensor.probe_freq
    solution._probe_tuple = sensor.probe_tuple

    # store the graph with fully_expanded dimensionality
    sensor._expand_dims()
    solution.couplings = deepcopy(sensor.couplings)
    _squeeze_dims(sensor.couplings)

    solution.axis_labels = (sensor.axis_labels() + ["density_matrix"])
    solution.axis_values = ([val for _,_,val,_ in sensor.variable_parameters()]
                            + [sensor.dm_basis()])
    solution.dm_basis = sensor.dm_basis()
    solution.rq_version = version("rydiqule")

    return solution