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


def doppler_1d_exact(sensor: Sensor, rtol: float = 1e-5, atol: float = 1e-9) -> Solution:
    """
    Analytically solves a sensor in steady-state in the presence of 1 dimensional
    Doppler broadening.

    Uses the method outlined in Ref [1]_.
    In particular, it uses Eq. 12 to analytically evaluate the Doppler average in 1D.

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
        http://arxiv.org/abs/2501.06134
    """

    if sensor.spatial_dim() != 1:
        raise RydiquleError(f'Sensor must have 1 spatial dimension of Doppler shifts, found {sensor.spatial_dim():d}')

    stack_shape = sensor._stack_shape()
    n = sensor.basis_size
    # Liouvillian superoperator for the non-doppler-broadened components
    L0, dummy_const = generate_eom(sensor.get_hamiltonian(), sensor.decoherence_matrix(),
                      remove_ground_state=False,
                      real_eom=True)

    ### construct L0^minus
    el0, ev0 = np.linalg.eig(L0)
    #pseudo invert the eigenvalues
    l_mask = np.isclose(el0, 0.0)
    el0[l_mask] = 1 # prevent divide by zero
    li = 1/el0
    li[l_mask] = 0 # set masked elements to zero
    # calculate Eq B2
    L0m = (ev0*li[...,None, :])@np.linalg.pinv(ev0)
    
    ### calculate rho0 (doppler-free solution) from nullspace eigenvectors in L0m calculation
    assert np.all(np.count_nonzero(l_mask, axis=-1) == 1), 'rho0 solution not unique'
    # select right eigenvector corresponding to 0-eigenvalue (marked by l_mask) for each equation set
    # using reshape restores stack axes after binary array indexing flatten
    rho0 = np.real_if_close(np.swapaxes(ev0, axis1=-1, axis2=-2)[l_mask].reshape(*stack_shape, -1))
    assert not np.iscomplexobj(rho0), 'rho0 solution is not real; it is unphysical'
    rho0 *= np.sign(rho0[...,0])[...,None]  # remove arbitrary sign from null-vector so all pops are positive
    pops = np.sum(rho0[...,::n+1], axis=-1)  # calculate trace of each vector
    rho0 /= pops[..., None]  # normalize vectors by trace

    ### Liouvillian superoperator for doppler only
    # these are already multiplied by sqrt(2)*sigma_v by rydiqule
    # as such, lambdas are redefined as sqrt(2)*sigma_v*lambdas of Eq12 in the paper
    dopp = sensor.get_doppler_shifts().squeeze()
    Lv_complex = _hamiltonian_term(dopp)
    Lv, _ = make_real(Lv_complex, dummy_const, ground_removed=False)

    ### Calculate doppler averaged steady-state density matrix from propagator
    el, R = np.linalg.eig(L0m@Lv)
    L = np.linalg.pinv(np.swapaxes(R,-1,-2))

    # calculate Eq 12
    prefix = _doppler_eigvec_array(el, rtol, atol)
    rho_dopp_complex = np.einsum('...j,...ij,...kj,...k->...i', prefix, R, L, rho0)
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
