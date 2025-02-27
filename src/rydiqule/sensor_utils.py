"""
Utilities used by the Sensor classes.
"""
import math
from itertools import product

import numpy as np
import networkx as nx
from .sensor_solution import Solution
from .exceptions import RydiquleError
import scipy.constants
from scipy.constants import hbar, e
from leveldiagram import LD

from typing import Dict, Tuple, Union, List, Callable, TYPE_CHECKING
if TYPE_CHECKING:
    # only import when type checking, avoid circular import
    from .sensor import Sensor

a0 = scipy.constants.physical_constants["Bohr radius"][0]

# put composite types of Sensor/Cell/Solution here
ScannableParameter = Union[float, List[float], np.ndarray]

CouplingDict = Dict

State = Union[int, str, Tuple[float, ...]]
States = Tuple[State, State]

Spec = Tuple[Union[float, List[float]], ...]
StateSpec = Union[State, List[State], Spec]
StateSpecs = Tuple[StateSpec, StateSpec]

TimeFunc = Callable[[float], complex]

RHO: Dict[int, np.ndarray] = {}
U: Dict[int, np.ndarray] = {}
B: Dict[int, int] = {}


def generate_eom(hamiltonian: np.ndarray, gamma_matrix: np.ndarray,
                 remove_ground_state: bool = True,
                 real_eom: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create the optical bloch equations for a hamiltonian and decoherence matrix
    using the Lindblad master equation.

    Parameters
    ----------
    hamiltonian : numpy.ndarray
        Complex array representing the Hamiltonian matrix
        of the system, the matrix should be of shape `(*l, n, n)`, where n is
        the basis size and l is the shape of the stack of hamiltonians. For
        example, if the hamiltonian varies in 2 parameters l might be `(10, 10)`.
    gamma_matrix : numpy.ndarray
        Complex array representing the decoherence matrix
        of the system, the matrix should be of size `(n, n)`, where n is the basis size.
    remove_ground_state : bool, optional
        Remove the ground state from the equations
        of motion using population conservation. Setting to `False` is intended
        for internal use only and is not officially supported.
        See :func:`~.remove_ground` for details.
    real_eom : bool, optional
        Transform the equations of motion from the complex basis
        to the real basis. Setting to `False` is intended for internal use only
        and is not officially supported.
        Seee :func:`~.make_real` for details.

    Returns
    -------
    equations : numpy.ndarray
        The array representing the Optical
        Bloch Equations (OBEs) of the system. The shape will be `(*l, n^2-1, n^2-1)`
        if `remove_ground_state` is `True` and `(*l, n^2, n^2)` otherwise. The
        datatype will be `np.float64` if `real_eom` is `True` and `np.complex128`
        otherwise.
    const : numpy.ndarray
        Array of which defines the constant
        term in the linear OBEs. The shape will be `(*l, n^2-1)` if
        `remove_ground_state` is `True` and `(*l, n^2)` otherwise. The
        datatype will be `np.float64` if `real_eom` is `True` and `np.complex128`
        otherwise.

    Raises
    ------
    RydiquleError: If the shapes of gamma_matrix and hamiltonian are not matching
        or not square in the last 2 dimensions

    Examples
    --------
    >>> ham = np.diag([1,-1])
    >>> gamma = np.array([[.1, 0],[.1,0]])
    >>> print(ham.shape)
    (2, 2)
    >>> eom, const = rq.sensor_utils.generate_eom(ham, gamma)
    >>> print(eom)
    [[-0.1  2.   0. ]
     [-2.  -0.1  0. ]
     [ 0.   0.  -0.1]]
    >>> print(const.shape)
    (3,)

    This also works with a "stack" of multiple hamiltonians:

    >>> ham_base = np.diag([1,-1])
    >>> ham_full = np.array([ham_base for _ in range(10)])
    >>> gamma = np.array([[.1, 0],[.1,0]])
    >>> print(ham_full.shape)
    (10, 2, 2)
    >>> eom, const = rq.sensor_utils.generate_eom(ham_full, gamma)
    >>> print(eom.shape)
    (10, 3, 3)
    >>> print(const.shape)
    (10, 3)

    """
    if not hamiltonian.shape[-2:] == gamma_matrix.shape[-2:]:
        raise RydiquleError("hamiltonian and gamma matrix must have matching shape")
    if not hamiltonian.shape[-1] == hamiltonian.shape[-2]:
        raise RydiquleError("hamiltonian and gamma matrix must be square")

    basis_size = hamiltonian.shape[-1]
    basis = np.array([[[m,n] for m in range(basis_size)] for n in range(basis_size)])
    basis = basis.reshape((basis_size)**2,2)

    # create optical bloch equations
    obes = _hamiltonian_term(hamiltonian) + _decoherence_term(gamma_matrix)
    stack_shape = obes.shape[:-2]
    const_shape = (*tuple(np.ones_like(stack_shape)), basis_size**2)
    const = np.zeros(const_shape)

    if remove_ground_state:
        obes, const = remove_ground(obes)

    # transform to real basis
    if real_eom:
        obes, const = make_real(obes, const, ground_removed=remove_ground_state)

    return obes, const


def _hamiltonian_term(ham: np.ndarray) -> np.ndarray:
    """
    Helper function to calculate the first term of the Lindblad master equation.

    Parameters
    ----------
    ham : numpy.ndarray
        Complex Hamiltonian of shape `(*l, n, n)`
        the describes the couplings between states and detunings.

    Returns
    -------
    numpy.ndarray
        Coherent Hamiltonian portion of the EOMs of shape `(*l, n^2, n^2)`
        of dtype np.complex128

    """
    n: int = ham.shape[-1]
    stack_shape = ham.shape[:-2]
    ham = ham.reshape(stack_shape+(1,1,n,n))
    rho = _get_rho(n)

    term: np.ndarray = -1j*((ham @ rho) - (rho @ ham))

    return term.reshape(stack_shape+(n*n, n*n))


def _decoherence_term(gamma: np.ndarray) -> np.ndarray:
    """
    Helper function to determine the second term of the Lindblad master equation.

    Parameters
    ----------
    gamma : numpy.ndarray
        Square decoherence matrix of shape `(*l, n, n)`

    Returns
    -------
    numpy.ndarray
        The decoherence portion of the EOMs of shape `(*l, n^2, n^2)`.

    """
    n = gamma.shape[-1]
    rho = _get_rho(n)
    stack_shape = gamma.shape[:-2]
    gamma_exp = gamma.reshape(stack_shape+(1,1,n,n))
    g = gamma.sum(axis=-1).reshape(stack_shape+(1,1,1,n)) * rho

    # equivalent to g_T = np.einsum('...ijkl->...lkji', g)
    g_axes = np.arange(len(g.shape))
    gT_axes = np.concatenate([g_axes[:-4], g_axes[-4:][::-1]])
    g_T = np.transpose(g, axes=gT_axes)

    term: np.ndarray = np.swapaxes((rho @ gamma_exp @ rho), -2, -3) - (g_T + g)/2
    return term.reshape(stack_shape+(n*n, n*n))


def remove_ground(equations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove the ground state from the equations of motion using population conservation.

    Population conservation enforces

    .. math:: \\rho_{(0,0)} = 1 - \\sum_{i=1}^{n-1} \\rho_{(i,i)}

    We use this equation to remove the EOM for rho_00 and
    enforce population conservation in the steady state.

    Parameters
    ----------
    equations : numpy.ndarray
        array of shape (n^2, n^2) representing the equations
        of motion of the system, where n is the number of basis states.

    Returns
    -------
    numpy.ndarray
        The modified equations of shape (n^2-1, n^2-1)

    """
    if equations.shape[-1] != math.isqrt(equations.shape[-1])**2:
        # full equations shape should be perfect square
        raise RydiquleError("Ground state already removed")

    basis_size = int(np.sqrt(equations.shape[-1]))
    eqn_size = equations.shape[-1]

    # get the constant term
    eqns_column1 = equations[...,0]
    constant_term = equations[...,1:,0]

    # find the indices where populations need to be subtracted
    plocations = np.array([(basis_size+1)*x for x in range(basis_size)])
    pvector = np.array([int(i in plocations) for i in range(eqn_size)])

    # make a matrix to subtract populations
    pop_subtract = np.einsum('...i,j', eqns_column1, pvector)

    # subtract populations
    equation_new = equations - pop_subtract

    # remove the ground state
    equations_reduced = equation_new[..., 1:, 1:]

    return equations_reduced, constant_term


def make_real(equations: np.ndarray, constant: np.ndarray,
              ground_removed: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts equations of motion from complex basis to real basis.

    Changes the density vector equation for p_ij into the Re[p_ij] equation
    and changing the density vector equation for p_ji into the equation for Im[p_ij].

    Parameters
    ----------
    equations : numpy.ndarray
        Complex equations of motion.
    constant : numpy.ndarray
        RHS of the equations of motion.
    ground_removed : bool, optional
        Indicates if `equations` has had the ground state removed.
        Default is `True`.

    Returns
    -------
    real_eqns : numpy.ndarray
        EOMs in real basis.
    real_const : numpy.ndarray
        RHS of EOMs in real basis.

    """
    # Define the basis for printout purposes for ground removed or not removed
    if np.sqrt(equations.shape[-1]) % 1 == 0:  # ground is not removed

        basis_size = int(np.sqrt(equations.shape[-1]))

    elif np.sqrt(equations.shape[-1]+1) % 1 == 0:  # ground is removed

        basis_size = int(np.sqrt(equations.shape[-1]+1))

    else:
        raise RydiquleError("unsupported equation shape")

    u, u_inv = get_basis_transform(basis_size)  # unitary transformation matrix

    if ground_removed:
        u = u[1:,1:]
        u_inv = u_inv[1:,1:]

    # transform to the real basis
    new_eqns = u@(equations@u_inv)
    new_const = np.zeros(equations.shape[:-1])

    if ground_removed:
        new_const = np.einsum('ij,...j', u, constant)

    # copies allow complex base arrays to be GC
    return new_eqns.real.copy(), new_const.real.copy()


def get_basis_transform(basis_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function that defines the basis transformation matrix u and its inverse u_i,
    between the real and complex basis.

    This matrix u implements that the :math:`\\rho[j,i] \\rightarrow Re(\\rho[j,i])`
    and :math:`\\rho[i,j] \\rightarrow Im(\\rho[j,i])`.

    The transformation is not quite unitary, due to the asymmetry of the factors of 1/2.

    Parameters
    ----------
    basis_size : int
        Size of the basis to generate transformations for.
    
    Returns
    -------
    u : numpy.ndarray
        Forward transformation matrix.
    u_inv : numpy.ndarray
        Inverse transformation matrix.

    Raises
    ------
    RydiquleError
        If `basis_size` does not match current basis.

    """
    try:
        u = U[0]
        u_inv = U[1]
        basis = B[0]
        if basis != basis_size:
            raise RydiquleError("new basis size")
        return u, u_inv

    except (KeyError, RydiquleError):
        eqn_size = basis_size**2
        u = np.zeros((eqn_size, eqn_size), dtype=complex)
        u_inv = np.zeros((eqn_size, eqn_size), dtype=complex)
        ipairs = np.array([[i for i in range(basis_size)] for j in range(basis_size)]).flatten()
        jpairs = np.array([[j for i in range(basis_size)] for j in range(basis_size)]).flatten()
        index = np.array(range(eqn_size))

        for i,j,a in zip(ipairs,jpairs, index):
            for k,l,b in zip(ipairs, jpairs, index):

                # population equations dont change
                if i == j and j == k and k == l:
                    u[a,b] = 1
                    u_inv[a,b] = 1
                # define the  2x2 transformation between real and imaginary
                elif i == k and j == l and i > j:
                    u[a,b] = 1/2
                    u_inv[a,b] = 1
                elif i == k and j == l and i < j:
                    u[a,b] = 1j/2
                    u_inv[a,b] = -1j
                elif i == l and j == k and i > j:
                    u[a,b] = 1/2
                    u_inv[a,b] = 1j
                elif i == l and j == k and i < j:
                    u[a,b] = -1j/2
                    u_inv[a,b] = 1

        U[0] = u
        U[1] = u_inv
        B[0] = basis_size
        return u, u_inv


def convert_to_full_dm(dm: np.ndarray) -> np.ndarray:
    """
    Converts density matrices from rydiqule's computational basis (real, with state 0 removed)
    to the full, real basis (ie with state 0 population inserted).

    Solutions computed using one of rydiqule's built in solvers will always output solutions in the 
    real computational basis which is intended as the input for this function.

    Parameters
    ----------
    dm: numpy.ndarray
        Density matrices in rydiqule's computational basis (real, with state 0 removed).
        Has shape `(..., b**2-1)` where `b` is the number of states in the basis.

    Returns
    -------
    numpy.ndarray
        Density matrices in the real basis with state 0 present.
        Will have shape `(..., b**2)`.

    Raises
    ------
    RydiquleError
        If final dimension is of invalid size
        (i.e. does not correspond to b**2-1, where b is an integer)

    Examples
    --------
    >>> [g, e] = rq.D2_states('Rb85')
    >>> c = rq.Cell('Rb85', [g, e], cell_length =  0.00001)
    >>> c.add_coupling(states=(g, e), rabi_frequency=1, detuning=1)
    >>> sols = rq.solve_steady_state(c)
    >>> print(sols.rho)
    [0.001371 0.02613  0.000686]
    >>> print(rq.sensor_utils.convert_to_full_dm(sols.rho))
    [9.993144e-01 1.371165e-03 2.612972e-02 6.855828e-04]

    """
    b, r = divmod(math.sqrt(dm.shape[-1]+1), 1.0)
    if r != 0.0:
        raise RydiquleError('Computational basis size incorrect. '
                            'Final dimension must correspond to b**2 - 1, '
                            'where b is an integer.')
    b = int(b)

    nonzero_state_pops = dm[..., b::b+1]
    zero_state_pops = 1.0 - nonzero_state_pops.sum(axis=-1)
    
    full_dm = np.concatenate((zero_state_pops[...,np.newaxis], dm), axis=-1)

    return full_dm


def convert_dm_to_complex(dm: np.ndarray) -> np.ndarray:
    """
    Converts density matrices from rydiqule's computational basis (real, with state 0 removed)
    to a standard complex basis with all states present.

    Solutions computed using one of rydiqule's built in solvers will always output solutions in the 
    real, 1-dimensional computational basis which is intended as the input for this function.
    Note that while a full complex density matrix can be useful to view solutions, performing
    calculations with the full complex density matrix can often add considerable unwanted rounding
    errors.

    Parameters
    ----------
    dm: numpy.ndarray
        Density matrices in rydiqule's computational basis (real, with state 0 removed).
        Has shape `(..., b**2-1)` where `b` is the number of states in the basis.

    Returns
    -------
    numpy.ndarray
        Density matrices in the complex basis with state 0 present.
        Will have shape `(..., b, b)`.

    Raises
    ------
    RydiquleError
        If final dimension is of invalid size
        (i.e. does not correspond to b**2-1, where b is an integer)

    Examples
    --------
    >>> [g, e] = rq.D2_states('Rb85')
    >>> c = rq.Cell('Rb85', [g, e], cell_length =  0.00001)
    >>> c.add_coupling(states=(g, e), rabi_frequency=1, detuning=1)
    >>> sols = rq.solve_steady_state(c)
    >>> print(sols.rho)
    [0.001371 0.02613  0.000686]
    >>> print(rq.sensor_utils.convert_dm_to_complex(sols.rho))
    [[9.993144e-01+0.j      1.371166e-03+0.02613j]
     [1.371166e-03-0.02613j 6.855828e-04+0.j     ]]

    """
    b, r = divmod(math.sqrt(dm.shape[-1]+1), 1.0)
    if r != 0.0:
        raise RydiquleError('Computational basis size incorrect. '
                            'Final dimension must correspond to b**2 - 1, '
                            'where b is an integer.')
    b = int(b)
    
    _, u_inv = get_basis_transform(b)
    
    full_dm = convert_to_full_dm(dm)
    
    stack_shape = full_dm.shape[:-1]

    complex_dm = np.einsum('ij,...j', u_inv, full_dm).reshape((*stack_shape, b, b))

    return complex_dm


def convert_complex_to_dm(complex_dm: np.ndarray) -> np.ndarray:
    """
    Converts a standard density matrices in the complex basis with ground state
    into rydiqule's computational real basis with ground state removed.

    `rydiqule`'s built-in functions do not return complex density matrices, so this function is used
     internally and to undo the conversion of :func:`~.sensor_utils.convert_dm_to_complex`. 

    Parameters
    ----------
    complex_dm: numpy.ndarray
        Stack of density matrices in the complex basis with ground state present.
        Has shape of `(..., b, b)` where `b` is the number of states in the basis.

    Returns
    -------
    numpy.ndarray
        Density matrices in rydiqule's computational basis (real with state 0 removed).
        Has shape `(..., b**2-1)`.

    Raises
    ------
    RydiquleError
        If the provided density matrices are not square.
    RydiquleError
        If the converted matrix is not real (implies non-hermitian)
    """
    
    shape = complex_dm.shape[-2:]
    stack_shape = complex_dm.shape[:-2]
    if not shape[0] == shape[1]:
        raise RydiquleError('Input density matrices must be square')
    b = shape[0]
    
    u, _ = get_basis_transform(b)
    
    flat_ground_removed = complex_dm.reshape(stack_shape + (b**2,))[...,1:]
    
    dm = np.einsum('ij,...j', u[1:,1:], flat_ground_removed)
    
    real_dm = np.real_if_close(dm)

    if np.iscomplexobj(real_dm):
        raise RydiquleError('Converted matrix is not real, likely unphysical')
    
    return dm


def check_positive_semi_definite(dm: np.ndarray):
    """
    Checks if the provided matrices is a physical density matrix.

    This is done by confirming each matrix of the stack is positive semi-definite.

    Parameters
    ----------
    dm: numpy.ndarray
        Stack of density matrices in rydiqule's computational basis
        (i.e. real with ground state removed).
        Expected shape is `(..., N)` where `N = basis_size**2-1`.

    Raises
    ------
    RydiquleError
        If at least one density matrix of the stack is not positive semi-definite.
    """
    
    complex_dm = convert_dm_to_complex(dm)
    
    # calculate eigenvalues, attempt to convert to real if no complex parts
    eigs = np.real_if_close(np.linalg.eigvals(complex_dm))
    # use same tolerance as np.real_if_close to determine if zero
    tol = np.finfo(eigs.dtype).eps * 100 
    try:
        if np.iscomplexobj(eigs):
            raise RydiquleError('Eigenvalues are not real')
        if not np.all(np.isclose(eigs.sum(axis=-1), 1.0)):
            raise RydiquleError('Eigenvalues do not sum to 1.0')
        if not np.all(eigs >= -tol):
            raise RydiquleError('Eigenvalues are not all >= 0.0')
    except RydiquleError as err:
        raise RydiquleError('At least one density matrix is not positive semi-definite. '
                            'This is unphysical.') from err


def _get_rho(n: int) -> np.ndarray:
    """
    Helper function which gets the projectors for calculating the EOMs.

    Uses a gobal dictionary to cache results and avoid recalculating.

    Parameters
    ----------
    n : int
        Basis size to get projectors for.

    Returns
    -------
    numpy.ndarray
        Projector matrices for the basis.

    """
    try:
        return RHO[n]
    except KeyError:
        rho = np.zeros((n,n,n,n))
        for i in range(n):
            for k in range(n):
                rho[i,k,i,k] = 1

        RHO[n] = rho
        return rho


def get_rho_ij(sols: Union[np.ndarray, Solution],
               i: int, j: int) -> Union[complex, np.ndarray]:
    """
    For a given density matrix solution, retrieve a specific element of the density matrix.

    Assumes the ground state of the solution is eliminated (as per :func:`~.remove_ground`),
    and assumes Rydiqule's nominal state ordering of the Density Vector (per :func:`~.make_real`).

    Parameters
    ----------
    sols : numpy.ndarray or :class:`~.Solution`
        Solutions to extract the matrix element for.
        Can be either the solution object returned by the solve or
        an N-D array representing density vectors, with ground state removed,
        and written in the totally real equations.
    i : int
        density matrix index i
    j : int
        density matrix index j

    Returns
    -------
    numpy.ndarray
        Array of rho_ij values.
        Will be of type float when `i==j`.
        Will be of type complex128 when `i!=j`.

    Examples
    --------
    >>> sols = np.arange(180).reshape((4,5,3,3))
    >>> print(sols.shape)
    (4, 5, 3, 3)
    >>> rho_01 = rq.get_rho_ij(sols, 0,1)
    >>> print(rho_01.shape)
    (4, 5, 3)
    >>> print(rho_01[0,0])
    [0.-1.j 3.-4.j 6.-7.j]

    """
    rhos = _validate_sols(sols)

    b = int(np.sqrt(rhos.shape[-1]+1))  # basis size
    if i == 0 and j == 0:
        zero_state_pop = 1.0
        for k in range(1,b):
            zero_state_pop -= rhos[...,b*k+k-1]
        return zero_state_pop

    elif i == j:
        return rhos[...,b*j+i-1]

    elif i > j:
        realpart = rhos[...,b*j+i-1]
        imagpart = rhos[...,b*i+j-1]
        return realpart + 1j*imagpart
    else:
        realpart = rhos[...,b*i+j-1]
        imagpart = rhos[...,b*j+i-1]
        return realpart - 1j*imagpart


def get_rho_populations(sols: Union[np.ndarray, Solution]) -> np.ndarray:
    """
    For a given density matrix solution, return the diagonal populations.

    Note that rydiqule's convention for removing the ground state forces population conservation,
    ie the sum of these populations will be 1.

    Parameters
    ----------
    sols: numpy.ndarray or :class:`~.Solution`
        Solutions to extract the matrix element for.
        Can be either the solution object returned by the solve or
        an N-D array representing density vectors, with ground state removed,
        and written in the totally real equations.

    Returns
    -------
    numpy.ndarray
        Populations of the density matrices.
        Will have same shape as input solutions, with the last dimension
        reduced to the basis size.
    """

    rhos = _validate_sols(sols)

    b = int(np.sqrt(rhos.shape[-1]+1))  # basis size
    nonzero_state_pops = rhos[...,b::b+1]
    zero_state_pop = 1.0 - nonzero_state_pops.sum(axis=-1)

    return np.concatenate((zero_state_pop[...,np.newaxis], nonzero_state_pops), axis=-1)


def scale_dipole(dipole: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Scale a dipole matrix from units of a0*e to Mrad/s when multiplied by a field in V/m.

    Parameters
    ----------
    dipole : float or numpy.ndarray
        Array of dipole moments in units of a0*e.
        These are the default units used by ARC.

    Returns
    -------
    numpy.ndarray
        Scaled array in units of (Mrad/s)/(V/m)

    """
    scale_factor = (a0*e)/hbar*1e-6
    dipole = dipole * scale_factor
    return dipole


def draw_diagram(sensor: "Sensor", include_dephasing: bool = True) -> LD:
    """
    Draw a matplotlib plot that shows the energy level diagram, couplings, and dephasing paths.

    To show the plot, call `plt.show()`.
    If in a jupyter notebook, this is handled automatically.

    Diagram has horizontal lines for the energy levels (spacing not to scale).
    Integer labels refer to the internal indexing for each state.
    If sensor is of type :class:`~.Cell`, will also add text labels to each state of the
    quantum numbers.

    Solid arrows between states are couplings defined with a non-zero Rabi frequency.
    Dashed arrows between states are couplings defined with a dipole moment.

    Wiggly arrows between states denote a dephasing pathway.
    Opacity represents strength of dephasing relative to the largest specified dephasing,
    where fully opaque is the largest dephasing.

    Parameters
    ----------
    sensor : :class:`~.Sensor`
        Sensor object to diagram.
    include_dephasing : bool, optional
        Whether to plot dephasing paths. Default is `True`.

    Returns
    -------
    :class:`leveldiagram.LD`
        Diagram handle

    """
    rq_g = sensor.couplings.copy()
    sensor_states = sensor.states

    ### level settings
    # use rotating frames to send levels up or down relative to others
    frames = sensor.get_rotating_frames()
    # get flattend dictionary of all state paths
    # also calculates energy based on sum of frame signs in path
    energies = {}
    for d in list(frames.values()):
        for k, v in d.items():
            energies[k] = energies[k] = sum([i for (_,i) in v[1:]])
    
    for lev, vals in rq_g.nodes.items():
        ld_kw = vals.get('ld_kw', {})
        ld_kw['energy'] = energies.get(lev, 0.0)
        
        rq_g.nodes[lev]['ld_kw'] = ld_kw

    ### get decoherence normalizations
    gamma_matrix = sensor.decoherence_matrix()
    # we get the biggest possible decoherence value for each term
    # by doing a max reduction along stack axes
    stack_axes = tuple(np.arange(0, gamma_matrix.ndim-2))
    gamma_matrix = gamma_matrix.max(axis=stack_axes)
    # get overall max/min
    if not np.any(gamma_matrix):
        # no dephasings defined, ignore
        max_dephase = 1.0
        min_dephase = 0.1
    else:
        max_dephase = gamma_matrix.max()
        min_dephase = gamma_matrix[gamma_matrix != 0.0].min()
        if np.isclose(min_dephase, max_dephase):
            # all non-zero dephasings are the same, prevent /0 error in normalization
            min_dephase = max_dephase*1e-1

    ### get rabi normalizations
    hamiltonian = sensor.get_hamiltonian() + sensor.get_time_hamiltonian(0)
    # get biggest element along each stack
    hamiltonian = np.abs(hamiltonian).max(axis=stack_axes)
    # set diagonals to zero
    np.einsum('...ii->...i', hamiltonian)[:] = 0
    # get overall max/min
    max_rabi = 2*hamiltonian.max()
    min_rabi = np.min(2*hamiltonian[hamiltonian != 0.0])
    if np.isclose(max_rabi, 0.0):
        raise RydiquleError('No non-zero couplings! Cannot draw diagram')
    if np.isclose(max_rabi, min_rabi):
        min_rabi = max_rabi*1e-1

    ### coupling settings
    max_lw = 3.0
    norm_lw = 1.0
    min_lw = 0.3
    for edge, vals in rq_g.edges.items():
        ld_kw = vals.get('ld_kw', {})
        if 'phase' in vals:
            # only add ld_kw here if coherent coupling
            if 'time_dependence' in vals:
                t_factor = vals.get('time_dependence')(0.0)
                ld_kw['linestyle'] = 'dashed'
            else:
                t_factor = 1.0
            c_rabi_max = np.max(np.abs(vals.get('rabi_frequency'))) * t_factor
            if np.isclose(c_rabi_max, 0.0):
                ld_kw['hidden'] = True
            else:
                # set lw based on rabi/max_dephasing
                ld_kw['lw'] = (max_lw - min_lw)/(max_rabi - min_rabi)*(
                    c_rabi_max/max_dephase - 1
                ) + norm_lw

            rq_g.edges[edge]['ld_kw'] = ld_kw

    ### decoherence settings
    if include_dephasing and gamma_matrix.any():
        # reversing order of traverse to prevent transit overlaps
        idxs = np.argwhere(gamma_matrix != 0.0)[::-1,:]
        for idx in idxs:
            
            states = (sensor_states[idx[0]], sensor_states[idx[1]])
            
            ld_kw = rq_g.edges[states].get('ld_kw', {})

            ld_kw['wavy'] = True
            ld_kw['deflect'] = True
            ld_kw['start_anchor'] = 'right'
            if idx[0] == idx[1]:
                ld_kw['deflection'] = 0.15
                ld_kw['stop_anchor'] = (0.1, 0.0)
            else:
                ld_kw['stop_anchor'] = (0.4, 0.0)
            # ensure alpha doesn't get too small to not be seen
            # also uses a log scale for the full range of non-zero dephasings
            alph = 1-(0.8*np.log10(gamma_matrix[tuple(idx)]/max_dephase
                                )/np.log10(min_dephase/max_dephase))
            ld_kw['alpha'] = alph

            rq_g.edges[states]['ld_kw'] = ld_kw

    else:
        # remove edges that have only dephasings on them
        edges_to_remove = [(i,j)for i,j,k in rq_g.edges(data=True)
                            if 'phase' not in k ]
        rq_g.remove_edges_from(edges_to_remove)

    ### create diagram handle and draw
    ld = LD(rq_g, use_ld_kw=True)
    ld.draw()

    return ld


def match_states(statespec: StateSpec, compare_list: List[State]) -> List[State]:
    """Return all states in a list matching the pattern described by a given specification.

    A `StateSpec` is described by a tuple containing floats or strings, or lists of floats or
    strings. A state `s` in `compare_list` is considered a match to `statespec` if `s` and 
    `statespec` are the same length and for each element in `statespec`, :

        1. The corresponding element in `s` is equal to the element in `statespec` (in the case of a
            single value)
        2. The element is `s` is an element of the list which is an element of `statespec` (in the
            case of a list element of `statespec`).
        3. The element of `statespec` is the string `"all"`

    Parameters
    ----------
    statespec : StateSpec
        The state specification against which to compare elements of the list.
    compare_list : List[State]
        The list of individual states to compare.

    Returns
    -------
    list of State
        Sublist of `compare_list` containing all elements of `compare_list` matching `statespec`. 

    Examples
    --------
    While primarily intended for internal use, `match_states` can be accessed directly through
    `sensor_utils`.

    >>> compare_states = [(0,0),
    ...       (1,0),(1,1),
    ...       (2,0),(2,1),(2,2)
    ...      ]
    >>> spec = (2,[0,1,2])
    >>> print(rq.sensor_utils.match_states(spec, compare_states))
    [(2, 0), (2, 1), (2, 2)]
    >>> wildcard_spec = (2,"all")
    >>> print(rq.sensor_utils.match_states(wildcard_spec, compare_states))
    [(2, 0), (2, 1), (2, 2)]
    
    """
    #handle the case where the state_spec is not a tuple but a list of single values
    #case 1. statesepc is a lis of integers
    if isinstance(statespec, list):
        return [state for state in statespec if state in compare_list]
    
    #case 2. state is a single int
    elif not isinstance(statespec, tuple):
        #should only ever return a 1 or 0 element list
        return [state 
                for state in compare_list
                if state==statespec]

    #case 3. the test statespec is in the compare_list directly. guarantee only 1 match
    elif statespec in compare_list:
        return [statespec]
    
    #case 4. everything else, need to check patterns 
    else:
        #shave down list to only tuples with length matching statespec-
        #anything else cant match and will be a problem later
        compare_list_tuple = [s for s in compare_list if isinstance(s, tuple) and len(s)==len(statespec)]
        
        #not strictly necessary, but casting all elements being list makes comparison easy with "in"
        spec_qnums_as_list = [[qn] if not isinstance(qn, (list,str)) else qn for qn in statespec]

        matching_states = []
        for state in compare_list_tuple:
            
            qnum_match = [qn1=="all" or qn_compare in qn1
                        for qn1, qn_compare in zip(spec_qnums_as_list, state)]
            if all(qnum_match):
                matching_states.append(state) #should be safe since tuples are immutable

        return matching_states


def expand_statespec(statespec: StateSpec) -> List[State]:
    """    Returns a list of all possible `states` corresponding to a given `statespec`.

    A `state` in `rydiqule` is defined by either a floating point or string value, or a tuple
    of such values. A `StateSpec` can replace any float or string value with a list of such
    values. The `expand_statespec` function's purpose is to convert a single `statespec` in which
    some number of elements are defined as lists into a list of all the states which correspond to 
    the state.

    If the provided spec is only a single state, a 1-element list containing that state is returned.

    Parameters
    ----------
    statespec : StateSpec
        State specification with either zero or one element defined as a list. 

    Returns
    -------
    list of State
        List of all states matching the provided statespec

    Raises
    ------
    RydiquleError
        If the provided `statespec` is not a valid state specification.

    Notes
    -----
    ..note:
        This function will preserve the state type for namedtuple statespecs. So for example,
        passing an :class:`~.atom_utils.A_QState` for example will return a list of states of the
        same type.

    Examples
    --------
    >>> ground = (0,0)
    >>> excited = (1, [0,1])
    >>> print(rq.expand_statespec(ground))
    [(0, 0)]
    >>> print(rq.expand_statespec(excited))
    [(1, 0), (1, 1)]

    This function has utility in allowing otherwise cubersome state definitions to be
    defined with variables in :class:`sensor.Sensor` functions. 

    >>> g = (0,0)
    >>> e = (1, [-1,0,1])
    >>> [em1, e0, ep1] = rq.expand_statespec(e)
    >>> s = rq.Sensor([g,e])
    >>> cc = {(g,em1): 0.25,
    ...       (g,e0): 0.5,
    ...       (g,ep1): 0.25}
    >>> s.add_coupling((g,e), detuning=1, rabi_frequency=2, coupling_coefficients=cc, label="probe")
    >>> print(s.get_hamiltonian())
    [[ 0.  +0.j  0.25+0.j  0.5 +0.j  0.25+0.j]
     [ 0.25-0.j -1.  +0.j  0.  +0.j  0.  +0.j]
     [ 0.5 -0.j  0.  +0.j -1.  +0.j  0.  +0.j]
     [ 0.25-0.j  0.  +0.j  0.  +0.j -1.  +0.j]]
        
    """
    ## resolve more trivial cases of single states
    if isinstance(statespec, (int, str)):
        return [statespec]
    # transparent to lists
    elif isinstance(statespec, list):
        return statespec
    #make sure its not a dumb type
    elif not isinstance(statespec, tuple):
        raise RydiquleError(f"{statespec} is not a valid state specification.")
    
    statespec_type = type(statespec)

    spec_qnums_as_list = [[qn] if not isinstance(qn, (list)) else qn for qn in statespec]
    states_list = [state for state in product(*spec_qnums_as_list)]

    if hasattr(statespec, "_make"):
        return [statespec_type._make(state) for state in states_list]
    else:
        return states_list   


def state_tuple_to_str(states:States) -> str:
    """Helper function to create a more terse string representation of a tuple of state tuples.

    The default python behavior for a str representation of tuples is to use `__repr__` for
    individual elements. We want to use `str`, since the output is pointlessly long otherwise.
    `A_QState.__repr__()` is longer than `A_QState.__str__()`

    Parameters
    ----------
    states : tuple of states
        States for which to produce a string representation.
    """
    return "(" + ",".join([str(s) for s in states]) + ")"


def _validate_sols(sols) -> np.ndarray:
    """Helper function to validate that solutions are of an appropriate type.
    There are 3 outcomes:
    
      - `sols` is a np.ndarray, returns `sols`.
      - `sols` is an object with a `rho` attribute that is a `numpy.ndarray`,
        in which case returns `rho`.
      - `sols` does not meet either of the above criteria, in which case raises an exception.

    Parameters
    ----------
    sols : any
        The value to validate, should be a np.ndarray or an object
        (like a :class:`~.sensor_solution.Solution`)
        with a `rho` attribute which is a numpy array.
        
    Raises
    ------
    RydiquleError:
        If `sols` is not an array or object with a `rho` attribute that is an array.
    """
    if hasattr(sols, "rho"):
        rho = sols.rho
    else:
        rho = sols
        
    if not isinstance(rho, np.ndarray):
        raise RydiquleError(
            "sols must be a numpy array or have an attribute \"rho\" which is a numpy array.")
    
    return rho


def _squeeze_dims(couplings:nx.Graph):
    """Squeezes all array parameters on the graph into 1-dimensional arrays.

    Modifies the parameters in-place rather than returning a modified object.
    If provided as a sensor, will modify the `couplings` graph of the sensor.

    Parameters
    ----------
    sensor : nx.DiGraph or Sensor
        The sensor or graph for which the parameters will be squeeze.

    Raises
    ------
    TypeError
        If `sensor` is not a graph or :class:`~.Sensor`.
    """
    for states in couplings.edges:
        for param in couplings.edges[states]:
            if isinstance(couplings.edges[states][param], np.ndarray):
                new_shape = [dim for dim in couplings.edges[states][param].shape if dim != 1]
                couplings.edges[states][param].shape = new_shape
    

        