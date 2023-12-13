"""
Utilities used by the Sensor classes.
"""
import string
import math

import numpy as np
from .sensor_solution import Solution
import scipy.constants
from scipy.constants import hbar, e
from leveldiagram import LD

from typing import Dict, Tuple, Union, Sequence, List, Callable, TYPE_CHECKING
if TYPE_CHECKING:
    # only import when type checking, avoid circular import
    from .sensor import Sensor

a0 = scipy.constants.physical_constants["Bohr radius"][0]

# put composite types of Sensor/Cell/Solution here
ScannableParameter = Union[float, List[float], np.ndarray]

CouplingDict = Dict
State = Union[int, str]
States = Tuple[State, State]
QState = Sequence  # TODO: consider using named tuples here

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
        and is not officially supported
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
    ValueError: If the shapes of gamma_matrix and hamiltonian are not matching
        or not square in the last 2 dimensions

    Examples
    --------
    >>> ham = np.diag([1,-1])
    >>> gamma = np.array([[.1, 0],[.1,0]])
    >>> print(ham.shape)
    >>> eom, const = rq.generate_eom(ham, gamma)
    >>> print(eom)
    >>> print(const.shape)
    (2, 2)
    [[-0.1  2.   0. ]
     [-2.  -0.1  0. ]
     [ 0.   0.  -0.1]]
    (3,)

    This also works with a "stack" of multiple hamiltonians:

    >>> ham_base = np.diag([1,-1])
    >>> ham_full = np.array([ham_base for _ in range(10)])
    >>> gamma = np.array([[.1, 0],[.1,0]])
    >>> print(ham_full.shape)
    >>> eom, const = rq.generate_eom(ham_full, gamma)
    >>> print(eom.shape)
    >>> print(const.shape)
    (10, 2, 2)
    (10, 3, 3)
    (10, 3)

    """
    if not hamiltonian.shape[-2:] == gamma_matrix.shape[-2:]:
        raise ValueError("hamiltonian and gamma matrix must have matching shape")
    if not hamiltonian.shape[-1] == hamiltonian.shape[-2]:
        raise ValueError("hamiltonian and gamma matrix must be square")

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
        obes, const = make_real(obes, const)

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
        raise ValueError("Ground state already removed")

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
        raise ValueError("unsupported equation shape")

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
    ValueError
        If `basis_size` does not match current basis.

    """
    try:
        u = U[0]
        u_inv = U[1]
        basis = B[0]
        if basis != basis_size:
            raise ValueError("new basis size")
        return u, u_inv

    except (KeyError, ValueError):
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
    >>> rho_01 = rq.get_rho_ij(sols, 0,1)
    >>> print(rho_01.shape)
    >>> print(rho_01[0,0])
    (4, 5, 3, 3)
    (4, 5, 3)
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
    from .cell import Cell
    rq_g = sensor.couplings.copy()

    # level settings
    # use rotating frames to send levels up or down relative to others
    frames = sensor.get_rotating_frames()
    # get flattend dictionary of all state paths
    # also calculates energy based on sum of frame signs in path
    energies = {k: np.sum(np.sign(v)[1:]) # ignore first state so all subgraphs start at 0
                for d in list(frames.values()) # get list of subgraph path dicts
                for k, v in d.items()}
    for lev, vals in rq_g.nodes.items():
        ld_kw = vals.get('ld_kw', {})
        ld_kw['energy'] = energies[lev]
        
        if isinstance(sensor, Cell):
            ld_kw['left_text'] = vals['qnums']
        
        rq_g.nodes[lev]['ld_kw'] = ld_kw

    # coupling settings
    for edge, vals in rq_g.edges.items():
        ld_kw = vals.get('ld_kw', {})
        if 'dipole_moment' in vals:
            ld_kw['linestyle'] = 'dashed'
        elif 'rabi_frequency' in vals:
            if not np.all(vals.get('rabi_frequency')):
                ld_kw['hidden'] = True

        rq_g.edges[edge]['ld_kw'] = ld_kw

    # decoherence settings

    # get decoherence normalizations
    gamma_matrix = sensor.decoherence_matrix()
    # we get the biggest possible decoherence value for each term
    # by doing a max reduction along stack axes
    stack_axes = tuple(np.arange(0, gamma_matrix.ndim-2))
    gamma_matrix = gamma_matrix.max(axis=stack_axes)

    if include_dephasing and gamma_matrix.any():
        max_dephase = gamma_matrix.max()
        min_dephase = gamma_matrix[gamma_matrix != 0.0].min()
        if np.isclose(min_dephase, max_dephase):
            # all non-zero dephasings are the same, prevent /0 error in normalization
            min_dephase = max_dephase*1e-1

        # reversing order of traverse to prevent transit overlaps
        idxs = np.argwhere(gamma_matrix != 0.0)[::-1,:]
        for idx in idxs:

            ld_kw = rq_g.edges[tuple(idx)].get('ld_kw', {})

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

            rq_g.edges[tuple(idx)]['ld_kw'] = ld_kw

    # create diagram handle
    ld = LD(rq_g, use_ld_kw=True)
    ld.draw()

    return ld


def _get_collapse_str(len: int, *matched_dims) -> str:
    """
    Internal helper function to build the string argument of `numpy.einsum`
    when dimensions are collapsed.

    Creates a string of the appropriate number of ascii characters
    (the first n for an n-dimensional stack).
    Then swaps in new characters in the appropriate place to produce a string that can be passed
    to `numpy`'s `einsum` function.
    """
    # left-hand side and right-hand side of equation in einstein summation convention
    idxs_rhs = string.ascii_lowercase[:len]
    idxs_lhs = string.ascii_lowercase[:len]

    for i, dims in enumerate(matched_dims):
        new_idx = string.ascii_lowercase[-(i+1)]

        for d in dims:
            old_idx = idxs_rhs[d]
            idxs_rhs = idxs_rhs.replace(old_idx, new_idx)
            idxs_lhs = idxs_lhs.replace(old_idx, "")

        idxs_lhs += new_idx

    full_expression = idxs_rhs + "...->" + idxs_lhs + "..."
    return full_expression


def _validate_sols(sols):
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
    ValueError:
        If `sols` is not an array or object with a `rho` attribute that is an array.
    """
    if hasattr(sols, "rho"):
        rho = sols.rho
    else:
        rho = sols
        
    if not isinstance(rho, np.ndarray):
        raise TypeError(
            "sols must be a numpy array or have an attribute \"rho\" which is a numpy array.")
    
    return rho
    

def _combine_parameter_labels(*labels: str) -> str:
    """
    Combine 2 or more parameter label strings into a single label.

    Labels are grouped by parameter type (detuning, rabi_freq, etc),
    separated by a |

    """

    return "|".join(labels)


    ## LEAVING IN CASE WE WANT TO USE THIS

    # new_label = []

    # couplings = []
    # params = []

    # #get lists of all the coupling labels and param types
    # for label in labels:
    #     coupling, param = label.split("_", 1)
    #     couplings.append(coupling)
    #     params.append(param)

    # #build the full label from the lists
    # #under this convention, couplings will be grouped in the label
    # #by parameter type, with parameter types separated by a pipe
    # for p_base in params: #loop over parameter types, skipping those that have already been done
    #     if (p_base + "|") in new_label:
    #         continue

    #     for c, p_test in zip(couplings, params): #for each parameter type, get all labels
    #         if p_test == p_base:
    #             new_label.append(c + "_")
    #     new_label.append(p_base + "|") #add a pipe to separate parameter types

    # #turn the full label array into a string, get rid of pipe on the end
    # return "".join(new_label)[:-1]
