"""
Utilities for interacting with atomic parameters and ARC.
"""

import arc.alkali_atom_data as arc_atoms
from scipy.constants import epsilon_0, hbar, c
import numpy as np
import re 
from .sensor_utils import expand_statespec

from typing import Union, Optional, Literal, Tuple, List, Dict, Callable, NamedTuple

from .exceptions import RydiquleError

ATOMS = {
    'H': arc_atoms.Hydrogen,
    'Li6': arc_atoms.Lithium6, 'Li7': arc_atoms.Lithium7,
    'Na': arc_atoms.Sodium,
    'K39': arc_atoms.Potassium39, 'K40': arc_atoms.Potassium40, 'K41': arc_atoms.Potassium41,
    'Rb85': arc_atoms.Rubidium85, 'Rb87': arc_atoms.Rubidium87,
    'Cs': arc_atoms.Caesium
}
"""
Alkali atoms defined by ARC that can be used with :class:`~.Cell`.
"""

ground_n = {
    "H": 1,
    "Li": 2,
    "Na": 3,
    "K": 4,
    "Rb": 5,
    "Cs": 6
}

splitting_qnums: Dict[Optional[str], tuple] = {
    None: (None, None, None),
    "fs": ("all", None, None),
    "hfs": (None, "all", "all")
}

# optional keys: (m_j, f, m_f)
# (n,l,j are required so not included here
STATE_TYPES = {
    (False, False, False): "NLJ",
    (True, False, False): "FS",
    (False, True, True): "HFS"
}

QSpec = Union[float, List[float], Literal['all']]
class A_QState(NamedTuple):
    """
    Named tuple class designed to represent the quantum numbers a state spec of an alkali
    atom. `n`, `l`, and `j` quantum numbers are required, with optional `m_j`, `f`, and `m_f`.
    """
    n: int
    l: int
    j: float
    m_j: Optional[QSpec] = None
    f: Optional[QSpec] = None
    m_f: Optional[QSpec] = None

    def __str__(self):
        """Compact string representation of an `A_QState` that removes labels for 
        n, l, and j as well as any of m_j, f, and m_f that are `None`. 
        Also removes `"A_QState"` from the print output. 

        Returns
        -------
        str
            String representation of state. Prunes redundant/bulky parts of a standard NamedTuple
            output. 
        """
        return self.__repr__().replace("n=","").replace("l=","").replace("j=","",1)
    
    def __repr__(self):
        """Overload of the standard `__repr__` function which removes "A_QState" from the front of
        the string and removes all the `None` values. This doesn't change any of the values of the
        state, just prunes the output string to a more printout-friendly format.

        Returns
        -------
        str
            Pruned representation of A_QState.
        """
        return '(' + ', '.join(f'{name}={val!r}' for name, val in zip(self._fields, self) if val is not None) + ')'

    @property
    def qnums(self) -> Tuple[QSpec, ...]:
        return tuple(i for i in self if i is not None)
    
    @property
    def stype(self) -> str:
        try:
            return STATE_TYPES[tuple(i is not None for i in self[3:])]  # pyright: ignore[reportArgumentType]
        except KeyError:
            raise RydiquleError(f'{self} has unrecognized type')
        
class QState(NamedTuple):
    """
    Named tuple class designed to represent the quantum numbers in the state of an alkali
    atom. `n`, `l`, and `j` quantum numbers are required, with optional `m_j`, `f`, and `m_f`.
    """
    n: int
    l: int
    j: float
    m_j: Optional[float] = None
    f: Optional[int] = None
    m_f: Optional[int] = None

    def __str__(self):
        """Compact string representation of an `A_QState` that removes labels for 
        n, l, and j as well as any of m_j, f, and m_f that are `None`. 
        Also removes `"A_QState"` from the print output. 

        Returns
        -------
        str
            String representation of state. Prunes redundant/bulky parts of a standard NamedTuple
            output. 
        """
        return self.__repr__().replace("n=","").replace("l=","").replace("j=","",1)
    
    def __repr__(self):
        """Overload of the standard `__repr__` function which removes "A_QState" from the front of
        the string and removes all the `None` values. This doesn't change any of the values of the
        state, just prunes the output string to a more printout-friendly format.

        Returns
        -------
        str
            Pruned representation of A_QState.
        """
        return '(' + ', '.join(f'{name}={val!r}' for name, val in zip(self._fields, self) if val is not None) + ')'

    @property
    def qnums(self) -> Tuple[float, ...]:
        """Return a basic `tuple` representation of an `A_QState` with all `None` values removed.

        Returns
        -------
        tuple
            Quantum numbers which are not `None`.
        """
        return tuple(i for i in self if i is not None)
    
    @property
    def stype(self) -> str:
        """Type of state. One of "NLJ", "FS", "HFS"

        Returns
        -------
        str
            String representing state type.
        """
        try:
            return STATE_TYPES[tuple(i is not None for i in self[3:])]  # pyright: ignore[reportArgumentType]
        except KeyError:
            raise RydiquleError(f'{self} has unrecognized type')


#STATE_TYPES but inverted and including nlj numbers (we reference often enough to precompute)
QNUMS_IN_STATE_SPEC = {state_type: (True, True, True) + qnums for qnums, state_type in STATE_TYPES.items()}


def ground_state(n: Union[int, str],
                 splitting:Literal[None, "fs", "hfs"]=None,
                 expand:bool=False) -> Union[A_QState, List[A_QState]]:
    """
    Retrieve `A_QState` for the ground state of an atom or principle quantum number.

    Optionally, include fine structure splitting or hyperfine splitting, which will include all 
    `m_j` or all `f` and `m_f` values respectively. By default, specifying splitting does not return
    a list of states, but rather the associated specification with `"all"` in the appropriate
    place. The `expand` keyword argument can be used to modify this behavior, returning a list.

    Parameters
    ----------
    n: int or str
        Either the string flag of the atom or the principle quantum number n of
        an atom. If string, must begin with ['H', 'Li', 'Na', 'K',
        'Rb', 'Cs'].
    splitting: {None, 'fs', 'hfs'}, optional
        Type of splitting. Must be one of `None`, `"fs"`, or `"hfs"`, corresponding to the
        inclusion of `(n,l,j)` only, `m_j`, or both `f` and `m_f` respectively.
    expand: boolean, optional
        For states with splitting, whether to return them as a list of all states. If `False`,
        return is a single state specification with `"all"` for the appropriate quantum numbers.
        If `True`, a list of all individual states is returned.

    Raises
    ------
    RydiquleError
        If `n` is not a valid atom string or integer value
    ValueError
        If `splitting` is not one of {None, "fs", "hfs"}.

    Returns
    -------
    A_QState:
        `A_QState` corresponding to the ground state of the provided atom with the provided
        splitting, or list of `A_QState`s if `expand` is `True`.

    Examples
    --------
    The simplest use is to return the nlj quantum numbers for a particular atom's ground state. 
    Principle quantum number and string atom flags can be used interchangeably.
    
    >>> atom = "Rb85"
    >>> print(rq.ground_state(atom))
    (5, 0, 0.5)
    >>> print(rq.ground_state(5))
    (5, 0, 0.5)

    This function also can return states with splitting, either as a list of states or as a
    manifold specification.

    >>> print(rq.ground_state(atom, splitting="fs"))
    (5, 0, 0.5, m_j='all')
    >>> print(rq.ground_state(5, splitting="fs", expand=True))
    [(n=5, l=0, j=0.5, m_j=-0.5), (n=5, l=0, j=0.5, m_j=0.5)]
    >>> print(rq.ground_state(atom, splitting="hfs"))
    (5, 0, 0.5, f='all', m_f='all')

    """
    if isinstance(n, str):

        # strip out isotope numbers, if any
        atom = re.sub(r'\d+', '', n)
        if atom in ground_n.keys():
            n = ground_n[atom]
        else:
            raise RydiquleError(f"For string value of n, must use one of {list(ground_n.keys())}")

    if not isinstance(n, int):
        raise RydiquleError(f"n must be int or str, but found type {type(n)}.")

    try:
        g_state = A_QState(n, 0, 1/2, *splitting_qnums[splitting])
    except KeyError:
        raise ValueError(f"Invalid splitting {splitting}")
    
    if expand:
        return expand_qnums([g_state])
    return g_state


def D1_excited(n: Union[int, str],
               splitting:Literal[None, "fs", "hfs"]=None,
               expand:bool=False) -> Union[A_QState, List[A_QState]]:
    """
    Retrieve `A_QState` of the excited state of the D1 line of an atom or principle quantum number.

    Optionally, include fine structure splitting or hyperfine splitting, which will include all 
    `m_j` or all `f` and `m_f` values respectively. By default, specifying splitting does not return
    a list of states, but rather the associated specification with `"all"` in the appropriate
    place. The `expand` keyword argument can be used to modify this behavior, returning a list.
    place.

    Parameters
    ----------
    n: int or str
        Either the string flag of the atom or the principle quantum number n of
        an atom. If string, must begin with ['H', 'Li', 'Na', 'K',
        'Rb', 'Cs'].
    splitting: {None, 'fs', 'hfs'}
        Type of splitting. Must be one of `None`, `"fs"`, or `"hfs"`, corresponding to the
        inclusion of `(n,l,j)` only, `m_j`, or both `f` and `m_f` respectively.
    expand: boolean, optional
        For states with splitting, whether to return them as a list of all states. If `False`,
        return is a single state specification with `"all"` for the appropriate quantum numbers.
        If `True`, a list of all individual states is returned.
    
    Raises
    ------
    RydiquleError
        If `n` is not a valid atom string or integer value
    ValueError
        If `splitting` is not one of {None, "fs", "hfs"}.

    Returns
    -------
    A_QState or list of A_QState:
        `A_QState` corresponding to the D1 excited state of the provided atom with the provided
        splitting, or list of `A_QState`s if `expand` is `True`.
    
    Examples
    --------
    The simplest use is to return the nlj quantum numbers for a particular atom's excited state of
    the D1 transtion. Principle quantum number and string atom flags can be used interchangeably.

    >>> atom = "Rb85"
    >>> print(rq.D1_excited(atom))
    (5, 1, 0.5)
    >>> print(rq.D1_excited(5))
    (5, 1, 0.5)

    This function also can return states with splitting, either as a list of states or as a
    manifold specification.

    >>> print(rq.D1_excited(atom, splitting="fs"))
    (5, 1, 0.5, m_j='all')
    >>> print(rq.D1_excited(5, splitting="fs", expand=True))
    [(n=5, l=1, j=0.5, m_j=-0.5), (n=5, l=1, j=0.5, m_j=0.5)]
    >>> print(rq.D1_excited(atom, splitting="hfs"))
    (5, 1, 0.5, f='all', m_f='all')

    """
    if isinstance(n, str):

        # strip out isotope numbers, if any
        atom = re.sub(r'\d+', '', n)
        if atom in ground_n.keys():
            n = ground_n[atom]
        else:
            raise RydiquleError(f"For string value of n, must use one of {list(ground_n.keys())}")

    if not isinstance(n, int):
        raise RydiquleError(f"n must be int or str, but found type {type(n)}.")

    try:
        e_state = A_QState(n, 1, 1/2, *splitting_qnums[splitting])
    except KeyError:
        raise ValueError(f"Invalid splitting {splitting}")
    
    if expand:
        return expand_qnums([e_state])
    return e_state


def D2_excited(n: Union[int, str],
               splitting:Literal[None, "fs", "hfs"]=None,
               expand:bool=False) -> Union[A_QState, List[A_QState]]:
    """
    Retrieve `A_QState` of the excited state of the D2 line of an atom or principle quantum number.

    Optionally, include fine structure splitting or hyperfine splitting, which will include all 
    `m_j` or all `f` and `m_f` values respectively. By default, specifying splitting does not return
    a list of states, but rather the associated specification with `"all"` in the appropriate
    place. The `expand` keyword argument can be used to modify this behavior, returning a list.

    Parameters
    ----------
    n: int or str
        Either the string flag of the atom or the principle quantum number n of
        an atom. If string, must begin with ['H', 'Li', 'Na', 'K',
        'Rb', 'Cs'].
    splitting: {None, 'fs', 'hfs'}
        Type of splitting. Must be one of `None`, `"fs"`, or `"hfs"`, corresponding to the
        inclusion of `(n,l,j)` only, `m_j`, or both `f` and `m_f` respectively.
    expand: boolean, optional
        For states with splitting, whether to return them as a list of all states. If `False`,
        return is a single state specification with `"all"` for the appropriate quantum numbers.
        If `True`, a list of all individual states is returned.

    Raises
    ------
    RydiquleError
        If `n` is not a valid atom string or integer value
    ValueError
        If `splitting` is not one of {None, "fs", "hfs"}.

    Returns
    -------
    A_QState or list of A_QState:
        `A_QState` corresponding to the D2 excited state of the provided atom with the provided
        splitting, or list of `A_QState`s if `expand` is `True`.
    
    Examples
    --------
    The simplest use is to return the nlj quantum numbers for a particular atom's excited state of
    the D2 transtion. Principle quantum number and string atom flags can be used interchangeably.

    >>> atom = "Rb85"
    >>> print(rq.D2_excited(atom))
    (5, 1, 1.5)
    >>> print(rq.D2_excited(5))
    (5, 1, 1.5)

    This function also can return states with splitting, either as a list of states or as a
    manifold specification.

    >>> print(rq.D2_excited(atom, splitting="fs"))
    (5, 1, 1.5, m_j='all')
    >>> print(rq.D2_excited(5, splitting="fs", expand=True))
    [(n=5, l=1, j=1.5, m_j=-1.5), 
     (n=5, l=1, j=1.5, m_j=-0.5), 
     (n=5, l=1, j=1.5, m_j=0.5),
     (n=5, l=1, j=1.5, m_j=1.5)]
    >>> print(rq.D2_excited(atom, splitting="hfs"))
    (5, 1, 1.5, f='all', m_f='all')
    
    """
    if isinstance(n, str):

        # strip out isotope numbers, if any
        atom = re.sub(r'\d+', '', n)
        if atom in ground_n.keys():
            n = ground_n[atom]
        else:
            raise RydiquleError(f"For string value of n, must use one of {list(ground_n.keys())}")

    if not isinstance(n, int):
        raise RydiquleError(f"n must be int or str, but found type {type(n)}.")

    try:
        e_state = A_QState(n, 1, 3/2, *splitting_qnums[splitting])
    except KeyError:
        raise ValueError(f"Invalid splitting {splitting}")
    
    if expand:
        return expand_qnums([e_state])
    return e_state


def D1_states(n: Union[int, str],
              splitting:Literal[None, "fs", "hfs"]=None, 
              g_splitting:Literal[None, "fs", "hfs"]=None,
              e_splitting:Literal[None, "fs", "hfs"]=None,
              expand:bool=False
              ) -> List[Union[A_QState, List[A_QState]]]:
    """Return the ground and excited states for the D1 line of a rydberg atom.

    States are returned as a pair of `A_QStates` with the provided splitting according to
    the :func:`~.atom_utils.rydberg_ground` and :func:`~.atom_utils.D1_excited` functions with
    the provided splitting values passed through. When splitting is `None`, the `g_splitting`
    and `e_splitting` values are passed to `rydberg_ground` and `D1_excited` respectively.
    Otherwise, the value of `splitting` is passed to both and `g_splitting` and `e_splitting`
    are ignored.

    By default, specifying splitting does not return a list of all states, but rather the associated 
    specifications with `"all"` in the appropriate place. The `expand` keyword argument can be
    used to modify this behavior, returning a full list.

    Parameters
    ----------
    n : int or str
        Either the string flag of the atom or the principle quantum number n of
        an atom. If string, must begin with ['H', 'Li', 'Na', 'K', 'Rb', 'Cs'].
    splitting : None, &quot;fs&quot;, &quot;hfs, optional
        Type of splitting for both states. Must be one of `None`, `"fs"`, or `"hfs"`, 
        corresponding to the inclusion of `(n,l,j)` only, `m_j`, or both `f` and `m_f` 
        respectively.
    g_splitting : None, &quot;fs&quot;, &quot;hfs, optional
        Type of splitting for both states. Must be one of `None`, `"fs"`, or `"hfs"`, 
        corresponding to the inclusion of `(n,l,j)` only, `m_j`, or both `f` and `m_f` 
        respectively. Ignored if `splitting` is specified.
    e_splitting : None, &quot;fs&quot;, &quot;hfs, optional
        Type of splitting for both states. Must be one of `None`, `"fs"`, or `"hfs"`, 
        corresponding to the inclusion of `(n,l,j)` only, `m_j`, or both `f` and `m_f` 
        respectively. Ignored if `splitting` is specified.

    Returns
    -------
    list of A_QState
        Ground and D1 excited state specifications of the provided atom or pricipal quantum number.

    
    Examples
    --------
    The basic use of this function is to return the A_QStates associated with the states of the D1
    transtition of a particular Rydberg atom. String flags and principle quantum numbers can be used
    interchangeably.

    >>> atom = "Rb85"
    >>> print(rq.D1_states(atom))
    [(n=5, l=0, j=0.5), (n=5, l=1, j=0.5)]
    >>> print(rq.D1_states(5))
    [(n=5, l=0, j=0.5), (n=5, l=1, j=0.5)]

    Furthermore, splitting can be specified either for each state individually, or just for one of
    the states using the optional `splitting`, `g_splitting`, or `e_splitting` argument.
    
    >>> print(rq.D1_states(5, splitting="fs"))
    [(n=5, l=0, j=0.5, m_j='all'), (n=5, l=1, j=0.5, m_j='all')]
    >>> print(rq.D1_states(5, splitting="fs", expand=True))
    [(n=5, l=0, j=0.5, m_j=-0.5), (n=5, l=0, j=0.5, m_j=0.5), (n=5, l=1, j=0.5, m_j=-0.5), (n=5, l=1, j=0.5, m_j=0.5)]
    >>> print(rq.D1_states(5, g_splitting="fs"))
    [(n=5, l=0, j=0.5, m_j='all'), (n=5, l=1, j=0.5)]

    """
    
    if splitting is not None:
        g_splitting = splitting
        e_splitting = splitting
    
    g_states = ground_state(n, splitting=g_splitting, expand=expand)
    e_states = D1_excited(n, splitting=e_splitting, expand=expand)

    if expand:
        # fully expanded list of all states
        assert isinstance(g_states, list) and isinstance(e_states, list)
        return [*g_states, *e_states]
    else:
        # 2-element list of specs
        return [g_states, e_states]


def D2_states(n: Union[int, str],
              splitting:Literal[None, "fs", "hfs"]=None, 
              g_splitting:Literal[None, "fs", "hfs"]=None,
              e_splitting:Literal[None, "fs", "hfs"]=None,
              expand: bool=False) -> List[Union[A_QState, List[A_QState]]]:
    """Return the ground and excited states for the D1 line of a rydberg atom.

    States are returned as a pair of `A_QStates` with the provided splitting according to
    the :func:`~.atom_utils.rydberg_ground` and :func:`~.atom_utils.D2_excited` functions with
    the provided splitting values passed through. When splitting is `None`, the `g_splitting`
    and `e_splitting` values are passed to `rydberg_ground` and `D2_excited` respectively.
    Otherwise, the value of `splitting` is passed to both and `g_splitting` and `e_splitting`
    are ignored.

    By default, specifying splitting does not return a list of all states, but rather the associated 
    specifications with `"all"` in the appropriate place. The `expand` keyword argument can be
    used to modify this behavior, returning a full list.

    Parameters
    ----------
    n : int or str
        Either the string flag of the atom or the principle quantum number n of
        an atom. If string, must begin with ['H', 'Li', 'Na', 'K', 'Rb', 'Cs'].
    splitting : None, &quot;fs&quot;, &quot;hfs, optional
        Type of splitting for both states. Must be one of `None`, `"fs"`, or `"hfs"`, 
        corresponding to the inclusion of `(n,l,j)` only, `m_j`, or both `f` and `m_f` 
        respectively.
    g_splitting : None, &quot;fs&quot;, &quot;hfs, optional
        Type of splitting for both states. Must be one of `None`, `"fs"`, or `"hfs"`, 
        corresponding to the inclusion of `(n,l,j)` only, `m_j`, or both `f` and `m_f` 
        respectively. Ignored if `splitting` is specified.
    e_splitting : None, &quot;fs&quot;, &quot;hfs, optional
        Type of splitting for both states. Must be one of `None`, `"fs"`, or `"hfs"`, 
        corresponding to the inclusion of `(n,l,j)` only, `m_j`, or both `f` and `m_f` 
        respectively. Ignored if `splitting` is specified.

    Returns
    -------
    list of A_QState
        Ground and D2 excited state specifications of the provided atom or pricipal quantum number.

    Examples
    --------
    The basic use of this function is to return the A_QStates associated with the states of the D2
    transtition of a particular Rydberg atom. String flags and principle quantum numbers can be used
    interchangeably.

    >>> atom = "Rb85"
    >>> print(rq.D2_states(atom))
    [(n=5, l=0, j=0.5), (n=5, l=1, j=1.5)]
    >>> print(rq.D2_states(5))
    [(n=5, l=0, j=0.5), (n=5, l=1, j=1.5)]

    Furthermore, splitting can be specified either for each state individually, or just for one of
    the states using the optional `splitting`, `g_splitting`, or `e_splitting` argument.
    
    >>> print(rq.D1_states(5, splitting="fs"))
    [(n=5, l=0, j=0.5, m_j='all'), (n=5, l=1, j=0.5, m_j='all')]
    >>> print(rq.D2_states(5, splitting="fs", expand=True))
    [(n=5, l=0, j=0.5, m_j=-0.5), 
     (n=5, l=0, j=0.5, m_j=0.5), 
     (n=5, l=1, j=1.5, m_j=-1.5), 
     (n=5, l=1, j=1.5, m_j=-0.5), 
     (n=5, l=1, j=1.5, m_j=0.5), 
     (n=5, l=1, j=1.5, m_j=1.5)]
    >>> print(rq.D1_states(5, g_splitting="fs"))
    [(n=5, l=0, j=0.5, m_j='all'), (n=5, l=1, j=0.5)]

    """
    
    if splitting is not None:
        g_splitting = splitting
        e_splitting = splitting
    
    g_states = ground_state(n, splitting=g_splitting, expand=expand)
    e_states = D2_excited(n, splitting=e_splitting, expand=expand)

    if expand:
        # fully expanded list of all states
        assert isinstance(g_states, list) and isinstance(e_states, list)
        return [*g_states, *e_states]
    else:
        # 2-element list of specs
        return [g_states, e_states]


def calc_kappa(omega: float, dipole_moment: float, density: float) -> float:
    """
    Calculates the kappa constant needed for observable calculations.

    The value is computed with the following formula Eq. 5 of
    Meyer et. al. PRA 104, 043103 (2021)

    .. math::

        \\kappa = \\frac{\\omega n \\mu^2}{2c \\epsilon_0 \\hbar}

    Where :math:`\\omega` is the probing frequency, :math:`\\mu` is the dipole moment,
    :math:`n` is atomic cloud density, :math:`c` is the speed of light, :math:`\\epsilon_0`
    is the dielectric constant, and :math:`\\hbar` is the reduced Plank constant.

    Parameters
    ----------
    omega: float
        Atomic transition frequency, in rad/s
    dipole_moment: float
        Dipole moment of the atomic transition, in C*m
    density: float
        The atomic number density, in m^(-3)

    Returns
    -------
    float
        The value of kappa, in (rad/s)/m
    """

    kappa = (omega*density*dipole_moment**2)/(2.0*c*epsilon_0*hbar)

    return kappa


def calc_eta(omega: float, dipole_moment:float, beam_area: float) -> float:
    """
    Calculates the eta constant needed from some experiment calculations

    The value is computed with the following formula Eq. 7 of
    Meyer et. al. PRA 104, 043103 (2021)

    .. math::

        \\eta = \\sqrt{\\frac{\\omega \\mu^2}{2 c \\epsilon_0 \\hbar A}}

    Where :math:`\\omega` is the probing frequency, :math:`\\mu` is the dipole moment,
    :math:`A` is the beam area, :math:`c` is the speed of light, :math:`\\epsilon_0`
    is the dielectric constant, and :math:`\\hbar` is the reduced Plank constant.

    Parameters
    ----------
    omega: float
        The atomic transition frequency, in rad/s
    dipole_moment: float
        The atomic transition dipole moment, in C*m
    beam_area: float
        The cross-sectional area of the beam, in m^2

    Returns
    -------
    float
        The value of eta, in root(Hz)
    """

    eta = np.sqrt((omega*dipole_moment**2)/(2.0*c*epsilon_0*hbar*beam_area))

    return eta


def expand_qnums(qstates: List[A_QState], I: Optional[float] = None,
                 ) -> List[A_QState]:
    """Expand all list-like A_QStates in a list.

    List-like quantum numbers are defined either with a list of quantum numbers or the string
    "all". In the "all" case, that quantum number will be expanded into all physically allowed
    values of that quantum number given the preceeeding numbers. 

    Iterates through the list, expanding each A_QState specifcation into a list of all states
    matching that specification. For each state specification in the list, quantum numbers are
    expanded from left to right. The final list of A_QStates will respect the ordering of the
    intial states by ordering the states corresponding to each specification by 
    n, l, j, m_j, f, and finally m_f

    Parameters
    ----------
    qstates : list of A_QState
        List of atomic quantum states specifications to be expanded.
    I : Union[float,None], optional
        Nuclear spin for the isotope of the atom. Used to calculate the f quantum number
        when relevant, by default None

    Returns
    -------
    list of A_QState
        List of all atomic states corresponding to all the specifications in the given list.

    Notes
    -----
    ..note::
        While this funcion can expand arbitrary states, it should be noted that the resulting
        lists of states can be quite long. If they are to be used as the states of a 
        :class:`~.Cell`, these long state lists can dramatically increase computation time, and 
        it is often worth ensuring that tracking hyperfine states individually is absolutely
        necessary. 


    Examples
    --------
    A basic piece of functionality for this function is as a shorthand for allstates in a
    given manifold.

    >>> D1_ground = A_QState(5,0,0.5, f="all")
    >>> D1_excited = A_QState(5,0,0.5,f="all")
    >>> #manifolds for rubidium 87 (I=3/2)
    >>> print(rq.expand_qnums([D1_ground], I=3/2))
    [(n=5, l=0, j=0.5, f=1.0), (n=5, l=0, j=0.5, f=2.0)]
    >>> print(rq.expand_qnums([D1_excited], I=3/2))
    [(n=5, l=0, j=0.5, f=1.0), (n=5, l=0, j=0.5, f=2.0)]
    >>> #manifolds for rubidium 85 (I=5/2)
    >>> print(rq.expand_qnums([D1_ground], I=5/2))
    [(n=5, l=0, j=0.5, f=2.0), (n=5, l=0, j=0.5, f=3.0)]
    >>> print(rq.expand_qnums([D1_excited], I=5/2))
    [(n=5, l=0, j=0.5, f=2.0), (n=5, l=0, j=0.5, f=3.0)]

    Note that while this function is capable of getting large numbers of states, the resulting
    lists can be quite cumbersome, and be substantially slower if used in a calculation, especially
    for high angular momentum states.

    >>> state = A_QState(7, 2, 2.5, f="all", m_f="all")
    >>> states_all = rq.expand_qnums([state], I=7/2)
    >>> print(len(states_all))
    48

    """
    has_len = [any([hasattr(qn, "__len__") for qn in state])
               for state in qstates]
    
    if not any(has_len):
        return qstates
    
    else:
        expanded_qnums = [
            expand_single_qnum(state, I=I)
            for state in qstates
        ]
        list_qnums = sum(expanded_qnums, start=[])
    
        return(expand_qnums(list_qnums, I=I))


def validate_qnums(qstate:A_QState, I: Optional[float]=None):
    """Validate that the provided named_tuple is a valid rydberg atomic state

    Parameters
    ----------
    qstate : A_QState
        Named tuple to check, should have fields `("n","l","j","m_j","f","m_f")`
    I : Union[None,float], optional
        Nuclear spin of the rydberg atom of which this is a state. If `None`, all f
        values are invalid automaticaly. Defaults to `None`

    Raises
    ------
    ValueError
        If the tuple representing the state does not have 6 elements
    AssertionError
        If the states of the state are not physically allowed
    """
    #confirm qstate is the correct type of state
    try:    
        (n,l,j,m_j,f,m_f) = (
            qstate.n,
            qstate.l,
            qstate.j,
            qstate.m_j,
            qstate.f,
            qstate.m_f
        )
        assert len(qstate) == 6

    except (AttributeError, ValueError, AssertionError):
        raise ValueError("Atomic states must be represented with a rq.A_QState namedtuple")

    none_qnums = tuple(i is not None for i in qstate[3:])
    if none_qnums not in STATE_TYPES:
        raise ValueError(f"State {qstate} is not a valid combination of quantum numbers")

    #validate (n,l) int, j half int
    assert int(n)==n, f"invalid n quantum number {n}."
    assert (int(l)==l) and (l < n), f"invalid l quantum number {l}."
    assert j==l+1/2 or j==np.abs(l-1/2), f"invalid j quantum number {j}"

    #test m_j, f, m_f are allowed values
    if m_j is not None:
        valid_mj = get_valid_mj(qstate, I=I)
        assert m_j in valid_mj, f"m_j must be one of {valid_mj}"
    if f is not None:
        valid_f = get_valid_f(qstate, I=I)
        assert f in valid_f, f"f must be one of {valid_f}"
    if m_f is not None:
        valid_mf = get_valid_mf(qstate, I=I)
        assert m_f in valid_mf, f"m_f must be one of {valid_mf}"


def expand_single_qnum(qstate: A_QState, I: Optional[float] = None, wildcard: str = "all"
                       ) -> List[A_QState]:
    """Generates a list of all valid states given a particular quantum number to be expanded.

    For a given `A_Qstate` spec with one or more tuple elements specified as either a list or
    the "all" string, returns a list of all valid state specifcations matching that state
    specification with the first list or string element only expanded. If multiple elemens of the
    statespec are specified with a list or string, only the first one is expanded. This function
    is intended as a helper function for a single quantum number, and is not designed to be
    used at the top-level

    The the case that the element to be expanded is a list, the list returned will have a single
    state specification corresponding to each element of that list, and allowed quantum number
    rules will not be enforced. In the case that the element to be expaned is the "all" string, 
    all valid values of that particular quantum number will be used. Note
    that only the `m_j`, `f`, and `m_f` quantum numbers can be expanded in this way.

    Parameters
    ----------
    qstate : A_QState
        NamedTuple with fields `(n, l, j, m_j, f, m_f)` representng the quantum numbers of
        the state. Each, element must be either a float, list of floats, or the "all" string. 
        Only `m_j`, `f`, and `m_f` may be specified with a "all". 
    I : float, optional
        The nuclear spin of the rydberg atom. Only used for calculations of valid `f` quantum
        numbers, defaults to 0.0

    Returns
    -------
    List of A_QState
        List of all possible quantum states matching the provided specification. Only the
        first list-like quantum number will be expanded.

    Raises
    ------
    RydiquleError
        If there is a string specification besides "all" in the provided state
    
    Examples
    --------
    A simple m_j expansion of the D1 states for Rubidium

    >>> D1_g = A_QState(5,0,0.5, m_j="all")
    >>> D1_e = A_QState(5,1,0.5, m_j="all")
    >>> print(rq.atom_utils.expand_single_qnum(D1_g))
    [(n=5, l=0, j=0.5, m_j=-0.5), (n=5, l=0, j=0.5, m_j=0.5)]
    >>> print(rq.atom_utils.expand_single_qnum(D1_e))
    [(n=5, l=1, j=0.5, m_j=-0.5), (n=5, l=1, j=0.5, m_j=0.5)]

    We can also expand into nuclear spin-coupled (f) states. Note that in this case we must
    provide the nuclear spin I to use this function on its own (in this case I=7/2 for Rb87)

    >>> D1_g = A_QState(5,0,0.5, f="all")
    >>> D1_e = A_QState(5,1,0.5, f="all")
    >>> print(rq.atom_utils.expand_single_qnum(D1_g, I=7/2))
    [(n=5, l=0, j=0.5, f=3.0), (n=5, l=0, j=0.5, f=4.0)]
    >>> print(rq.atom_utils.expand_single_qnum(D1_e, I=7/2))
    [(n=5, l=1, j=0.5, f=3.0), (n=5, l=1, j=0.5, f=4.0)]

    While we can provide the "all" flag to expand to all states, we may want to use a specific
    subset of states if, for example, selection rules limit the states at play. In this case,
    one can pass a list for a given quantum number.

    >>> state = A_QState(5, 2 ,2.5, m_j=[-2.5, -1.5, -0.5])
    >>> print(rq.atom_utils.expand_single_qnum(state))
    [(n=5, l=2, j=2.5, m_j=-2.5), (n=5, l=2, j=2.5, m_j=-1.5), (n=5, l=2, j=2.5, m_j=-0.5)]

    """

    list_idxs = [i for i,qn in enumerate(qstate)
                 if isinstance(qn, (list, str))]
    if len(list_idxs) == 0:
        return [qstate]
    
    idx = list_idxs[0]

    if isinstance(qstate[idx], str):
        if not qstate[idx] == "all":
            raise RydiquleError("String must be 'all' to specify all possible quantum numbers.")
        
        qstate = A_QState(*(qn if i != idx
                            else valid_qnum_fns[i](qstate,I=I)
                            for i, qn in enumerate(qstate)
                            ))
        
    return expand_statespec(qstate) 


def match_A_QState(qstate: A_QState, compare_list=[], I: Optional[float] =None
                   ) -> List[A_QState]:
    """Function to return all states in a list matching the provided pattern.

    States are considered a match for `qstate` if they are an element of the list returned
    by calling the :func:`~.expand_qnums` on `qstate`. 

    Parameters
    ----------
    qstate : A_QState
        The state against which elements of the list are compared.
    compare_list : list, optional
        List of states to test. Any matching the pattern provided by `qstate` will be in the
        returned list, by default []
    I : float, optional
        Nuclear spin I of the atom containing the provided states, by default None

    Returns
    -------
    List of A_QState
        List of all the elements matching the pattern defined by `qstate`
    """
    
    all_qstates = expand_qnums([qstate], I=I)
    
    return [state for state in all_qstates if state in compare_list]

#functions to compute valid quantum numbers for rydberg atoms. While not
#underscored, these functions are not strictly designed as user-facing

def get_valid_j(state: A_QState, I:Optional[float]=None) -> List[float]:
    """Return the valid values of j for given other quantum numbers.

    For a given quantum state with principal and orbital quantum numbers 
    :math:`(n,l)`, the valid values of j are given by 
    
    .. math:: j = |l - \\frac{1}{2}|, l + \\frac{1}{2}

    Note that if both values are the same, a list of length 1 is returned. 
    """
    L_qnum = state[1]
    if not isinstance(L_qnum, (int, float)):
        raise RydiquleError(f"Invalid J qunatum number type {type(L_qnum)}.")
    return list(set(L_qnum + s for s in [-.5,.5]))


def get_valid_mj(state: A_QState, I:Optional[float]=None) -> List[float]:
    """Return the valid values of m_J for given other quantum numbers.

    For a given quantum state with principl, orbital, and total quantum numbers 
    :math:`(n,L,J)`, the valid values of m_J are given by 
    
    .. math:: m_J = -J, -J+1, -J+2, ... , J-2, J-1, J
    """
    J_qnum = state[2]
    if not isinstance(J_qnum, (int, float)):
        raise RydiquleError(f"Invalid J qunatum number type {type(J_qnum)}.")
    return np.arange(-1*J_qnum,J_qnum + 1).tolist()


def get_valid_f(state: A_QState, I: Optional[float]=None) -> List[float]:
    """Return the valid values of f for given other quantum numbers.

    For a given quantum state with principal, orbital, and spin-orbit quantum numbers 
    :math:`(n,L,J)` and nuclear quantum number :math:`I`, the valid values of m_f are given by
    
    .. math:: f = |I-J|, |I-J|+1, ..., I+J
        
    """
    J_qnum=state[2]
    if not isinstance(J_qnum, (int, float)) or not isinstance(I, (int, float)):
        raise ValueError(f"Invalid I,J qunatum number types {(type(I),type(J_qnum))}.")
    return np.arange(np.abs(J_qnum - I), J_qnum + I + 1).tolist()


def get_valid_mf(state: A_QState, I: Optional[float]=None) -> List[float]:
    """Return the valid values of m_f for given other quantum numbers.

    For a given quantum state with principal, orbital, and total quantum numbers 
    :math:`(n,L,J,f)`, the valid values of m_f are given by 
    
    .. math:: m_f = -f, -f+1, -f+2, ... , f-2, f-1, f
    """
    f_qnum = state[4]
    if not isinstance(f_qnum, (float, int)):
        raise RydiquleError(f"Invalid f qunatum number type {type(f_qnum)}.")
    return np.arange(-1*f_qnum,f_qnum + 1).tolist()


valid_qnum_fns: Dict[int, Callable] = {
    2: get_valid_j,
    3: get_valid_mj,
    4: get_valid_f,
    5: get_valid_mf
}
