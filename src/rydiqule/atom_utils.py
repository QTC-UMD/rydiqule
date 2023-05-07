"""
Utilities for interacting with atomic parameters and ARC.
"""

import arc.alkali_atom_data as arc_atoms

from typing import Union

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
    "Li": 2, "Li6": 2, "Li7": 2,
    "Na": 3,
    "K": 4, "K39": 4, "K40": 4, "K41": 4,
    "Rb": 5, "Rb85": 5, "Rb87": 5,
    "Cs": 6
}


def D1_states(n: Union[int, str]):
    """
    Retrieve the quantum numbers for the states corresponding to
    the D1 line for a given Rydberg atom or principle quantum number.

    Parameters
    ----------
    n: int or str
        Either the string flag of the atom or the principle quantum number n of
        an atom. If string, must be one of ['H', 'Li', 'Li6', 'Li7', 'Na', 'K', 'K39', 'K40',
        'K41', 'Rb', 'Rb85', 'Rb87', 'Cs'].

    Returns
    -------
    list
        Quantum numbers [n, l, j, m] of the atoms ground state.
    list
        Quantum numbers [n, l, j, m] of the first excited state corresponding
        to the D1 line of the Rydberg atom.
    """

    if type(n) is str:

        if n in ground_n.keys():
            n = ground_n[n]
        else:
            raise ValueError(f"For string value of n, must use one of {list(ground_n.keys())}")

    if type(n) != int:
        raise ValueError(f"n must be int or str, but found type {type(n)}.")

    return [n, 0, 1/2, 1/2], [n, 1, 1/2, 1/2]


def D2_states(n: Union[int, str]):
    """
    Retrieve the quantum numbers for the states corresponding to the D2 line for a
    given Rydberg atom or principle quantum number.

    Parameters
    ----------
    n: int or str
        Either the string flag of the atom or the principle quantum number n of
        an atom. If string, must be one of ['H', 'Li', 'Li6', 'Li7', 'Na', 'K', 'K39', 'K40',
        'K41', 'Rb', 'Rb85', 'Rb87', 'Cs'].

    Returns
    -------
    list
        Quantum numbers [n, l, j, m] of the atoms ground state.
    list
        Quantum numbers [n, l, j, m] of the first excited state corresponding
        to the D2 line of the Rydberg atom.
    """

    if type(n) is str:

        if n in ground_n.keys():
            n = ground_n[n]
        else:
            raise ValueError(f"For string value of n, must use one of {list(ground_n.keys())}")

    if type(n) != int:
        raise ValueError(f"n must be int or str, but found type {type(n)}.")

    return [n, 0, 1/2, 1/2], [n, 1, 3/2, 1/2]
