"""
Utilities for interacting with atomic parameters and ARC.
"""

import arc.alkali_atom_data as arc_atoms
from scipy.constants import epsilon_0, hbar, c
import numpy as np
import re

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
    "Li": 2,
    "Na": 3,
    "K": 4,
    "Rb": 5,
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
        an atom. If string, must begin with ['H', 'Li', 'Na', 'K',
        'Rb', 'Cs'].

    Returns
    -------
    list
        Quantum numbers [n, l, j, m] of the atoms ground state.
    list
        Quantum numbers [n, l, j, m] of the first excited state corresponding
        to the D1 line of the Rydberg atom.
    """

    if isinstance(n, str):

        # strip out isotope numbers, if any
        n = re.sub('\d+', '', n)
        if n in ground_n.keys():
            n = ground_n[n]
        else:
            raise ValueError(f"For string value of n, must use one of {list(ground_n.keys())}")

    if not isinstance(n, int):
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
        an atom. If string, must begin with ['H', 'Li', 'Na', 'K',
        'Rb', 'Cs'].

    Returns
    -------
    list
        Quantum numbers [n, l, j, m] of the atoms ground state.
    list
        Quantum numbers [n, l, j, m] of the first excited state corresponding
        to the D2 line of the Rydberg atom.
    """

    if isinstance(n, str):

        # strip out isotope numbers, if any
        n = re.sub('\d+', '', n)
        if n in ground_n.keys():
            n = ground_n[n]
        else:
            raise ValueError(f"For string value of n, must use one of {list(ground_n.keys())}")

    if not isinstance(n, int):
        raise ValueError(f"n must be int or str, but found type {type(n)}.")

    return [n, 0, 1/2, 1/2], [n, 1, 3/2, 1/2]


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
