"""
Helper methods for interfacing with ARC atom classes
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Dict, Callable, Tuple

from arc import C_e, C_c, C_h
from scipy.constants import pi, hbar, epsilon_0, physical_constants
import numpy as np

from math import sqrt

from .exceptions import AtomError
from .atom_utils import A_QState

if TYPE_CHECKING:
    import arc

C_a_0 = physical_constants["Bohr radius"][0]


class RQ_AlkaliAtom(object):

    def __init__(self, arc_atom: arc.alkali_atom_functions.AlkaliAtom):
        """Rydiqule's wrapper class around ARC's Alkai atom classes.

        Designed predominantly for internal use, seldom needs to be accessed directly.

        Parameters
        ----------
        arc_atom: :external+arc:class:`~arc.alkali_atom_functions.AlkaliAtom`
            ARC atom to use for calculations.
            Stored internally in the :attr:`atom` attribute.
        """

        self.arc_atom = arc_atom
        "ARC atom with which to perfom calculations."

        self._arc_dipole_functions: Dict[Tuple[str, str], Callable] = {
            ("HFS", "FS"): self.arc_atom.getDipoleMatrixElementHFStoFS,
            ("FS", "HFS"): self._getDipoleMatrixElementFStoHFS,
            ("HFS", "HFS"): self.arc_atom.getDipoleMatrixElementHFS,
            ("FS", "FS"): self.arc_atom.getDipoleMatrixElement,
            ("NLJ", "NLJ"): self._get_nlj_dipole
        }

    def get_dipole_matrix_element(self, state1: A_QState, state2: A_QState,
                                  q: Literal[-1, 0, 1], s: float = 0.5) -> float:
        """Get dipole matrix element :math:`\\langle s1|e\\mathbf{r}|s2\\rangle` in units of :math:`a_0 e`

        If states 1 and 2 are sublevels, either FS or HFS, appropriate ARC function is used.
        If states 1 and 2 are NLJ, a simple average of the magnitudes of the dipole-allowed moments
        between mJ1 and mJ2 is returned.

        Cannot calculate dipole matrix elements between states with NLJ specication and those with
        either FS or HS splitting.

        ARC functions used are:

        - :external+arc:meth:`~arc.alkali_atom_functions.AlkaliAtom.getDipoleMatrixElement`
        - :external+arc:meth:`~arc.alkali_atom_functions.AlkaliAtom.getDipoleMatrixElementHFS`
        - :external+arc:meth:`~arc.alkali_atom_functions.AlkaliAtom.getDipoleMatrixElementHFStoFS`

        Parameters
        ----------
        state1: A_QState
            `A_QState` namedtuple of quantum numbers for the quantum state :math:`s1`.
        state2: A_QState
            `A_QState` namedtuple of quantum numbers for the quantum state :math:`s2`.
        q: int
            Polarization of coupling field in spherical basis (+1, 0, -1),
            corresponding to
            :math:`\\sigma^+`, :math:`\\pi`, or :math:`\\sigma^-`.
        s: float, optional
            total spin angular momentum of the state. Default is 0.5 for Alkali atoms.

        Returns
        -------
        float
            Dipole moment of the transition in atomic units (:math:`a_0 e`).
            Will be 0 if the transition is not dipole-allowed.

        Raises
        ------
        AtomError
            If the two states to be coupled are in one each of the NLJ and FS/HFS defintions.

        Examples
        --------
        >>> import arc
        >>> g_nlj = rq.A_QState(5, 0, 0.5)
        >>> g_fs = rq.A_QState(5, 0, 0.5, m_j=-0.5)
        >>> e_nlj = rq.A_QState(5, 1, 0.5)
        >>> e_hfs = rq.A_QState(5, 1, 0.5,f=2, m_f=0)
        >>> arc_atom = arc.alkali_atom_data.Rubidium85()
        >>> my_atom = rq.RQ_AlkaliAtom(arc_atom)
        >>> print(my_atom.get_dipole_matrix_element(g_nlj, e_nlj, q=0))
        1.7277475900721146
        >>> print(my_atom.get_dipole_matrix_element(g_fs, e_hfs, q=0))
        1.2217020371187075
        >>> print(my_atom.get_dipole_matrix_element(g_nlj, e_hfs, q=0)) # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        rydiqule.exceptions.AtomError: Invalid transition type for dipole calculation.
        """
        if (state1.stype, state2.stype) not in self._arc_dipole_functions.keys():
            raise AtomError("Invalid transition type for dipole calculation. "
                            + f"Allowed types are {list(self._arc_dipole_functions.keys())}")

        return self._arc_dipole_functions[(state1.stype, state2.stype)](*state1.qnums, *state2.qnums, q, s)


    def get_transition_frequency(self, state1: A_QState, state2: A_QState,
                                 s1: float = 0.5, s2: float = 0.5) -> float:
        """Returns the transition frequency (energy difference) between two states, in Hz.

        Uses :meth:`get_state_energy` on both states to determine the energy difference.

        Parameters
        ----------
        state1: A_QState
            `A_QState` namedtuple of quantum numbers for the first quantum state.
        state2: A_QState
            `A_QState` namedtuple of quantum numbers for the second quantum state.
        s1: float, optional
            spin of the initial state. Default is 0.5 for Alkali atom.
        s2: float, optional
            spin of the final state. Default is 0.5 for Alkali atom.

        Returns
        -------
        float
            Transition frequency between the two states, in Hz.
            If negative, state1 has higher energy than state2.
        
        Examples
        --------
        >>> import arc
        >>> g_nlj = rq.A_QState(5, 0, 0.5)
        >>> g_fs = rq.A_QState(5, 0, 0.5, m_j=-0.5)
        >>> e_nlj = rq.A_QState(5, 1, 0.5)
        >>> e_hfs = rq.A_QState(5, 1, 0.5,f=2, m_f=0)
        >>> arc_atom = arc.alkali_atom_data.Rubidium85()
        >>> my_atom = rq.RQ_AlkaliAtom(arc_atom)
        >>> print(my_atom.get_transition_frequency(g_nlj, e_nlj))# doctest: +SKIP
        377107433259213.6
        >>> print(my_atom.get_transition_frequency(g_fs, e_hfs))# doctest: +SKIP
        377107222336963.6
        >>> print(my_atom.get_transition_frequency(g_nlj, e_hfs))# doctest: +SKIP
        377107222336963.6

        """

        e1 = self.get_state_energy(state1, s1)
        e2 = self.get_state_energy(state2, s2)

        return e2 - e1


    def get_transition_wavelength(self, state1: A_QState, state2: A_QState,
                                  s1: float = 0.5, s2: float = 0.5) -> float:
        """Returns the transition wavelength between two states, in m.
        
        Uses :external+arc:meth:`~arc.alkali_atom_functions.AlkaliAtom.getTransitionWavelength`.

        Parameters
        ----------
        state1: A_QState
            `A_QState` namedtuple of quantum numbers for the first quantum state.
        state2: A_QState
            `A_QState` namedtuple of quantum numbers for the first quantum state.
        s1: float, optional
            spin of the initial state. Default is 0.5 for Alkali atom.
        s2: float, optional
            spin of the final state. Default is 0.5 for Alkali atom.

        Returns
        -------
        float
            Transition wavelength between the two states, in m.
            If negative, state2 has higher energy than state1.

        Examples
        --------
        >>> import arc
        >>> g_nlj = rq.A_QState(5, 0, 0.5)
        >>> g_fs = rq.A_QState(5, 0, 0.5, m_j=-0.5)
        >>> e_nlj = rq.A_QState(5, 1, 0.5)
        >>> e_hfs = rq.A_QState(5, 1, 0.5,f=2, m_f=0)
        >>> arc_atom = arc.alkali_atom_data.Rubidium85()
        >>> my_atom = rq.RQ_AlkaliAtom(arc_atom)
        >>> print(my_atom.get_transition_wavelength(g_nlj, e_nlj))
        7.949789146530309e-07
        >>> print(my_atom.get_transition_wavelength(g_fs, e_hfs))
        7.949789146530309e-07
        >>> print(my_atom.get_transition_wavelength(g_nlj, e_hfs))
        7.949789146530309e-07

        """
        return self.arc_atom.getTransitionWavelength(*state1.qnums[:3], *state2.qnums[:3],
                                                     s1, s2)


    def get_state_energy(self, state: A_QState, s: float = 0.5) -> float:
        """Returns the energy of the level relative to the ionisation level in Hz.
        
        If `state` is in the fine basis or NLJ, energies are relative to the center of gravity
        of the hyperfine split states. If `state` is in the hyperfine basis,
        hyperfine shifts are applied.

        Uses ARC's :external+arc:meth:`~arc.alkali_atom_functions.AlkaliAtom.getEnergy`
        to get the energy of the fine structure state.
        Hyperfine shifts are applied using ARC's
        :external+arc:meth:`~arc.alkali_atom_functions.AlkaliAtom.getHFSCoefficients` and
        :external+arc:meth:`~arc.alkali_atom_functions.AlkaliAtom.getHFSEnergyShift`
        methods.


        Parameters
        ----------
        state: A_QState
            `A_QState` namedtuple of quantum numbers of the quantum state for which energy 
            will be calculated.
        s: float
            Total spin of the state. Default is 0.5 for Alkali atom.

        Returns
        -------
        float
            Energy of state relative to the ionisation level in Hz.

        Examples
        --------
        >>> from rydiqule import A_QState
        >>> import arc
        >>> g_nlj = A_QState(5, 0, 0.5)
        >>> g_fs = A_QState(5, 0, 0.5, m_j=-0.5)
        >>> e_hfs = A_QState(5, 1, 0.5,f=2, m_f=0)
        >>> arc_atom = arc.alkali_atom_data.Rubidium85()
        >>> my_atom = rq.RQ_AlkaliAtom(arc_atom)
        >>> print(my_atom.get_state_energy(g_nlj)/1e9) #GHz
        -1010024.7
        >>> print(my_atom.get_state_energy(g_fs)/1e9) #GHz
        -1010024.7
        >>> print(my_atom.get_state_energy(e_hfs)/1e9) #GHz
        -632917.5
        """
        base_energy = self.arc_atom.getEnergy(*state[:3], s)*C_e/C_h  # in Hz

        if state.stype == "HFS":
            A, B = self.arc_atom.getHFSCoefficients(*state[:3])
            hfs = self.arc_atom.getHFSEnergyShift(state[2], state[4], A, B) # in Hz
        else:
            hfs = 0.0

        return base_energy + hfs


    def gaussian_center_field(self, laserPower: float, laserWaist: float) -> float:
        """Returns the electric field for the center of a TEM00 gaussian spatial mode

        This calculates the peak intensity of the gaussian mode,
        and uses a plane wave assumption to get the electric field amplitude.
        
        Parameters
        ----------
        laserPower: float
            laser power in Watts
        laserWaist: float
            laser :math:`1/e^2` waist (radius) in meters

        Returns
        -------
        float
            Peak electric field for a gaussian spatial mode, in V/m

        """

        maxIntensity = 2 * laserPower / (pi * laserWaist**2)
        electricField = sqrt(2.0 * maxIntensity / (C_c * epsilon_0))
        return electricField


    def get_rabi_frequency(self, state1: A_QState, state2: A_QState,
                           q: Literal[-1, 0, 1], laserPower: float, laserWaist: float,
                           s: float = 0.5) -> float:
        """Returns the Rabi frequency for resonantly driven atom in center of a TEM00 mode of a field.

        The field is calculated using :meth:`gaussian_center_field`.
        It then calls :meth:`get_rabi_frequency2` to get the rabi frequency.
        
        Parameters
        ----------
        state1: A_QState
            NamedTuple of quantum numbers for state driving from
        state2: A_QState
            NamedTuple of quantum numbers for state driving to
        q: int
            laser polarization in spherical basis (-1,0,1) corresponding to
            :math:`\\sigma^-`, :math:`\\pi`, and :math:`\\sigma^+`
        laserPower: float
            laser power in Watts
        laserWaist: float
            laser :math:`1/e^2` waist (radius) in meters
        s: float, optional
            total spin angular momentum of the states. By default 0.5 for Alkali atoms.

        Returns
        -------
        rabi_frequency: float
            Rabi frequency in rad/s. To get Hz, divide by :math:`2\\pi`
        
        Examples
        --------
        >>> from rydiqule import A_QState
        >>> import arc
        >>> g_nlj = A_QState(5, 0, 0.5)
        >>> g_fs = A_QState(5, 0, 0.5, m_j=-0.5)
        >>> e_nlj = A_QState(5, 1, 0.5)
        >>> e_hfs = A_QState(5, 1, 0.5,f=2, m_f=0)
        >>> arc_atom = arc.alkali_atom_data.Rubidium85()
        >>> my_atom = rq.RQ_AlkaliAtom(arc_atom)
        >>> print(my_atom.get_rabi_frequency(g_nlj, e_nlj, q=0, laserPower=1, laserWaist=0.01)/1e6) #MHz
        304.22
        >>> print(my_atom.get_rabi_frequency(g_fs, e_hfs, q=0, laserPower=1, laserWaist=0.01)/1e6) #MHz
        215.11
        >>> print(my_atom.get_rabi_frequency(g_nlj, e_hfs, q=0, laserPower=1, laserWaist=0.01)) # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
          ...
        rydiqule.exceptions.AtomError: Invalid transition type for dipole calculation.

        """

        electricField = self.gaussian_center_field(laserPower, laserWaist)
        
        return self.get_rabi_frequency2(state1, state2, q, electricField, s)


    def get_rabi_frequency2(self, state1: A_QState, state2: A_QState,
                            q: Literal[-1, 0, 1], electricFieldAmplitude:float,
                            s: float = 0.5) -> float:
        """Returns the Rabi frequency for resonantly driven atom in a given electric field amplitude.

        Uses :meth:`get_dipole_matrix_element` for the calculation.
        
        Parameters
        ----------
        state1: A_QState
            NamedTuple of quantum numbers for state driving from
        state2: A_QState
            NamedTuple of quantum numbers for state driving to
        q: int
            laser polarization in spherical basis (-1,0,1) corresponding to
            :math:`\\sigma^-`, :math:`\\pi`, and :math:`\\sigma^+`
        electricFieldAmplitude: float
            amplitude of driving electric field, in V/m
        s: float, optional
            total spin angular momentum of the states. By default 0.5 for Alkali atoms.

        Returns
        -------
        rabi_frequency: float
            Rabi frequency in rad/s. To get Hz, divide by :math:`2\\pi`

        Examples
        --------
        >>> from rydiqule import A_QState
        >>> import arc
        >>> g_nlj = rq.A_QState(5, 0, 0.5)
        >>> g_fs = rq.A_QState(5, 0, 0.5, m_j=-0.5)
        >>> e_nlj = rq.A_QState(5, 1, 0.5)
        >>> e_hfs = rq.A_QState(5, 1, 0.5,f=2, m_f=0)
        >>> arc_atom = arc.alkali_atom_data.Rubidium85()
        >>> my_atom = rq.RQ_AlkaliAtom(arc_atom)
        >>> e_field = 0.1 #V/m
        >>> print(my_atom.get_rabi_frequency2(g_nlj, e_nlj, q=0, electricFieldAmplitude=e_field))
        13890.429
        >>> print(my_atom.get_rabi_frequency2(g_fs, e_hfs, q=0, electricFieldAmplitude=e_field))
        9822.0166
        >>> print(my_atom.get_rabi_frequency2(g_nlj, e_hfs, q=0, electricFieldAmplitude=e_field)) # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
          ...
        rydiqule.exceptions.AtomError: Invalid transition type for dipole calculation.

        """
        # TODO: may need to put checks to return 0 if just not dipole allowed

        dipole = self.get_dipole_matrix_element(state1, state2, q, s) * C_e * C_a_0

        freq = electricFieldAmplitude * abs(dipole) / hbar

        return freq
    
    def get_reduced_rabi_frequency(self, state1: A_QState, state2: A_QState,
                                   laserPower: float, laserWaist: float,
                                   s: float = 0.5) -> float:
        """Returns the Rabi frequency for resonantly driven atom in center of a TEM00 mode of a field.

        The field is calculated using :meth:`gaussian_center_field`.
        It then calls :meth:`get_rabi_frequency2` to get the rabi frequency.

        Note
        ----
        This function preserves the sign of the dipole moment
        (i.e. the result could be negative).
        As such, state calling order matters.
        To get correct convention for use in Cell,
        `state` must be lower energy than `state2`.
        
        Parameters
        ----------
        state1: A_QState
            NamedTuple of quantum numbers for state driving from
        state2: A_QState
            NamedTuple of quantum numbers for state driving to
        laserPower: float
            laser power in Watts
        laserWaist: float
            laser :math:`1/e^2` waist (radius) in meters
        s: float, optional
            total spin angular momentum of the states. By default 0.5 for Alkali atoms.

        Returns
        -------
        rabi_frequency: float
            Rabi frequency in rad/s. To get Hz, divide by :math:`2\\pi`

        Examples
        --------
        >>> from rydiqule import A_QState
        >>> import arc
        >>> g_nlj = A_QState(5, 0, 0.5)
        >>> g_fs = A_QState(5, 0, 0.5, m_j=-0.5)
        >>> e_nlj = A_QState(5, 1, 0.5)
        >>> e_hfs = A_QState(5, 1, 0.5,f=2, m_f=0)
        >>> arc_atom = arc.alkali_atom_data.Rubidium85()
        >>> my_atom = rq.RQ_AlkaliAtom(arc_atom)
        >>> print(my_atom.get_reduced_rabi_frequency(g_nlj, e_nlj, laserPower=1, laserWaist=0.01)/1e6) #MHz
        372.59
        >>> print(my_atom.get_reduced_rabi_frequency(g_fs, e_hfs, laserPower=1, laserWaist=0.01)/1e6) #MHz
        372.59
        >>> print(my_atom.get_reduced_rabi_frequency(g_nlj, e_hfs, laserPower=1, laserWaist=0.01)/1e6) #MHz
        372.59
        """

        electricField = self.gaussian_center_field(laserPower, laserWaist)
        
        return self.get_reduced_rabi_frequency2(state1, state2, electricField, s)


    def get_reduced_rabi_frequency2(self, state1: A_QState, state2: A_QState,
                                    electricFieldAmplitude:float,
                                    s: float = 0.5) -> float:
        """Returns the reduced Rabi frequency for resonantly driven atom in a given electric field amplitude.

        Uses  :math:`1/2`:meth:`get_reduced_matrix_elementJ`.

        Note
        ----
        This function preserves the sign of the dipole moment
        (i.e. the result could be negative).
        As such, state calling order matters.
        To get correct convention for use in Cell,
        `state1` must be lower energy than `state2`.
        
        Parameters
        ----------
        state1: A_QState
            NamedTuple of quantum numbers for state driving from
        state2: A_QState
            NamedTuple of quantum numbers for state driving to
        electricFieldAmplitude: float
            amplitude of driving electric field, in V/m
        s: float, optional
            total spin angular momentum of the states. By default 0.5 for Alkali atoms.

        Returns
        -------
        rabi_frequency: float
            Rabi frequency in rad/s. To get Hz, divide by :math:`2\\pi`

        Examples
        --------
        >>> from rydiqule import A_QState
        >>> import arc
        >>> g_nlj = A_QState(5, 0, 0.5)
        >>> g_fs = A_QState(5, 0, 0.5, m_j=-0.5)
        >>> e_nlj = A_QState(5, 1, 0.5)
        >>> e_hfs = A_QState(5, 1, 0.5,f=2, m_f=0)
        >>> arc_atom = arc.alkali_atom_data.Rubidium85()
        >>> my_atom = rq.RQ_AlkaliAtom(arc_atom)
        >>> e_field = 0.1 #V/m
        >>> print(my_atom.get_reduced_rabi_frequency2(g_nlj, e_nlj, electricFieldAmplitude=e_field))
        17012.23
        >>> print(my_atom.get_reduced_rabi_frequency2(g_fs, e_hfs, electricFieldAmplitude=e_field))
        17012.23
        >>> print(my_atom.get_reduced_rabi_frequency2(g_nlj, e_hfs, electricFieldAmplitude=e_field))
        17012.23
        
        """

        reduced_dipole = self.get_reduced_matrix_elementJ(state1, state2, s) * C_e * C_a_0 / 2

        return electricFieldAmplitude * reduced_dipole / hbar

        
    def get_spherical_dipole_matrix_element(self, 
                                            state1: A_QState, state2: A_QState,
                                            q: Literal[-1, 0, 1],
                                            s: float = 0.5) -> float:
        """Returns the spherical part of the dipole matrix element for a transition.
        
        Calculated by dividing the transition dipole moment by the
        reduced J matrix element.
        
        Parameters
        ----------
        state1: A_QState
            NamedTuple of quantum numbers for first state
        state2: A_QState
            NamedTuple of quantum numbers for second state
        q: int
            field polarization in the spherical basis (-1,0,1) corresponding to
            :math:`\\sigma^-`, :math:`\\pi`, and :math:`\\sigma^+`
        s: float, optional
            total spin angular momentum of the states. By default 0.5 for Alkali atoms.

        Returns
        -------
        float
            Spherical part of the dipole matrix element,
            in units of reduced matrix element :math:`\\langle J||d||J'\\rangle`

        Examples
        --------
        >>> from rydiqule import A_QState
        >>> import arc
        >>> g_nlj = A_QState(5, 0, 0.5)
        >>> g_fs = A_QState(5, 0, 0.5, m_j=-0.5)
        >>> e_nlj = A_QState(5, 1, 0.5)
        >>> e_hfs = A_QState(5, 1, 0.5,f=2, m_f=0)
        >>> arc_atom = arc.alkali_atom_data.Rubidium85()
        >>> my_atom = rq.RQ_AlkaliAtom(arc_atom)
        >>> print(my_atom.get_spherical_dipole_matrix_element(g_nlj, e_nlj, q=0))
        0.4082
        >>> print(my_atom.get_spherical_dipole_matrix_element(g_fs, e_hfs, q=0))
        0.2887
        >>> print(my_atom.get_spherical_dipole_matrix_element(g_nlj, e_hfs, q=0)) # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        rydiqule.exceptions.AtomError: Invalid transition type for dipole calculatdion. Allowed types are [('HFS', 'FS'), ('FS', 'HFS'), ('HFS', 'HFS'), ('FS', 'FS'), ('NLJ', 'NLJ')]
        """

        return (self.get_dipole_matrix_element(state1, state2, q, s)
                / self.get_reduced_matrix_elementJ(state1, state2, s)
        )
    

    def get_reduced_matrix_elementJ(self, state1: A_QState, state2: A_QState,
                                    s: float = 0.5) -> float:
        """Returns the reduced dipole matrix element in the J basis.
        
        A convenience wrapper for 
        :external+arc:meth:`~arc.alkali_atom_functions.AlkaliAtom.getReducedMatrixElementJ`

        Note
        ----
        To get proper sign conventions, state order must go from lower energy
        to higher energy state.

        Parameters
        ----------
        state1: A_QState
            NamedTuple of quantum numbers for lower state
        state2: A_QState
            NamedTuple of quantum numbers for higher state
        s: float, optional
            total spin angular momentum of the states. By default 0.5 for Alkali atoms.

        Returns
        -------
        float
            Reduced matrix element :math:`\\langle J||d||J'\\rangle`

        Examples
        --------
        >>> from rydiqule import A_QState
        >>> import arc
        >>> g_nlj = A_QState(5, 0, 0.5)
        >>> g_fs = A_QState(5, 0, 0.5, m_j=-0.5)
        >>> e_nlj = A_QState(5, 1, 0.5)
        >>> e_hfs = A_QState(5, 1, 0.5,f=2, m_f=0)
        >>> arc_atom = arc.alkali_atom_data.Rubidium85()
        >>> my_atom = rq.RQ_AlkaliAtom(arc_atom)
        >>> print(my_atom.get_reduced_matrix_elementJ(g_nlj, e_nlj))
        4.2321
        >>> print(my_atom.get_reduced_matrix_elementJ(g_fs, e_hfs))
        4.2321
        >>> print(my_atom.get_reduced_matrix_elementJ(g_nlj, e_hfs))
        4.2321

        """

        return self.arc_atom.getReducedMatrixElementJ(*state1[:3], *state2[:3], s)


    def get_transition_rate(self, state1: A_QState, state2: A_QState,
                            temperature: float = 0.0, s: float = 0.5) -> float:
        """Returns transition rate between two states due to spontaneous emission.

        If temperature is provided, Black-Body Radiation induced transitions are included.
        Otherwise, rate is due to the natural radiative lifetime only.

        Uses ARC's :external+arc:meth:`~arc.alkali_atom_functions.AlkaliAtom.getTransitionRate`
        to get the base transition rate between :math:`|n_1,l_1,j_1\\rangle` 
        to :math:`|n_2,l_2,j_2\\rangle`.
        If state1 and/or state2 are sublevels in the fine or hyperfine structures,
        this further applies the appropriate branching ratio.

        Parameters
        ----------
        state1: A_QState
            NamedTuple of quantum numbers of the originating state
        state2: A_QState
            NamedTuple of quantum numbers of the target state
        temperature: float, optional
            Temperature of th atomic environment for calculationg BBR-induced
            decays, in Kelvin.
            With default of 0.0, only include natural lifetime.
        s: float, optional
            total spin angular momentum. Default of 0.5 for Alkali atoms

        Returns
        -------
        float
            Transition rate in 1/s

        Raises
        ------
        AtomError
            If states are in both NLJ and FS/HFS definition.

        Examples
        --------
        >>> from rydiqule import A_QState
        >>> import arc
        >>> g_nlj = A_QState(5, 0, 0.5)
        >>> g_fs = A_QState(5, 0, 0.5, m_j=-0.5)
        >>> e_nlj = A_QState(5, 1, 0.5)
        >>> e_hfs = A_QState(5, 1, 0.5,f=2, m_f=0)
        >>> arc_atom = arc.alkali_atom_data.Rubidium85()
        >>> my_atom = rq.RQ_AlkaliAtom(arc_atom)
        >>> print(my_atom.get_transition_rate(e_nlj, g_nlj)/1e6) #GHz
        36.11
        >>> print(my_atom.get_transition_rate(e_hfs, g_fs)/1e6) #GHz
        18.06
        >>> print(my_atom.get_transition_rate(e_hfs, g_nlj)) # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        rydiqule.exceptions.AtomError: Transition between HFS and NLJ are not allowed.
        """
        
        #dict of functions based on state types
        arc_transition_prefactor_functions: Dict[Tuple[str, str], Callable] = {
            ("NLJ","NLJ"): lambda *x: 1, #there is no prefactor
            ("FS","FS"): self._tr_prefactor_fs_fs,
            ("HFS", "HFS"): self._tr_prefactor_hfs_hfs,
            ("FS", "HFS"): self._tr_prefactor_fs_hfs,
            ("HFS", "FS"): self._tr_prefactor_hfs_fs
        }

        try:
            base_transition_rate = self.arc_atom.getTransitionRate(*state1[:3],
                                                          *state2[:3],
                                                          temperature,
                                                          s)
        except ValueError:
            return 0.0

        try:
            prefactor = arc_transition_prefactor_functions[(state1.stype, state2.stype)](state1, state2, s)
        except KeyError:
            msg = f"Transition between {state1.stype} and {state2.stype} are not allowed. \
            Allowed transition types are {list(arc_transition_prefactor_functions.keys())}."
            raise AtomError(msg)
        
        return base_transition_rate*prefactor


    def get_state_lifetime(self, state: A_QState,
                           temperature: float = 0.0, includeLevelsUpTo:int = 0,
                           s: float = 0.5) -> float:
        """Get lifetime of the state.

        If temperature is provided, includes Black-Body Radiation induced transitions.
        Otherwise, this is the natural lifetime of the state.

        This is a thin wrapper around ARC's
        :external+arc:meth:`~arc.alkali_atom_functions.AlkaliAtom.getStateLifetime` method.
        It adds basic validation of the state and selects the correct quantum numbers for the 
        calcuation.

        Parameters
        ----------
        state: A_QState
            NamedTuple of quantum numbers of state for which to calculate lifetime. 
        temperature: float, optional
            Temperature at which the atom environmnet is, in Kelvin.
            Used for calculating the black-body-induced state lifetime.
            If 0.0 (default), result does not inlclude BBR term.
        includeLevelsUpTo: int, optional
            If `temperature` is non-zero, this specifies the highest principal
            quantum number states to include in the BBR calculation.
            Must be at least `n+1` of the provided state.
        s: float, optional
            total spin angular momentum of the state. Default is 0.5, for alkali atoms.

        Returns
        -------
        float
            State lifetime in seconds.

        Examples
        --------
        >>> from rydiqule import A_QState
        >>> import arc
        >>> e = A_QState(10, 0, 0.5)
        >>> e_fs = A_QState(10, 0, 0.5, m_j=-0.5)
        >>> e_nlj = A_QState(10, 1, 0.5)
        >>> e_hfs = A_QState(10, 1, 0.5,f=2, m_f=0)
        >>> arc_atom = arc.alkali_atom_data.Rubidium85()
        >>> my_atom = rq.RQ_AlkaliAtom(arc_atom)
        >>> print(my_atom.get_state_lifetime(e_nlj))
        1.1626e-06
        >>> print(my_atom.get_state_lifetime(e_fs))
        4.20984e-07
        >>> print(my_atom.get_state_lifetime(e_nlj))
        1.16261e-06

        """
        return self.arc_atom.getStateLifetime(*state[:3], temperature, includeLevelsUpTo, s)

    
    """Utility functions for appropriately dispatching between state types not directly supported by arc"""

    #dipole functions 
    def _get_nlj_dipole(self, n1, l1, j1, n2, l2, j2, q, s):
        """Dipole matrix element between a pair of NLJ states, in units of e*a0"""
        dme = 0.0
        n_dme = 0
        mj1s = np.arange(-1*j1, j1+1)
        mj2s = mj1s + q
        for mj1, mj2 in zip(mj1s, mj2s):
            if abs(mj2) > j2:
                continue
            dme += abs(self.arc_atom.getDipoleMatrixElement(n1, l1, j1, mj1, n2, l2, j2, mj2, q, s))
            n_dme += 1
        return dme/n_dme
    
    def _getDipoleMatrixElementFStoHFS(self, n1, l1, j1, mj1, n2, l2, j2, f2, mf2, q, s):
        """Funtion which inverts arcs HFS to FS dipole matrix element function, , in units of e*a0"""
        return self.arc_atom.getDipoleMatrixElementHFStoFS(n2, l2, j2, f2, mf2, n1, l1, j1, mj1, q, s)


    #transition rate prefactor functions
    def _tr_prefactor_fs_fs(self, state1, state2, s):
        """transition rate prefactor between 2 hyperfine states.
        Units of reduced matrix element :math:`<j|er|j'>`"""
        return self.arc_atom.getBranchingRatioFStoFS(*state2.qnums[2:],
                                                     *state1.qnums[2:],
                                                     s)
            

    def _tr_prefactor_hfs_hfs(self, state1, state2, s):
        """transition rate prefactor between hyperfine states.
        returns the branching ratio for a particular F state.
        Units of reduced matrix element :math:`<j|er|j'>`"""
        return self.arc_atom.getBranchingRatio(*state2.qnums[2:],
                                               *state1.qnums[2:],
                                               s)
    
    def _tr_prefactor_fs_hfs(self, state1, state2, s):
        """transition rate prefactor for transitions from a fine
        structure state to a hyperfine state.
        Units of reduced matrix element :math:`<j|er|j'>`"""
        return self.arc_atom.getBranchingRatioFStoHFS(*state2.qnums[2:],
                                                      *state1.qnums[2:],
                                                      s)
    
    def _tr_prefactor_hfs_fs(self, state1, state2, s):
        """transition rate prefactor for transitions from a
        hyperfine state to a fine structure state.
        Units of reduced matrix element :math:`<j|er|j'>`"""
        return self.arc_atom.getBranchingRatioHFStoFS(*state2.qnums[2:],
                                                      *state1.qnums[2:],
                                                      s)
    