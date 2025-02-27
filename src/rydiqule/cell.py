"""
Subclass of `Sensor` with functionality for representing real atoms.
"""

import scipy
import numpy as np
import warnings
import itertools 

import scipy.constants
from scipy.constants import Boltzmann, e

# rydiqule imports
from .sensor import Sensor
from .sensor_utils import scale_dipole
from .sensor_utils import ScannableParameter, TimeFunc
from .atom_utils import ATOMS, calc_kappa, calc_eta, expand_qnums, validate_qnums, A_QState, ground_state
from .arc_utils import RQ_AlkaliAtom
from .exceptions import RydiquleError, AtomError, CouplingNotAllowedError
from .exceptions import RydiquleWarning, debug_state

from typing import Literal, Optional, Sequence, List, Tuple, Callable, Union, Dict

a0 = scipy.constants.physical_constants["Bohr radius"][0]

AtomFlags = Literal['H', 'Li6', 'Li7', 'Na', 'K39', 'K40', 'K41', 'Rb85', 'Rb87', 'Cs']

class Cell(Sensor):
    """
    Subclass of :class:`~.Sensor` that creates a Sensor with additional physical properties
    corresponding to a specific Rydberg atom.

    In addition to the core functionality of `~.Sensor`, this class requires labelling states
    with `namedtuple`s of quantum numbers, automatically calculating of state lifetimes and
    decoherences and tracking of of some physical laser parameters.
    A key distictinction between a :class:`~.Cell` and a :class:`~.Sensor` is that
    a cell supports (and requires) and absolute ordering of energy between states,
    which allows for implicit calculation of decay rates and transition frequencies.

    """

    def __init__(self, atom_flag: AtomFlags, atomic_states: List[A_QState],
                 cell_length: float = 1e-3, gamma_transit: Optional[float] = None,
                 gamma_mismatch:Union[str, dict]="ground", beam_area: float = 1e-6, 
                 beam_diam: Optional[float] = None, temp: float = 300.0
                 ) -> None:
        """
        Initialize the Rydberg cell from the given parameters. 

        Parameters
        ----------
        atom_flag : str 
            Which atom is used in the cell for calculating physical properties with ARC Rydberg.
            One of {'H', 'Li6', 'Li7', 'Na', 'K39', 'K40', 'K41', 'Rb85', 'Rb87', 'Cs'}.
        atomic_states : list of A_QState
            List of :class:`~.atom_utils.A_QState` representing the states of the atom. 
            More details about the `A_QState` class can be found in its documentation, but it
            includes the elemends (`n`, `l`, `j`, `m_j`, `f`, `m_f`). These represent the 
            usual Hydrogen-like atom quantum numbers with the usual restrictions:
                
                - `n` must be a positive integer.
                - `l` must be a non-negative integer less than `n`.
                - `j` must be a positive half-integer such that :math:`j=l \\pm \\frac{1}{2}`
                - `m_j` must be a half integer such that :math:`-j \\leq m_j \\leq +j`
                - `f` must be an integer satisfying :math:`|j-I| \\leq f \\leq (j+I)`
                - `m_f` must be an integer such that :math:`-f \\leq m_f \\leq +f`

            Additionally, `n`,`l`, and `j` must always be specified, in addition to the following
            restrictions on other quantum numbers:

                - `m_j` cannot be specified with `f`
                - If `m_f` is specified, `f` must also be specified.

            All quantum numbers can be specified with lists of valid values, in which case the
            they will be expanded into a list of states, with one corresponding to each value. If
            multiple quantum numbers are specified with lists, the resulting list of states will
            contain all combinations of values. Furthermore, the `j`, `m_j`, `f`, and `m_f` 
            quantum numbers can each be specifed using `"all"`, which corresponds to a list of
            all physically allowed values for that quantum number. This convention allows quick
            specifications of entire manifolds of states to be added to the sensor. See the 
            `Examples` section to see how to use these powerful specifications.

        cell_length: float
            Length of the atomic vapor in meters.
        gamma_transit : float, optional
            Decoherence due to atom transit through the optical
            beams. Specified in units of Mrad/s. If `None`, will calculate based
            on value of `beam_area`. See :meth:`~.Sensor.add_transit_broadening` for
            details on how transit broadening is treated. Default is None.
        gamma_mismatch : str or dict
            How to resolve discrepancies between calculated eacb state lifetime and the sum of 
            all transtion rates out of each state. In practice, these discrepancies are a result of
            transition pathways which exist between states when one is accounted for is `states`
            and one is not. For example, if there is a decoherent transition between states 3->1
            and states 3->2, but `states` only includes states 0, 1, and 3, the calculated 
            `gamma_lifetime` of state 3 will be greater than the sum of all computed
            `gamma_transition` values. In many cases, it is desirable to account for this
            decoherence in other ways. The options for handling the dicrepancy are:

                - `"ground"` which adds a decoherent coupling coupling between a state `s` with
                  a discrepancy :math:`\\Delta \\gamma` and divides :math:`\\Delta \\gamma` among 
                  all the ground states (states matching the `n,l,j` values of the lowest energy
                  state). 
                - `"all"` which divides :math:`\\Delta \\gamma` amongst all states in the 
                  :class:`~.Cell` which already have a `"gamma_transition"` value. The fraction
                  each transition gets is weighted by the fraction of the total thdat transitions
                  `"gamma_transition"` value accounts for. 
                - `"none"` which will not account for this discrepancy at all. In this case this
                  physics is not guaranteed to be accurate and it is assumed the decoherence will
                  be accounted for manually in other ways using :meth:`~.add_decoherence`

            In all cases, the accounting is done by adding a `"gamma_mismatch"` value to each
            relevant edge, which will subsequently be accounted for when 
            :meth:`~.Sensor.decoherence_matrix` is called. Note that older versions of 
            `rydiqule` did not have this option and implicitly used the `"ground"` option,
            and `"ground"` is the current default.
        beam_area : float, optional
            Area of probing field cross-section in m^2.
            Used to calculate `kappa` and `gamma_transit`. Default is 1e-6.
        beam_diam : float, optonal 
            Diameter of the probing field cross section in meters.
            Used to calculate `gamma_transit`. If `None`, it is calculated from
            `beam_area` assuming the beam cross-section is a circle. Default is `None`.
        temp : float, optional
            Temperature of the gas in Kelvin.
            Used in calculations of enery level lifetime. Default is 300 K.

        Raises
        ------
        RydiquleError
            If at least two atomic states are not provided.
        AtomError
            If atom_flag is not one of ARC's supported alkali atoms.

        Warns
        -----
        NLJWarning
            If called using old-style state specification

        Examples
        --------
        All the hyperfine states for the D1 line of Rubidium-87 can be defined as follows.

        >>> from rydiqule import A_QState
        >>> D1_g = A_QState(5, 0, 0.5, f="all", m_f="all")
        >>> D1_e = A_QState(5, 1, 0.5, f="all",m_f="all")
        >>> c = rq.Cell("Rb87",[D1_e, D1_g])
        >>> for state in c.states:
        ...     print(state)
        (5, 1, 0.5, f=1.0, m_f=-1.0)
        (5, 1, 0.5, f=1.0, m_f=0.0)
        (5, 1, 0.5, f=1.0, m_f=1.0)
        (5, 1, 0.5, f=2.0, m_f=-2.0)
        (5, 1, 0.5, f=2.0, m_f=-1.0)
        (5, 1, 0.5, f=2.0, m_f=0.0)
        (5, 1, 0.5, f=2.0, m_f=1.0)
        (5, 1, 0.5, f=2.0, m_f=2.0)
        (5, 0, 0.5, f=1.0, m_f=-1.0)
        (5, 0, 0.5, f=1.0, m_f=0.0)
        (5, 0, 0.5, f=1.0, m_f=1.0)
        (5, 0, 0.5, f=2.0, m_f=-2.0)
        (5, 0, 0.5, f=2.0, m_f=-1.0)
        (5, 0, 0.5, f=2.0, m_f=0.0)
        (5, 0, 0.5, f=2.0, m_f=1.0)
        (5, 0, 0.5, f=2.0, m_f=2.0)

        """
        if atom_flag not in ATOMS.keys():
            raise AtomError(f"Atom flag must be one of {ATOMS.keys()}")

        self.atom_flag = atom_flag
        self.atom = RQ_AlkaliAtom(ATOMS[atom_flag]())
        self.I = self.atom.arc_atom.I

        #prepare states by expanding statespec into list and intialize graph
        qstates_list = expand_qnums(atomic_states, I=self.I)
        if len(qstates_list) < 2:
            raise RydiquleError(("At least 2 states must be specified in a Cell"))

        self._validate_input_states(qstates_list)
        super().__init__(qstates_list)
        if debug_state():
            print(f'Cell states added: {qstates_list}')

        #set physical constant attributes
        self.cell_length = cell_length  # Default 0 to do optically thin sample
        self.temp = temp  # K
        self.beam_area = beam_area
        self.density = self.atom.arc_atom.getNumberDensity(self.temp)
        self.atom_mass = self.atom.arc_atom.mass
        if beam_diam is None:
            self.beam_diameter = 2.0*np.sqrt(beam_area/np.pi)
        else:
            self.beam_diameter = beam_diam

        if gamma_transit is None:
            gamma_transit = 1E-6*np.sqrt(8*Boltzmann*self.temp/(self.atom_mass*np.pi)
                                         )/(self.beam_diameter/2*np.sqrt(2*np.log(2)))
        self.gamma_transit = gamma_transit

        # most probable speed for a 3D Maxwell-Boltzmann distribution
        # used when defining doppler averaging
        self.vP = np.sqrt(2*Boltzmann*self.temp/self.atom_mass)
   
        self._add_state_energies()
        self._add_state_lifetimes()
        self._add_decoherence_rates()
        self._add_gamma_mismatches(gamma_mismatch)

    
    def set_experiment_values(self,
                              probe_freq: float,
                              kappa: float,
                              eta: Optional[float] = None,
                              cell_length: Optional[float] = None,
                              beam_area: Optional[float] = None):
        """`Sensor` specific method. Do not use with `Cell`.

        This function does not do anything as Cell automatically handles 
        this functionality internally.

        Warns
        -----
        RydiquleWarning: Warns if function is used.
        """

        warnings.warn('set_experiment_values not used with Cell',
                      RydiquleWarning)
    

    @Sensor.probe_tuple.setter
    def probe_tuple(self, probe_tuple: Tuple[A_QState, A_QState]):
        """Setter method for the `probe_tuple` attribute.

        The `probe_tuple` of a `Cell` is the transition used by
        default in calculations of observable values after solving. 
        Both states must be:
        1. Specified using only NLJ quantum numbers
        2. Both use either fine or hyperfine structure splitting interchangeably.

        Parameters
        ----------
        probe_tuple : tuple of A_QState
            Length-2 tuple of states to set as the `probe_tuple` attribute for the `Cell`

        Raises
        ------
        ValueError
            If the states provided are not nodes of the `Cell`
        RydiquleError
            If the states are of incompatible types. (e.g. NLJ and sublevel) 

        Notes
        -----
        This setter is often uneccesary to call directly, as the `probe_tuple` attibute is also
        set implicitly by the first coupling added to the system using the 
        :meth:`~.Sensor.add_coupling` method.

        """
                
        state1_list = self.states_with_spec(probe_tuple[0])
        state2_list = self.states_with_spec(probe_tuple[1])

        if not (len(state1_list) > 0 and len(state2_list) > 0):
            raise ValueError(f"Probe tuple specification {probe_tuple} contains invalid state specs") 

        n1, l1, j1, _, _, _ = state1_list[0]
        n2, l2, j2, _, _, _ = state2_list[0]

        nlj_match1 = all([(n,l,j) == (n1, l1, j1) for (n,l,j,_,_,_) in state1_list])
        nlj_match2 = all([(n,l,j) == (n2, l2, j2) for (n,l,j,_,_,_) in state2_list])

        if not (nlj_match1 and nlj_match2):
            msg = "Either upper or lower manifolds of probing transition have differing nlj "\
                  "values. Please ensure all states in each of the upper and lower manifolds "\
                  "of probe tuple or the first coupling added have matching nlj values."
            raise RydiquleError(msg)

        self._probe_tuple = probe_tuple


    def level_ordering(self) -> List[A_QState]:
        """
        Return a list of the states in the `Cell` in ascending energy order. 

        All energies are calculated with respect to the ground state energy, which is defined as 0.
        Ground state is determined by the rydiqule's calculation of ground energy, which
        uses `arc` to get the energy of the :math:`nP^{\\frac{1}{2}}` state, where `n` is 1 for
        Hydrogen, 2 for Lithium, etc.

        Returns
        -------
        list of A_QState
            The Cell states in order of decending energy
            relative to the ground state :math:`nS^{\\frac{1}{2}}`.

        Examples
        --------
        For the following example, states are in the list passed to the constructor in
        ascending energy order, so the ordering the basis is identical to the `level_ordering`.
        Computed `Cell` attributes the Hamiltonian will, for clarity, always appear in the
        ording of levels in the list passed to the constructor.
        
        >>> from rydiqule import A_QState
        >>> atom = "Rb85"
        >>> [g, e] = rq.D2_states(atom)  #uses the D2 line of Rb85
        >>> state1 = A_QState(50, 2, 2.5)
        >>> state2 = A_QState(51, 2, 2.5)
        >>> my_cell = rq.Cell(atom, [g, e, state1, state2])
        >>> print(my_cell.states)
        [(n=5, l=0, j=0.5), (n=5, l=1, j=1.5), (n=50, l=2, j=2.5), (n=51, l=2, j=2.5)]
        >>> # levels in order
        >>> for i, state in enumerate(my_cell.level_ordering()):
        ...     print(f"{i}: {state}, E={my_cell.couplings.nodes[state]['energy']*2*np.pi*1e-6} Mrad/s")
        0: (5, 0, 0.5), E=0.0 Mrad/s
        1: (5, 1, 1.5), E=15168.8 Mrad/s
        2: (50, 2, 2.5), E=39819.3 Mrad/s
        3: (51, 2, 2.5), E=39821.5 Mrad/s

        If we scramble the states in the constructor, the output of this function remains
        the same even though the order of basis states changes to match the list ordering
        in the constuctor.

        >>> from rydiqule import A_QState
        >>> atom = "Rb85"
        >>> [g, e] = rq.D2_states(atom)  #uses the D2 line of Rb85
        >>> state1 = A_QState(50, 2, 2.5)
        >>> state2 = A_QState(51, 2, 2.5)
        >>> my_cell = rq.Cell(atom, [state2, e, g, state1])
        >>> print(my_cell.states)
        [(n=51, l=2, j=2.5), (n=5, l=1, j=1.5), (n=5, l=0, j=0.5), (n=50, l=2, j=2.5)]
        >>> for i, state in enumerate(my_cell.level_ordering()):
        ...     print(f"{i}: {state}, E={my_cell.couplings.nodes[state]['energy']*2*np.pi*1e-6} Mrad/s")
        0: (5, 0, 0.5), E=0.0 Mrad/s
        1: (5, 1, 1.5), E=15168.8 Mrad/s
        2: (50, 2, 2.5), E=39819.3 Mrad/s
        3: (51, 2, 2.5), E=39821.5 Mrad/s

        """
        energies = list(self.couplings.nodes("energy")).copy()
        energies.sort(key=lambda val:val[1],reverse=False)
        return [s[0] for s in energies]


    @property
    def kappa(self):
        """Property to calculate the kappa value of the system. 

        The value is computed with the following formula Eq. 5 of
        Meyer et. al. PRA 104, 043103 (2021)

        .. math::

            \\kappa = \\frac{\\omega n \\mu^2}{2c \\epsilon_0 \\hbar}

        Where :math:`\\omega` is the probing frequency, :math:`\\mu` is the dipole moment,
        :math:`n` is atomic cloud density, :math:`c` is the speed of light, :math:`\\epsilon_0`
        is the dielectric constant, and :math:`\\hbar` is the reduced Plank constant.

        This value is only computed if there is not a `_kappa` attribute in the system.
        If this attribute does exist, this function acts as an accessor for that attribute.

        Returns
        -------
        float
            The value kappa for the system. 
        """ 

        #does nothing now, possible future-proofing
        if hasattr(self, "_kappa"):
            return self._kappa
        
        if self.probe_tuple is None:
            raise RydiquleError("Cell.probe_tuple not set. Either set manually or add at least one coupling before calculation.")
        
        ground_manifold = self.states_with_spec(self.probe_tuple[0])
        excited_manifold = self.states_with_spec(self.probe_tuple[1])

        #get the probing transition states for nlj only
        probe_g_nlj = A_QState(ground_manifold[0].n, ground_manifold[0].l, ground_manifold[0].j)
        probe_e_nlj = A_QState(excited_manifold[0].n, excited_manifold[0].l, excited_manifold[0].j)

        #ensure nlj all match
        nlj_match_g = not any([s[:3] != probe_g_nlj[:3] for s in ground_manifold])
        nlj_match_e = not any([s[:3] != probe_e_nlj[:3] for s in excited_manifold])

        if not (nlj_match_g and nlj_match_e):
            msg = "automatic kappa calcuations not supported for probing transitons between state manifolds "\
                "containg differing values of j quantum numbers. kappa can be set manually using <Cell>.kappa "\
                    " = <value> prior to running calculations"
            warnings.warn(msg)

        q = self.couplings.edges[ground_manifold[0], excited_manifold[0]]["q"]

        omega_rad = self.atom.get_transition_frequency(probe_g_nlj, probe_e_nlj)*2*np.pi
        dipole_moment = self.atom.get_dipole_matrix_element(probe_g_nlj, probe_e_nlj, q=q)*a0*e

        kappa = calc_kappa(omega_rad, dipole_moment, self.density)
    
        return kappa


    @kappa.setter
    def kappa(self, value: float):
        """Setter for the kappa attribute.

        Typically not required, as the `kappa` attribute is inferred implicitly
        by the formula described in its `property` description.

        Updates the self._kappa class attribute.

        Parameters
        ----------
        value : float
            The floating-point value to set as the kappa parameter for the system.
        """
        self._kappa = value


    @kappa.deleter
    def kappa(self):
        """Setter for the kappa attribute.

        Removes the self._kappa class attribute.

        Raises
        ------
        RydiquleError:
            If kappa has not been set.
        """
        try:
            del self._kappa
        except AttributeError as err:
            raise RydiquleError("The \"kappa\" attribute has not been set") from err
    

    @property
    def eta(self):
        """Get the eta value for the system.

        The value is computed with the following formula Eq. 7 of
        Meyer et. al. PRA 104, 043103 (2021)

        .. math::

            \\eta = \\sqrt{\\frac{\\omega \\mu^2}{2 c \\epsilon_0 \\hbar A}}

        Where :math:`\\omega` is the probing frequency, :math:`\\mu` is the dipole moment,
        :math:`A` is the beam area, :math:`c` is the speed of light, :math:`\\epsilon_0`
        is the dielectric constant, and :math:`\\hbar` is the reduced Plank constant.

        This value is only computed if there is not a `_eta` attribute in the system.
        If this attribute does exist, this function acts as an accessor for that attribute.

        Returns
        -------
        float
            The value eta for the system.
        """
        #does nothing now, possible future-proofing
        if hasattr(self, "_eta"):
            return self._eta
        if self.probe_tuple is None:
            raise RydiquleError("Cell.probe_tuple not set. Either set manually or add at least one coupling before calculation.")
        
        ground_manifold = self.states_with_spec(self.probe_tuple[0])
        excited_manifold = self.states_with_spec(self.probe_tuple[1])

        #get the probing transition states for nlj only
        probe_g_nlj = A_QState(ground_manifold[0].n, ground_manifold[0].l, ground_manifold[0].j)
        probe_e_nlj = A_QState(excited_manifold[0].n, excited_manifold[0].l, excited_manifold[0].j)

        #ensure nlj all match
        nlj_match_g = not any([s[:3] != probe_g_nlj[:3] for s in ground_manifold])
        nlj_match_e = not any([s[:3] != probe_e_nlj[:3] for s in excited_manifold])

        if not (nlj_match_g and nlj_match_e):
            msg = "automatic eta calcuations not supported for probing transitons between state manifolds "\
                "containg differing values of j quantum numbers. eta can be set manually using <Cell>.eta "\
                    " = <value> prior to running calculations"
            warnings.warn(msg)

        q = self.couplings.edges[ground_manifold[0], excited_manifold[0]]["q"]

        omega_rad = self.atom.get_transition_frequency(probe_g_nlj, probe_e_nlj)*2.0*np.pi
        dipole_moment = self.atom.get_dipole_matrix_element(probe_g_nlj, probe_e_nlj, q=q)*a0*e
        eta = calc_eta(omega_rad, dipole_moment, self.beam_area)

        return eta
    

    @eta.setter
    def eta(self, value):
        """Setter for the eta attribute.

        Updates the self._eta class attribute.

        Parameters
        ----------
        value : float
            The floating-point value to set as the eta parameter for the system.
        """
        self._eta = value


    @eta.deleter
    def eta(self):
        """Deleter for the eta attribute.

        Removes the self._eta class attribute.
        
        Raises
        ------
        RydiquleError:
            If eta has not been set.
        """
        try:
            del self._eta
        except AttributeError as err:
            raise RydiquleError("The \"eta\" attribute has not been set") from err


    @property
    def probe_freq(self):
        """Get the probe transition frequency, in rad/s.

        Note that for :class:`~.Cell`, probing transition frequency is calculated using only
        the `(n,l,j)` states of the upper and lower manifolds of the `probe_tuple` attribute. For 
        more precise calculations accounting for atomic splitting etc, `probe_freq` must be set 
        manually; `rydiqule` does not support doing these calculations automatically.
        
        Returns
        -------
        float
            Probe transitiion frequency, in rad/s, between probing nlj states. 
        """

        if hasattr(self, '_probe_freq'):
            return self._probe_freq
        
        probe_lower_manifold = self.states_with_spec(self.probe_tuple[0])
        probe_upper_manifold = self.states_with_spec(self.probe_tuple[1])

        (n1, l1, j1) = probe_lower_manifold[0][:3]
        (n2, l2, j2) = probe_upper_manifold[0][:3]

        energy_lower = self.atom.get_state_energy(A_QState(n1, l1, j1), s=0.5)*2*np.pi
        energy_upper = self.atom.get_state_energy(A_QState(n2, l2, j2), s=0.5)*2*np.pi
        
        return np.abs(energy_upper - energy_lower)
    
    @probe_freq.setter
    def probe_freq(self, value):
        """Setter for the probe_freq attribute.

        Updates the self._probe_freq class attribute.

        Parameters
        ----------
        value : float
            The floating-point value to set as the probe frequency parameter for the system.
        """
        self._probe_freq = value


    @probe_freq.deleter
    def probe_freq(self):
        """Deleter for the probe_freq attribute.

        Removes the self._probe_freq class attribute.
        
        Raises
        ------
        RydiquleError:
            If probe_freq has not been set.
        """
        try:
            del self._probe_freq
        except AttributeError as err:
            raise RydiquleError("The \"probe_freq\" attribute has not been set") from err


    def add_single_coupling(
            self, states: Tuple[A_QState, A_QState], rabi_frequency: Optional[ScannableParameter] = None,
            detuning: Optional[ScannableParameter] = None,
            transition_frequency: Optional[float] = None,
            phase: Optional[ScannableParameter] = None,
            kunit: Sequence[float] = (0,0,0),
            time_dependence: Optional[TimeFunc] = None, label: Optional[str] = None,
            e_field: Optional[ScannableParameter] = None, beam_power: Optional[float] = None,
            beam_waist: Optional[float] = None, coherent_cc: Optional[float]=None,
            q: Literal[-1, 0, 1] = 0,
            **extra_kwargs) -> None:
        """
        Overload of :meth:`~.Sensor.add_single_coupling`, which allows for alternate specifications
        and automatic calculations of some parameter. 

        This overload fundamentally works identically the `super` method in `Sensor`, with several
        additions to the functionality that make some assumptions about the underlying system. 
        Because of this, it still preferred to call :meth:`~.Sensor.add_coupling` on a 
        :class:`~.Cell` as well. Please refer to that methods documentation for further detail

        Rabi frequency is a mandatory argument in :class:`~.Sensor` but in :class:`~.Cell`, 
        there are 3 options for laser power specification:

        1. Explicit rabi-frequency definition identical to :class:`~.Sensor`.
        2. Electric field strength, in V/m.
        3. Specification of both beam power and beam waist, in W and m respectively.

        Any one of these options can be used in place of the standard `rabi_frequency` argument
        of :meth:`~.Sensor.add_coupling`. Note that in all cases, a `rabi_frequency` will be
        computed and passed to :meth:`~.Sensor.add_single_coupling`, and none of the above
        arguments will be added to the graph. Note that in any of these cases, if the computed
        dipole moment for the transition is zero, the coupling will be left off the graph.

        In all cases, the relative coupling strengths between sublevels of states (if present)
        is calculated and saved as the `coherent_cc` parameter on the graph.
        These coefficients are defined to be in units of :math:`1/2\\langle J||d||J'\\rangle`,
        as calculated by 
        :external+arc:meth:`~arc.alkali_atom_functions.AlkaliAtom.getReducedMatrixElementJ`.
        The corresponding Rabi frequency that Cell calculates therefore corresponds to
        :math:`\\Omega_{red} = E \\langle J||d||J'\\rangle / 2\\hbar`.
        The Rabi frequency for each transition added to the hamiltonian is then given by
        :math:`\\text{coherent_cc}\\cdot\\Omega_{red}\\cdot e^{i\\text{phase}}`.
        In the case of NLJ states (ie no sublevels), `coherent_cc=1` and the 
        non-reduced Rabi frequency is used instead.
        See the :doc:`physics documentation </writeups/sublevels>` for further details.

        As in :class:`~.Sensor`, if `detuning` is specified, the coupling is assumed
        to act under the rotating-wave approximation (RWA), and `transition_frequency`
        can not be specified. However, unlike in a :class:`~.Sensor`, if `detuning`
        is not specified, in a :class:`~.cell.Cell`, `transition_frequency` will be 
        calculated automatically based on atomic properties rather than taken as an argument.

        Parameters
        ----------
        states: sequence
            Length-2 list-like object (list or tuple) of integers
            corresponding to the numbered states of the cell.
            Tuple order indicates which state to has higher energy:
            namely the second state is always assumed to have higher energy.
            This order must match the actual energy levels of the atom.
        rabi_frequency: float, optional
            The rabi frequency, in Mrad/s, of the coupling
            field. If specified, `e_field`, `beam_power`, and `beam_waist` cannot
            be specified.
        detuning: float, optional
            Field detuning, in Mrad/s, of a coupling
            in the RWA. If specified, RWA is assumed, otherwise RWA not assumed,
            and transition frequency will be calculated based on atomic properties.
        transition_frequency: float, optional
            Kept such that method signature matches parent.
            Value must be `None` as the transition frequency
            is calculated from atomic properties.
        phase : float, optional
            Static phase offset in the rotating frame.
            Cannot be used outside the rotating frame, ie when detuning is not defined.
            Default is undefined, which is interpreted as 0 for couplings in the rotating frame.
        kunit: sequence, optional
            A three-element iterable that defines the
            propagation direction of the field.
            It should be a normalized vector.
            This differs from `Sensor`'s `kvec` parameter,
            since appropriate scale factors are applied automatically in `Cell`.
            If equal to `(0,0,0)`, solvers will ignore doppler shifts
            on this field.  Defaults to `(0,0,0)`.
        time_dependence: scalar function, optional
            A scalar function that
            specifies a time-dependent field. The time dependence function
            is defined as a funtion that returns a unitless value as a function
            of time that is multiplied by the `rabi_frequency` parameter.
        label: str, optional
            The user-defined name of the coupling. This does not change
            any calculations, but can be used to track individual couplings, and
            will be reflected in the output of :meth:`~.Sensor.axis_labels`
            Default None results in using the states tuple as the label.
        e_field: float, optional
            Electric field strenth of the coupling in Volts/meter.
            If specified, `rabi_frequency`, `beam_power`, and `beam_waist` cannot
            be specified.
        beam_power: float, optional
            Beam power in Watts.  If specified, `beam_waist`
            must also be supplied, and `rabi_frequency` and `e_field` cannot
            be specified. `beam_power` and `beam_waist` cannot be scanned simultaneously.
        beam_waist: float, optional
            1/e^2 Beam waist (radius) in units of
            meters.  Only necessary when specifying `beam_power`.
        q: int, optional
            Coupling polarization in spherical basis.
            Valid values are -1, 0, 1 for :math:`-\\sigma`, linear, :math:`+\\sigma`.
            Default is 0 for linear.

        Raises
        ------
        RydiquleError
            If `states` is not a list-like of 2 integers.
        RydiquleError
            If an invalid combination of `rabi_frequency`, `e_field`,
            `beam_power`, and `beam_waist` is provided.
        RydiquleError
            If `tranistion_frequency` is passed as an argument (it is
            calculated from atomic properties).
        RydiquleError
            If `beam_power` and `beam_waist` are both sequences.
        CouplingNotAllowedError
            If the coupling is not dipole-allowed.
        RydiquleError
            If `kvec` is supplied instead of `kunit`.

        Notes
        -----
        .. note::
            Note that while this function can be used directly just as in :class:`~.sensor.Sensor`,
            it will often be called implicitly via :meth:`~.Sensor.add_coupling` which `Cell`
            inherits. While they are equivalent, the second of these options is
            often the more clear approach, and it automatically sets the 
            `probe_tuple` attribute.
            
        .. note::
            Specifying the beam power by beam parameters or electric field still computes
            the `rabi_frequency` and adds that quantity to the `Cell` to maintain consistency across
            `rydiqule`'s other calculations.
            In other words, `beam_power`, `beam_waist`, and `e_field` will never appear
            as quantities on the graph of a `Cell`.
        
        Examples
        --------
        In the simplest case, physical properties are calculated automatically in a `Cell`
        All the familiar quantities are present, as well as many more. Note that while not 
        strictly necessary, it is often convenient to alias states with shorthand variables
        to avoid very cumbersome state specification. 

        >>> [g, e] = rq.D2_states("Rb87")
        >>> cell = rq.Cell("Rb87", [g, e])
        >>> cell.add_single_coupling((g,e), detuning=1.0, rabi_frequency=2.0, label="probe")
        >>> print(cell)
        <class 'rydiqule.cell.Cell'> object with 2 states and 1 coherent couplings.
        States: [(n=5, l=0, j=0.5), (n=5, l=1, j=1.5)]
        Coherent Couplings: 
            ((5, 0, 0.5),(5, 1, 1.5)): {rabi_frequency: 2.0, detuning: 1.0, phase: 0, kvec: (0, 0, 0), label: probe, coherent_cc: 1, dipole_moment: 2.44, q: 0}
        Decoherent Couplings:
            ((5, 1, 1.5),(5, 0, 0.5)): {gamma_transition: 38.11316}
        Energy Shifts:
            None
        
        Since :meth:`~.Sensor.add_couplings`, :meth:`~.Sensor.add_coupling`, 
        and :meth:`~.Sensor.add_coupling_group` only iterate over calls of this function,
        they do not need to be overloaded. 
        
        >>> [g, e] = rq.D2_states("Rb87")
        >>> cell = rq.Cell("Rb87", [g, e])
        >>> probe = dict(states=(g,e), detuning=1.0, rabi_frequency=2.0, label="probe")
        >>> cell.add_couplings(probe)
        >>> print(cell)
        <class 'rydiqule.cell.Cell'> object with 2 states and 1 coherent couplings.
        States: [(n=5, l=0, j=0.5), (n=5, l=1, j=1.5)]
        Coherent Couplings: 
            ((5, 0, 0.5),(5, 1, 1.5)): {rabi_frequency: 2.0, detuning: 1.0, phase: 0, kvec: (0, 0, 0), label: probe, coherent_cc: 1, dipole_moment: 2.44, q: 0}
        Decoherent Couplings:
            ((5, 1, 1.5),(5, 0, 0.5)): {gamma_transition: 38.11}
        Energy Shifts:
            None

        `e_field` can be specified instead of `rabi_frequency`,
        but a `rabi_frequency` will still be added to the system based on the `e_field`
        and computed dipole moment rather than `e_field` directly.
        
        >>> [g, e] = rq.D2_states("Rb87")
        >>> cell = rq.Cell("Rb87", [g, e])
        >>> cell.add_coupling((g,e), detuning=1.0, e_field=6.0, label="probe")
        >>> print(cell)
        <class 'rydiqule.cell.Cell'> object with 2 states and 1 coherent couplings.
        States: [(n=5, l=0, j=0.5), (n=5, l=1, j=1.5)]
        Coherent Couplings: 
            ((5, 0, 0.5),(5, 1, 1.5)): {rabi_frequency: 1.177, detuning: 1.0, phase: 0, kvec: (0, 0, 0), label: probe, coherent_cc: 1, dipole_moment: 2.44, q: 0}
        Decoherent Couplings:
            ((5, 1, 1.5),(5, 0, 0.5)): {gamma_transition: 38.11}
        Energy Shifts:
            None

        As can `beam_power` and `beam_waist`,
        with similar behavior regarding how information is stored.

        >>> cell = rq.Cell("Rb85", rq.D2_states("Rb85"), cell_length = .0001)
        >>> cell.add_coupling((g,e), detuning=1.0, beam_power=1.0, beam_waist=1.0, label="probe")
        >>> print(cell)
        <class 'rydiqule.cell.Cell'> object with 2 states and 1 coherent couplings.
        States: [(n=5, l=0, j=0.5), (n=5, l=1, j=1.5)]
        Coherent Couplings: 
            ((5, 0, 0.5),(5, 1, 1.5)): {rabi_frequency: 4.3, detuning: 1.0, phase: 0, kvec: (0, 0, 0), label: probe, coherent_cc: 1, dipole_moment: 2.44, q: 0}
        Decoherent Couplings:
            ((5, 1, 1.5),(5, 0, 0.5)): {gamma_transition: 38.11}
        Energy Shifts:
            None

        """ 
        state1 = states[0]
        state2 = states[1]
        not_nlj = (state1.stype != 'NLJ' and state2.stype != 'NLJ')

        # check that tuple energy convention matches atomic properties
        freq_diff = 2*np.pi*self.atom.get_transition_frequency(state1, state2)*1e-6
        det_sign = np.sign(freq_diff)
        if np.sign(freq_diff) != 1:
            if det_sign > 0:
                msg = ' higher energy, but it is actually lower. '
            else:
                msg = ' lower energy, but it is actually higher. '
            raise RydiquleError(f'Coupling {states} implies second state is'
                                + msg + 'Please reverse indeces of states tuple.')
        
        if 'suppress_dipole_warn' in extra_kwargs:
            warnings.warn("The 'suppress_dipole_warn' kwarg is deprecated.",
                           FutureWarning)
        
        dipole_moment = self.atom.get_dipole_matrix_element(state1, state2, q)
        if dipole_moment == 0 or dipole_moment == np.nan:
            raise CouplingNotAllowedError(f'{state1}-->{state2} is not dipole allowed!')
        if coherent_cc is None:
            if not_nlj:
                # calculate spherical dipole matrix element and reduced matrix element
                # using ARC's conventions
                sph_moment = self.atom.get_spherical_dipole_matrix_element(state1, state2, q)
                red_dipole_moment = self.atom.get_reduced_matrix_elementJ(state1, state2)
                # define coherent_cc as double the spherical moment
                # this ensures coefficients are closer to 1
                # Equivalent to defining the spherical moments as being in units of
                # reduced_matrix_element_J/2
                passed_cc = 2 * sph_moment
                dme = red_dipole_moment / 2
            else:
                # for NLJ transitions, no sublevel structure so spherical part
                # is inconsequential (all diffs in reducedJ)
                passed_cc = 1
                dme = dipole_moment
        else:
            passed_cc = coherent_cc
            dme = dipole_moment
                
        if (e_field is not None
            and beam_power is None
            and beam_waist is None
            and rabi_frequency is None
            ):
            # E-field definition of Rabi frequency
            if isinstance(e_field, Sequence):
                e_field = np.asarray(e_field)
            passed_rabi = e_field*scale_dipole(dme)

        elif (e_field is None
              and beam_power is not None
              and rabi_frequency is None
              and beam_waist is not None
              ):
            # beam power definition of Rabi frequency
            if isinstance(beam_power, Sequence) and isinstance(beam_waist, Sequence):
                raise RydiquleError('beam_power and beam_waist cannot be scanned simultaneously')
            else:
                if not_nlj:
                    # get the reduced Rabi frequency
                    passed_rabi = np.array([[1e-6*self.atom.get_reduced_rabi_frequency(state1, state2,
                                                                                       bp, bw)
                                            for bp in np.array(beam_power, ndmin=1)]
                                            for bw in np.array(beam_waist, ndmin=1)]).squeeze()
                else:
                    # for NLJ states, use the full Rabi frequency
                    passed_rabi = np.array([[1e-6*self.atom.get_rabi_frequency(state1, state2,
                                                                               q, bp, bw)
                                            for bp in np.array(beam_power, ndmin=1)]
                                            for bw in np.array(beam_waist, ndmin=1)]).squeeze()
                if passed_rabi.shape == tuple():
                    passed_rabi = float(passed_rabi)

        elif (e_field is None
              and beam_power is None
              and beam_waist is None
              and rabi_frequency is not None):
            # Rabi frequency directly provided
            # this case assumes user has correctly scaled rabi already
            passed_rabi = rabi_frequency

        else:
            msg = ("Please only define one of: 1) rabi_frequency or "
                   "2) e_field or 3) beam_power and beam_waist.")
            raise RydiquleError(msg)

        if detuning is None:
            if transition_frequency is not None:
                msg = """Cell does not support explicit definition of transition_frequency,
                it is calculated based on atomic properties."""
                raise RydiquleError(msg)
            else:
                transition_frequency = freq_diff

        if 'kvec' in extra_kwargs:
            raise RydiquleError("Cell couplings no longer accept 'kvec' as a parameter. " +
                                "Use 'kunit' instead, which is the unit propagation axis. " +
                                "Cell calculates necessary prefactors to define 'kvec' from 'kunit'.")
        # apply kvec scaling factors for use in underlying Sensor
        k_norm_sq = np.sum(np.asarray(kunit)**2)
        if np.isclose(k_norm_sq, 0.0):
            # doppler not requested for this coupling, pass default along
            kvec = kunit
        elif np.isclose(k_norm_sq, 1.0):
            # apply standard dopper shift
            lam = abs(self.atom.get_transition_wavelength(state1, state2)) # in m
            kvec = 2*np.pi/lam*np.asarray(kunit)*1e-6 # scaled to Mrad/m
        else:
            raise RydiquleError(f'Coupling {states} has un-normalized |kunit|={np.sqrt(k_norm_sq):.2f}!=1')

        super().add_single_coupling(states=states,
                                    rabi_frequency=passed_rabi,
                                    coherent_cc=passed_cc,
                                    detuning=detuning,
                                    transition_frequency=transition_frequency,
                                    phase=phase,
                                    kvec=kvec,
                                    time_dependence=time_dependence,
                                    label=label,
                                    dipole_moment=dipole_moment,
                                    q=q,
                                    **extra_kwargs)
    

    def add_transit_broadening(self, gamma_transit: ScannableParameter,
                               repop: Optional[Union[Dict[A_QState, float], List[A_QState]]] = None,
                               label: str = "transit"):
        """Overload of the :meth:`~.Sensor.add_transit_broadening` method of `Sensor` which
        automatically populates the `repop` dictionary with a equal decay to all sublevels of
        the ground `(n, l, j)` state. 

        Parameters
        ----------
        gamma_transit : ScannableParameter
            Transit brodening of the system. Passed transparently to super function.
        repop: dict, optional
            Dictionary of states for transit to repopulate in to.
            The keys represent tshe state labels. The values represent
            the fractional amount that goes to that state.
            If the sum of value does not equal 1, population will not be conserved.
            If `None`, raises population decay due to transit broadening will be evenly
            divided amongst all states matching the `(n, l, j)` quantum numbers of the
            lowest-energy state in the system. Defaults to `None`. 
        label: str, optional
            Label to be passed to :meth:`~.Sensor.add_decoherence`. Defaults to "transit"

        Examples
        --------
        >>> atom = "Rb85"
        >>> g = rq.ground_state(atom, splitting="fs")
        >>> e = rq.D1_excited(atom, splitting="fs")
        >>> Rb_Cell = rq.Cell(atom, [g,e])
        >>> Rb_Cell.add_transit_broadening(0.1)
        >>> for e in Rb_Cell.couplings.edges.data("gamma_transit"):
        ...     print(e)
        ((n=5, l=0, j=0.5, m_j=-0.5), (n=5, l=0, j=0.5, m_j=-0.5), 0.05)
        ((n=5, l=0, j=0.5, m_j=-0.5), (n=5, l=0, j=0.5, m_j=0.5), 0.05)
        ((n=5, l=0, j=0.5, m_j=0.5), (n=5, l=0, j=0.5, m_j=-0.5), 0.05)
        ((n=5, l=0, j=0.5, m_j=0.5), (n=5, l=0, j=0.5, m_j=0.5), 0.05)
        ((n=5, l=1, j=0.5, m_j=-0.5), (n=5, l=0, j=0.5, m_j=-0.5), 0.05)
        ((n=5, l=1, j=0.5, m_j=-0.5), (n=5, l=0, j=0.5, m_j=0.5), 0.05)
        ((n=5, l=1, j=0.5, m_j=0.5), (n=5, l=0, j=0.5, m_j=-0.5), 0.05)
        ((n=5, l=1, j=0.5, m_j=0.5), (n=5, l=0, j=0.5, m_j=0.5), 0.05)

        """
        if repop is None:
            g = self.level_ordering()[0]
            
            #construct the ground state manifold
            m_j = None if g.m_j is None else "all"
            (f, m_f) = (None, None) if g.f is None else ("all","all")
            ground_manifold = A_QState(g.n, g.l, g.j, m_j=m_j, f=f, m_f=m_f)

            ground_states_all = self.states_with_spec(ground_manifold)
            repop = {state: 1/len(ground_states_all) for state in ground_states_all}

        super().add_transit_broadening(gamma_transit=gamma_transit, repop=repop, label=label)


    def states_with_spec(self, statespec: A_QState) -> List[A_QState]:
        """Return a list of all states in the sensor matching the `state_spec` pattern.

        Matching is determined by same rules as :meth:`~.Sensor.states_with_spec`, with no
        additional logic to account for the different typing of the states. This means that
        there is expansion of any quantum numbers specified by the "all" keyword, and only
        states that are already nodes of the `Cell` graph will be included.

        Parameters
        ----------
        statespec : A_QState
            State specification againt which to perform matching,

        Returns
        -------
        List[A_QState]
            List of all states in :class:`~.Cell` instance which match the provided specification.

        Examples
        --------
        >>> atom = "Rb85"
        >>> g = rq.ground_state(atom, splitting="fs")
        >>> e = rq.D1_excited(atom, splitting="fs")
        >>> Rb_Cell = rq.Cell(atom, [g,e])
        >>> print(Rb_Cell.states_with_spec(A_QState(5, 0, 0.5)))
        []
        >>> print(Rb_Cell.states_with_spec(A_QState(5, 0, 0.5, m_j="all")))
        [(n=5, l=0, j=0.5, m_j=-0.5), (n=5, l=0, j=0.5, m_j=0.5)]
        >>> print(Rb_Cell.states_with_spec(A_QState(5, 0, 0.5, m_j=[-0.5, 0.5])))
        [(n=5, l=0, j=0.5, m_j=-0.5), (n=5, l=0, j=0.5, m_j=0.5)]
        
        """
        return super().states_with_spec(statespec)
    

    def _add_state_energies(self):
        """
        Helper function to add all the "energy" key to all the nodes of the graph for the state
        energies relative to the first state.

        All energies are relative to the ground state, defined as the :math:`nS^{1/2}` state, where
        n is the principle quantum number of the lowest energy atomic state.
        """
        g_state = ground_state(self.atom_flag)
        ground_energy = self.atom.get_state_energy(g_state)*1e-6*2.0*np.pi

        for state in self.couplings.nodes:
            state_energy = self.atom.get_state_energy(state)*1e-6*2.0*np.pi #Mrad/s
            self.couplings.nodes[state]["energy"] = state_energy - ground_energy


    def _add_state_lifetimes(self):
        """
        Helper function to add all "gamma_lifetime" key-value appropriate to the atom to all nodes 
        on the graph.
        """

        for i, state in enumerate(self.couplings.nodes):
            if i==0:
                self.couplings.nodes[state]["gamma_lifetime"] = 0.0
            else:
                self.couplings.nodes[state]["gamma_lifetime"] = 1e-6/self.atom.get_state_lifetime(state)

    def _add_decoherence_rates(self):
        """
        Helper function to add natural decay rates to all transitions in the cell. Values
        for decay rates are calculated with arc, and skipped if selection rules prohibit the
        transition or if the second state is a higher energy than the first state
        """
        
        for s1, s2 in itertools.product(self.couplings.nodes, self.couplings.nodes):
            #states don't decay to lower energies
            if self.couplings.nodes[s1]["energy"] <= self.couplings.nodes[s2]["energy"]:
                continue
            else:
                try:
                    gamma=self.atom.get_transition_rate(s1, s2)/1e6
                except ValueError:
                    continue
                self.add_decoherence((s1,s2), gamma, label="transition")


    def _add_gamma_mismatches(self, method:Union[str, dict]="ground"):
        """Adds couplings to the graph accounting for differences between computed 
        lifetimes and decay rates. 

        In a cell in which all atomic states are accounted for, the computed values of 
        `gamma_lifetime` for a particular state will be equal to the sum of all 
        `gamma_transition` values on edges leaving that state. However, it is not always
        desirable to account for all states in this way for simplicity or computational
        complexity reasons. This function allows the :class:`~.Cell` to account for any
        differences in these values that arise as a result of excluding physicalstates from a
        :class:`~.Cell`. There are multiple ways to resolve these discrepancies, specified by
        the `method` argument, which is detailed in the `Parameters` section.


        Parameters
        ----------
        method : str or dict mapping states to str
            The method by which discrepancies in computed values are resolved. The available
            methods are as follows:
                
                - `"ground"` which adds a decoherent coupling coupling between a state `s` with
                  a discrepancy :math:`\\Delta \\gamma` and divides :math:`\\Delta \\gamma` among 
                  all the ground states (states matching the `n,l,j` values of the lowest energy
                  state). 
                - `"all"` which divides :math:`\\Delta \\gamma` amongst all states in the 
                  :class:`~.Cell` which already have a `"gamma_transition"` value. The fraction
                  each transition gets is weighted by the fraction of the total thdat transitions
                  `"gamma_transition"` value accounts for. If this method is used, every state in
                  the `Cell` must have at least one dipole-allowed decay path. 
                - `"none"` which will not account for this discrepancy at all. In this case this
                  physics is not guaranteed to be accurate and it is assumed the decoherence will
                  be accounted for manually in other ways using :meth:`"~.add_decoherence"`
            
            Note that in addition to one of these strings `method` can be a dictionary which
            maps states to one of these strings. In this case, the discrepancy is resolved
            separately for each state using the method specified for that state. Defaults to "ground"

        Raises
        ------
        ValueError
            If the method is not one of the allowed strings or a dictionary mapping states to
            one of the allowed strings.
        RydiquleError
            If the "all" option is selected for `method` and any of the states have no lower
            state to decay to.
        """
      
        mismatch_fns: Dict[str, Callable] = {
            "ground": self._add_gamma_mismatch_to_ground,
            "all": self._add_gamma_mismatch_to_all,
            "none": lambda *x, **y: None
        }
        
        if method in mismatch_fns.keys():
            methods_all = {s:method for s in self.couplings.nodes()}
        elif isinstance(method, dict):
            methods_all = method
        else:
            msg = f"'method' must be one of {mismatch_fns.keys()} or "\
                "a dictionary mapping states to one of those values"
            raise ValueError(msg)
 

        for state in self.couplings.nodes():

            meth_i = methods_all.get(state, "none")
            try:
                fn_i = mismatch_fns[meth_i]
            except KeyError:
                msg = f"'method' must be one of {mismatch_fns.keys()} or "\
                    "a dictionary mapping states to one of those values"
                raise ValueError(msg)

            fn_i(state)


    def _add_gamma_mismatch_to_ground(self, state: A_QState):
        """Helper function which implements the "ground" option of :meth:`_add_gamma_mismatches`
        for a single state `state`.
        
        """
        try:
            lifetime = self.couplings.nodes[state]["gamma_lifetime"]
        except KeyError:
            raise KeyError(f"State {state} is not a state of cell.")
        
        out_edges = self.couplings.out_edges(state, data="gamma_transition")
        transition_total = sum([e[2] for e in out_edges if e[2]])
        
        #if they dom't match, we add a decoherence to the entire ground state manifold
        #that matches what remains
        if not np.isclose(transition_total, lifetime):
            g = self.level_ordering()[0]
            
            #construct the ground state manifold
            m_j = None if g.m_j is None else "all"
            (f, m_f) = (None, None) if g.f is None else ("all","all")
            ground_manifold = A_QState(g.n, g.l, g.j, m_j=m_j, f=f, m_f=m_f)

            self.add_decoherence((state, ground_manifold), gamma=(lifetime-transition_total), label="mismatch")

    
    def _add_gamma_mismatch_to_all(self, state:A_QState):
        """Helper function which implements the "all" option of :meth:`_add_gamma_mismatches`
        for a single state `state`.
        
        """
        try:
            lifetime = self.couplings.nodes[state]["gamma_lifetime"]
        except KeyError:
            raise KeyError(f"State {state} is not a state of cell.")
        
        out_edges = self.couplings.out_edges(state, data="gamma_transition")

        ground_nlj = ground_state(self.atom_flag)[:3]

        if len(out_edges) == 0 and state[:3] != ground_nlj:
            msg = "'all' option selected for gamma_mismatch handling but no dipole-allowed "\
                f"states have been been added for tranistion out of {state}"
            raise RydiquleError(msg)

        transition_total = sum(e[2] for e in out_edges)
        
        #if they dom't match, we add a decoherence to the entire ground state manifold
        #that matches what remains
        if not np.isclose(transition_total, lifetime):
            #construct the dictionary of coupling coefficients coefficients
            cc = {
                (s1, s2):gamma/transition_total for s1, s2, gamma in out_edges
                if gamma
            }

            out_states_list = [s2 for _, s2, _ in out_edges]
            self.add_decoherence_group([state], out_states_list, gamma = transition_total, coupling_coefficients=cc, label="mismatch")


    def _validate_input_states(self, atomic_states: List[A_QState]):
        """Helper function to check that input states are compatible and defined"""
        
        for state in atomic_states:
            validate_qnums(state, I=self.I)
            
