"""
Physical Cell objects for use in solvers.
"""

import scipy
import numpy as np
import warnings

import scipy.constants
from scipy.constants import Boltzmann, e

# rydiqule imports
from .sensor import Sensor
from .sensor_utils import scale_dipole
from .sensor_utils import ScannableParameter, QState, States
from .atom_utils import ATOMS, calc_kappa, calc_eta

from typing import Literal, Optional, Sequence, List, Tuple, Callable, Union

a0 = scipy.constants.physical_constants["Bohr radius"][0]

AtomFlags = Literal['H', 'Li6', 'Li7', 'Na', 'K39', 'K40', 'K41', 'Rb85', 'Rb87', 'Cs']


class Cell(Sensor):
    """
    Subclass of :class:`~.Sensor` that creates a Sensor with additional physical properties
    corresponding to a specific Rydberg atom.

    In addition to the core functionality of `~.Sensor`, this class allows for labelling of states
    with quantum numbers, calculating of state lifetimes and decoherences and tracking of
    of some physical laser parameters.
    A key distictinction between a :class:`~.Cell` and a :class:`~.Sensor` is that
    a cell supports (and requires) and absolute ordering of energy between states,
    which allows for implicit calculation of decay rates an transition frequencies.

    """

    def __init__(self, atom_flag: AtomFlags, *atomic_states: QState,
                 cell_length: float = 1e-3, gamma_transit: Optional[float] = None,
                 beam_area: float = 1e-6, beam_diam: Optional[float] = None,
                 temp: float = 300.0, probe_tuple: tuple = (0,1)) -> None:
        """
        Initialize the Rydberg cell from the given parameters. 

        Parameters
        ----------
        atom_flag : str 
            Which atom is used in the cell for calculating physical properties with ARC Rydberg.
            One of ['H', 'Li6', 'Li7', 'Na', 'K39', 'K40', 'K41', 'Rb85', 'Rb87', 'Cs'].
        atomic_states : list[list]
            List of states to be added to the cell. Each state is 
            an iterable whose elements are each a list of the form [n, l, j, m], represnting the 
            Rydberg atomic quantum numbers of the state. At least two states must be added so 
            that the system is nontrivial. The number of states will determine the basis size
            of the system. Note that the first state in the list is assumed to be a meta-stable
            ground.
        cell_length: float
            Length of the atomic vapor in meters.
        gamma_transit : float, optional
            Decoherence due to atom transit through the optical
            beams. Specified in units of Mrad/s. If `None`, will calculate based
            on value of `beam_area`. See :meth:`~.Sensor.add_transit_broadening` for
            details on how transit broadening is treated. Default is None.
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
        ValueError
            If at least two atomic states are not provided.
        ValueError
            If atom_flag is not one of ARC's supported alkali atoms.

        """
        if atom_flag not in ATOMS.keys():
            raise ValueError(f"Atom flag must be one of {ATOMS.keys()}")

        if len(atomic_states) < 2:
            raise ValueError(("No states added to system. "
                             "Specify at least two states of the form [n, l, j, m]"))

        self.atom_flag = atom_flag
        self.atom = ATOMS[atom_flag]()

        self.cell_length = cell_length  # Default 0 to do optically thin sample
        self.temp = temp  # K
        self.beam_area = beam_area
        self.density = self.atom.getNumberDensity(self.temp)
        self.atom_mass = self.atom.mass

        if beam_diam is None:
            beam_diam = 2.0*np.sqrt(beam_area/np.pi)

        self.beam_diameter = beam_diam

        if gamma_transit is None:
            gamma_transit = 1E-6*np.sqrt(8*Boltzmann*self.temp/(self.atom_mass*np.pi)
                                         )/(self.beam_diameter/2*np.sqrt(2*np.log(2)))

        super().__init__(0)

        self._add_states(*atomic_states)

        self.probe_tuple = self._states_valid(probe_tuple)

        self.add_transit_broadening(gamma_transit)

    
    def set_experiment_values(self, probe_tuple: Tuple[int,int],
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
        UserWarning: Warns if function is used.
        """

        warnings.warn(UserWarning('set_experiment_values not used with Cell'))

    def add_states(self):
        """
        Deprecated.
        
        Use \"atomic_state\" keyword argument of the constructor instead.
        """
        raise NotImplementedError(("Adding states after a cell has been created is "
                                   "no longer supported. Please specify states with "
                                   "the \"atomic_states\" keyword argument in the constructor."))


    def _add_state(self, state: QState) -> None:
        """Internal method to add a single state to the Cell.

        Should only be used by the :meth:`~.Cell._add_states` function, but can be overloaded
        in custom implementations of :class:`~.Cell`

        Parameters
        ----------
        state: int or list of ints 
            The quantum numbers of the state to be added. Should be either length
            4 of form [n, l, j, m] or length 5 of form [n, l, j, F, mF]
        """

        #add node and get index of current node
        self.couplings.add_node(self.basis_size)
        n = self.basis_size - 1 

        state = self._validate_qnums(state)

        #add quantum numbers to node
        self.couplings.nodes[n]["qnums"] = state

        #compute hyperfine splitting if relevant
        if len(state) == 5:
            self.couplings.nodes[n]["hfs"] = self.get_cell_hfs_shift(state)

        #add state energy to node
        if self.basis_size == 1:
            energy = 0 
        else:
            energy = self.get_cell_tansition_frequency(0, state)*1e-6*2.0*np.pi
        self.couplings.nodes[n]["energy"] = energy

        #add state lifetime to node
        if self.basis_size == 1:
            gamma_lifetime = 0
        else:
            gamma_lifetime = 1E-6/self.atom.getStateLifetime(*state[0:3])
        self.couplings.nodes[n]["gamma_lifetime"] = gamma_lifetime


    def _add_states(self, *states: QState) -> None:
        """
        Internal method to add states to the system and update internal variables
        for energy and decay rates. 
        
        Quantum numbers and other states information are stored on the couplings graph nodes.
        The first state added to the system is treated as the ground state,
        and all "absolute energies" are calculated as a difference from ground.
        Should only be called in :meth:`~.Cell.__init__`.

        Parameters
        ----------
        *states : list[list[int or float]]
            States that are added to the list of atomic states of interest for the cell.
            Arguments should be lists of the form `[n, l, j, m]`, where n, l, j, and m
            are the ususal quantum numbers describing the state: principal,
            orbital, total angular momentum, and magnetic quantum numbers respectively.

        """
        for state in states:

            self._add_state(state)

        self._add_decay_to_graph()


    def states_list(self) -> List[QState]:
        """
        Returns a list of quantum numbers for all states in the cell.

        States position in the list will be in the order they were added,
        and correspond to the density matrix values numbered with the same index.

        Returns
        -------
        list[list]
            List of quantum states of the form [n, l, j, m] that are stored
            on graph nodes in the cell.

        Examples
        --------
        >>> state1 = [50, 2, 2.5, 2.5] # states are written with quantum numbers [n, l, j, m] 
        >>> state2 = [51, 2, 2.5, 2.5]
        >>> cell = rq.Cell("Rb85", *rq.D2_states(5), state1, state2,
        >>> cell_length = .0001) #D2 states gets the states for the Rubidium 85 D2 line
        >>> print(cell.states_list())
        [[5, 0, 0.5, 0.5], [5, 1, 1.5, 0.5], [50, 2, 2.5, 2.5], [51, 2, 2.5, 2.5]]

        """ 
        return [state[1] for state in self.couplings.nodes("qnums")]


    def level_ordering(self) -> List[int]:
        """
        Return a list of the integer numbers of each state (*not* the quantum numbers) 
        in descending order by energy order.

        All energies are calculated with respect to the ground state energy, which is defined as 0.
        Ground state is determined by the first state added to the system (state 0). Thus, state 0 
        will always be last in the list.

        Returns
        -------
        list[int]
            The level numbers of the states in order of decending energy
            relative to the ground state (state 0).

        Examples
        --------
        For the following example, states are added to the cell in ascending
        energy order, so the return reflects that, with the highest-energy state first.

        >>> state1 = [50, 2, 2.5, 2.5]
        >>> state2 = [51, 2, 2.5, 2.5]
        >>> cell = rq.Cell("Rb85", *rq.D2_states("Rb85"), state1, state2,
        >>> cell_length = .00001) #uses the D2 line of Rb85
        >>> print(cell.states_list())
        >>> print(cell.level_ordering())
        [[5, 0, 0.5, 0.5], [5, 1, 1.5, 0.5], [50, 2, 2.5, 2.5], [51, 2, 2.5, 2.5]]
        [3, 2, 1, 0]

        If we add `state1` and `state2` in the opposite order (thus switching
        their positions in the states list), the level ordering will change since
        `state2` is still a higher energy.

        >>> state1 = [50, 2, 2.5, 2.5]
        >>> state2 = [51, 2, 2.5, 2.5]
        >>> cell = rq.Cell("Rb85", *rq.D2_states("Rb85"), state2, state1) #uses the D2 line of Rb85
        >>> print(cell.states_list())
        >>> print(cell.level_ordering())
        [[5, 0, 0.5, 0.5], [5, 1, 1.5, 0.5], [51, 2, 2.5, 2.5], [50, 2, 2.5, 2.5]]
        [2, 3, 1, 0]

        """
        energies = list(self.couplings.nodes("energy")).copy()
        energies.sort(key=lambda val:val[1],reverse=True)
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
        
        Notes
        -----
        There is no way to set the `kappa` attribute directly at presence, it is always
        inferred with the above formula. However, this functionality exists so that
        custom implementations or specific use cases of :class:`~.Cell` may set it
        for their own purposes.    

        Returns
        -------
        float
            The value kappa for the system. 
        """ 

        #does nothing now, possible future-proofing
        if hasattr(self, "_kappa"):
            return self._kappa
        
        omega_rad = 1e6*self.couplings.nodes[1]["energy"] #convert from Mrad to rad
        dipole_moment = self.couplings.edges[0,1]["dipole_moment"]*e*a0 #to C*m

        kappa = calc_kappa(omega_rad, dipole_moment, self.density)
    
        return kappa


    @kappa.setter
    def kappa(self, value: float):
        """Setter for the kappa attribute.

        Updates the self._kappa class attribute.

        Parameters
        ----------
        value : float
            The floating-point value to set as the eta parameter for the system.
        """
        self._kappa = value


    @kappa.deleter
    def kappa(self):
        """Setter for the kappa attribute.

        Removes the self._kappa class attribute.

        Raises
        ------
        AttributeError:
            If kappa has not been set.
        """
        try:
            del self._kappa
        except AttributeError:
            raise AttributeError("The \"kappa\" attribute has not been set")
    

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
        
        Notes
        -----
        There is no way to set the `eta` attribute directly at present, it is always
        inferred with the above formula. However, this functionality exists so that
        custom implementations or specific use cases of :class:`~.Cell` may set it
        for their own purposes.   

        Returns
        -------
        float
            The value eta for the system.
        """
        #does nothing now, possible future-proofing
        if hasattr(self, "_eta"):
            return self._eta
        
        omega_rad = 1e6*self.couplings.nodes[1]["energy"]
        dipole_moment = self.couplings.edges[0,1]["dipole_moment"]*a0*e
        
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
        AttributeError:
            If eta has not been set.
        """
        try:
            del self._eta
        except AttributeError:
            raise AttributeError("The \"eta\" attribute has not been set")


    @property
    def probe_freq(self):
        """Get the probe transition frequency, in rad/s
        
        Returns
        -------
        float
            Probe transitiion frequency, in rad/s
        """

        if hasattr(self, '_probe_freq'):
            return self._probe_freq
        
        energies = self.couplings.nodes("energy")
        
        return np.abs(energies[self.probe_tuple[1]] - energies[self.probe_tuple[0]])*1e6
        

    def decoherence_matrix(self) -> np.ndarray:
        """
        Get the decoherence matrix for a system.

        This overload differs from :meth:`~Sensor.decoherence_matrix`
        by including state lifetimes and decay rates calculated from arc rydberg without any
        explicit definition of decoherence terms. In other words, it ensures that the
        calculated decoherence terms match what is expected for a particular
        real-world atom. 
        
        Returns
        -------
        numpy.ndarray
            The decoherence matrix stack of the system.

        """
        # start with base gamma matrix from the transitions on graph
        gamma = super().decoherence_matrix()

        for state, lifetime in list(self.couplings.nodes(data="gamma_lifetime"))[1:]:

            decay_sum = 0.0

            for edge in self.couplings.out_edges(state):
                decay_sum += self.couplings.edges[edge].get("gamma_transition", 0)

            gamma[..., state, 0] += lifetime - decay_sum

        return gamma


    def _add_decay_to_graph(self) -> None:
        """
        Internal helper method to add population decay rates to the nodes to calculate gamma matrix.

        1. add the state lifetime to each node
        2. add the transition rate to each edge
        
        """
        # First going to get absolute lifetimes of each state
        states = self.level_ordering()

        for index,state in enumerate(states):

            qnum = self.couplings.nodes[state]["qnums"]

            # Starts at 1, for the ground state, we want to avoid calculating decay
            # loop over all couplings to get transition rates
            for lower_state in states[index+1:]:
                qnum_lower = self.couplings.nodes[lower_state]["qnums"]
                try:
                    gamma_transition = 1E-6*self.atom.getTransitionRate(*qnum[0:3],
                                                                        *qnum_lower[0:3])
                except ValueError:
                    gamma_transition = 0

                if gamma_transition != 0:
                    self.add_decoherence((state, lower_state), gamma_transition, label="transition")


    def add_coupling(
            self, states: States, rabi_frequency: Optional[ScannableParameter] = None,
            detuning: Optional[ScannableParameter] = None,
            transition_frequency: Optional[float] = None,
            phase: ScannableParameter = 0, kvec: Tuple[float,float,float] = (0,0,0),
            time_dependence: Optional[Callable[[float],float]] = None, label: Optional[str] = None,
            e_field: Optional[ScannableParameter] = None, beam_power: Optional[float] = None,
            beam_waist: Optional[float] = None,
            q: Literal[-1, 0, 1] = 0,
            **extra_kwargs) -> None:
        """
        Overload of :meth:`~.Sensor.add_coupling` which allows for different specification of
        coupling fields which are more reflective of real experimental setups.

        Rabi frequency is a mandatory argument in :class:`~.Sensor` but in :class:`~.Cell`, 
        there are 3 options for laser power specification:

        1. Explicit rabi-frequency definition identical to :class:`~.Sensor`.
        2. Electric field strength, in V/m.
        3. Specification of both beam power and beam waist.

        Any one of these options can be used in place of the standard`rabi_frequency` argument
        of :meth:`~.Sensor.add_coupling`.

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
        phase: float, optional
            The relative phase of the field in radians.
            Defaults to zero.
        kvec: sequence, optional
            A three-element iterable that defines the
            atomic doppler shift on a particular coupling field.
            It should have magntiude equal to the doppler shift (in the units of Mrad/s)
            of an atom moving at the Maxwell-Boltzmann distribution most probable
            speed, `vP=np.sqrt(2*kB*T/m)`. I.E. `np.linalg.norm(kvec)=2*np.pi/lambda*vP`.
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
        ValueError
            If `states` is not a list-like of 2 integers.
        ValueError
            If an invalid combination of `rabi_frequency`, `e_field`,
            `beam_power`, and `beam_waist` is provided.
        ValueError
            If `tranistion_frequency` is passed as an argument (it is
            calculated from atomic properties).
        ValueError
            If `beam_power` and `beam_waist` are both sequences.

        Warns
        -----
        UserWarning
            If any coupling has time-dependence specified,
            which `get_snr` cannot currently handle as it is steady-state only.

        Notes
        -----
        .. note::
            Note that while this function can be used directly just as in :class:`~.sensor.Sensor`,
            it will often be called implicitly via :meth:`~.Sensor.add_couplings` which `Cell`
            inherits. While they are equivalent, the second of these options is
            often the more clear approach.
            
        .. note::
            Specifying the beam power by beam parameters or electric field still computes
            the `rabi_frequency` and adds that quantity to the `Cell` to maintain consistency across
            `rydiqule`'s other calculations.
            In other words, `beam_power`, `beam_waist`, and `e_field` will never appear
            as quantities on the graph of a `Cell`.
        
        Examples
        --------
        In the simplest case, physical properties are calculated automatically in a `Cell`
        All the familiar quantities are present, as well as many more.

        >>> cell = rq.Cell("Rb85", *rq.D2_states("Rb85"), cell_length = .0001)
        >>> cell.add_coupling(states=(0,1), detuning=1, rabi_frequency=2)
        >>> print(dict(cell.couplings.edges))
        {(0, 0): {'gamma_transit': 0.41172855461658464, 'label': '(0,0)'}, 
        (0, 1): {'rabi_frequency': 2, 'detuning': 1, 
        'phase': 0, 'kvec': (0, 0, 0), 'label': '(0,1)'}, 
        (1, 0): {'gamma_transition': 37.829349995476726, 
        'label': '(1,0)', 'gamma_transit': 0.41172855461658464}}
        
        Here we see implicitly calling this overloaded 
        function through :meth:`~.Sensor.add_couplings`.
        
        >>> cell = rq.Cell("Rb85", *rq.D2_states("Rb85"), cell_length = .0001)
        >>> c = {"states":(0,1), "detuning":1, "rabi_frequency":2}
        >>> cell.add_couplings(c)
        >>> print(dict(cell.couplings.edges))
        {(0, 0): {'gamma_transit': 0.41172855461658464, 
        'label': '(0,0)'}, (0, 1): {'rabi_frequency': 2, 'detuning': 1, 
        'phase': 0, 'kvec': (0, 0, 0), 
        'label': '(0,1)'}, (1, 0): {'gamma_transition': 37.829349995476726, 
        'label': '(1,0)', 'gamma_transit': 0.41172855461658464}}

        `e_field` can be specified in stead of `rabi_frequency`,
        but a `rabi_frequency` will still be added to the system based on the `e_field`,
        rather than `e_field` directly.
        
        >>> cell = rq.Cell("Rb85", *rq.D2_states("Rb85"), cell_length = .0001)
        >>> c = {"states":(0,1), "detuning":1, "e_field":6}
        >>> cell.add_couplings(c)
        >>> print(cell.couplings.edges(data='e_field'))
        >>> print(cell.couplings.edges(data='rabi_frequency'))
        [(0, 0, None), (0, 1, None), (1, 0, None)]
        [(0, 0, None), (0, 1, -1.172912676105507), (1, 0, None)]
        
        As can `beam_power` and `beam_waist`,
        with similar behavior regarding how information is stored.

        >>> cell = rq.Cell("Rb85", *rq.D2_states("Rb85"), cell_length = .0001)
        >>> c = {"states":(0,1), "detuning":1, "beam_power":1, "beam_waist":1}
        >>> cell.add_couplings(c)
        >>> print(cell.couplings.edges(data='beam_power'))
        >>> print(cell.couplings.edges(data='rabi_frequency'))
        [(0, 0, None), (0, 1, None), (1, 0, None)]
        [(0, 0, None), (0, 1, 4.28138982322625), (1, 0, None)]

        """ 
        states = self._states_valid(states)
        state1 = self.states_list()[states[0]]
        state2 = self.states_list()[states[1]]

        # check that tuple energy convention matches atomic properties
        freq_diff = 2*np.pi*self.get_cell_tansition_frequency(state1, state2)*1e-6
        det_sign = np.sign(states[1]-states[0])
        if np.sign(freq_diff) != 1:
            if det_sign > 0:
                msg = ' higher energy, but it is actually lower. '
            else:
                msg = ' lower energy, but it is actually higher. '
            raise ValueError(f'Coupling {states} implies second state is'
                             + msg + 'Please reverse indeces of states tuple.')
        
        suppress_dipole_warn = extra_kwargs.pop("suppress_dipole_warn", False)
        
        try:
            dipole_moment = self.get_cell_dipole_moment(state1, state2, q=q)
            # ARC>=3.4 no longer gives error here, raise manually
            if dipole_moment == 0 or dipole_moment == np.nan:
                raise ValueError
        except ValueError:
            msg = (f"Transition between states {state1} and {state2} not electric-dipole allowed "
                   f"for q={q:d} polarization. Solutions may be innacurate. "
                   "Suppress this by passing \"suppress_dipole_warn=True\" to Cell.add_coupling")
            dipole_moment = 0
            if not suppress_dipole_warn:
                warnings.warn(msg)
        
        if isinstance(e_field, list):
            e_field = np.array(e_field)

        if (e_field is not None
            and beam_power is None
            and beam_waist is None
            and rabi_frequency is None
            ):
            passed_rabi = e_field*scale_dipole(dipole_moment)

        elif (e_field is None
              and beam_power is not None
              and rabi_frequency is None
              and beam_waist is not None
              ):
            if isinstance(beam_power, Sequence) and isinstance(beam_waist, Sequence):
                raise ValueError('beam_power and beam_waist cannot be scanned simultaneously')
            else:
                passed_rabi = np.array([[1e-6*self.atom.getRabiFrequency(*state1, *state2[:-1],
                                                                         q, bp, bw)
                                        for bp in np.array(beam_power, ndmin=1)]
                                        for bw in np.array(beam_waist, ndmin=1)]).squeeze()
                if passed_rabi.shape == tuple():
                    passed_rabi = float(passed_rabi)

        elif (e_field is None and beam_power is None and rabi_frequency is not None):
            passed_rabi = rabi_frequency

        else:
            msg = ("Please only define one of: 1) rabi_frequency or "
                   "2) e_field or 3) beam_power and beam_waist.")
            raise ValueError(msg)

        if detuning is None:
            if transition_frequency is not None:
                msg = """Cell does not support explicit definition of transition_frequency,
                it is calculated based on atomic properties."""
                raise ValueError(msg)
            else:
                transition_frequency = freq_diff

        super().add_coupling(states=states, rabi_frequency=passed_rabi,
                             detuning=detuning, transition_frequency=transition_frequency,
                             phase=phase, kvec=kvec, time_dependence=time_dependence, label=label,
                             dipole_moment=dipole_moment, **extra_kwargs)


    def _validate_qnums(self, qnums: QState):
        """Validate the quantum numbers provided are appropriately formated and cast to a list.
        
        States should either be 4 quantum numbers [n, l, j, m_j] for a pure electron angular
        momentum state or 5 quantum numbers [n, l, j, F, m_F] for a hyperfine state. 

        Parameters
        ----------
        qnums : list(float)
            Quantum numbers for the state.

        Raises
        ------
        ValueError
            If the list of quantum numbers is not length 4 or 5.
        """
        if len(qnums) not in [4,5]:
            raise ValueError("States must either be of length 4 or 5.")
        return list(qnums)


    def get_cell_hfs_coefficient(self, state: Union[QState, int]):
        """Get the hyperfine splitting coefficients of the given state using ARC.
        
        State can either be given as a list of quantum numbers or an integer value
        corresponding to a key in the basis of the Cell. Returns 0 if not a hyperfine
        state and the values A, B returned by ARC Rydberg's `getHFSCoefficients()`
        function if it is a hyperfine state.

        Parameters
        ----------
        state : Union[QState, int]
            The state to calculate. Can either be an integer corresponding to a graph
            node or the quantum numbers of a state in the system.

        Returns
        -------
        float
            The hyperfine splitting coeffiecient A.
        float
            The hyperfine splitting coeffiecient B.
        """
        state = self.get_qnums(state)
        
        if len(state) == 4:
            return 0, 0
        
        elif len(state) == 5:
            return self.atom.getHFSCoefficients(*state[0:3])


    def get_cell_hfs_shift(self, state: Union[QState, int]):
        """Return the hyperfine energy shift for the given hyperfine state.
        
        Returns the energy shift if `state` is a hyperfine state (defined with
        5 quantum numbers) or 0 if the `state` is not a hyperfine state
        (defined by 4 quantum numbers)

        Parameters
        ----------
        state : Union[QState, int]
            The state for which to calculate the hyperfine shift. Can be a list
            of quantum numbers or an integer state level of `Cell`

        Returns
        -------
        float
            The hyperfine energy shift in units of Mrad/s.
        """
        state = self.get_qnums(state)
        
        A, B = self.get_cell_hfs_coefficient(state)
        return self.atom.getHFSEnergyShift(state[2], state[3], A, B)
    
    
    def get_cell_tansition_frequency(self, state1: Union[QState, int], state2: Union[QState, int]):
        """Get the transition frequency between 2 states, accounting for hyperfine splitting. 
        
        If either state is a hyperfine state, its associated hyperfine shift is added or
        or subtracted to calculate the total energy difference between 2 states

        Parameters
        ----------
        state1 : Union[QState, int]
            The state the electron is transitioning from. Either a integer of one of the basis
            states, or a list of quantum numbers.
        state2 : Union[QState, int]
            The state the electron is transitioning to. Either an integer of one of the basis
            states, or a list of quantum numbers.

        Returns
        -------
        float
            The total energy difference between two states accounting for hyperfine splitting.
        """
        state1 = self.get_state_num(state1)
        state2 = self.get_state_num(state2)
        
        #get the hyperfine splitting for states
        E_hfs_1 = self.couplings.nodes[state1].get("hfs", 0)
        E_hfs_2 = self.couplings.nodes[state2].get("hfs", 0)

        qnums1 = self.couplings.nodes[state1]["qnums"]
        qnums2 = self.couplings.nodes[state2]["qnums"]


        E_base = self.atom.getTransitionFrequency(*qnums1[:3], *qnums2[:3])
        
        return E_base - E_hfs_1 + E_hfs_2
    
    
    def get_cell_dipole_moment(self, state1: Union[QState, int], state2: Union[QState, int], q=0):
        """Get the diploe moment between 2 cell state.
        
        Either both states must be hyperfine, or both states must not be hyperfine states.
        Currently dipole moments cannot be calculated for a mix of state types.

        Parameters
        ----------
        state1 : Union[QState, int]
            The state from which the electron is transitioning.
        state2 : Union[QState, int]
            The state to which the electron is transitioning.
        q : int, optional
            Polarization of probing optical field in spherical basis. Must be -1, 0, 1.
            Defaults to 0 for linear polarization.

        Returns
        -------
        float
            The dipole moment of the transition in units of e*a_0.

        Raises
        ------
        ValueError
            If both states are not of the same type.
        """
        state1 = self.get_qnums(state1)
        state2 = self.get_qnums(state2)
        #both states validated to be length 4 or 5
        
        #Handle the cases when both states are of the same type        
        if len(state1) == len(state2):
            #hfs case
            if len(state1) == 5:
                return self.atom.getDipoleMatrixElementHFS(*state1, *state2, q)
            #fs case
            elif len(state1) == 4:
                return self.atom.getDipoleMatrixElement(*state1, *state2, q)

        #calculate between hfs and fs states
        if len(state1) > len(state2):
            return self.atom.getDipoleMatrixElementHFStoFS(*state1, *state2, q)
        else:
            return -1*self.atom.getDipoleMatrixElementHFStoFS(*state2, *state1, q)
    
    
    def get_state_num(self, state: Union[QState, int]):
        """Return the integer number in the bases of a node with the given quantum numbers.

        Quantum numbers for a state are determined by the "qnums" dictionary key on that node.
        Can handle either a list of quantum numbers or an integer state. In the case of an
        integer state, just returns the integer value passed in.

        Parameters
        ----------
        state: Union[QState, int]
            Either the quantum numbers or integer value of the states.

        Raises
        ------
        ValueError:
            If there is no state matching the given quantum numbers

        Returns
        -------
        int: 
            The integer value corresponding the number of the state in the basis.
        """

        if isinstance(state, int):
            return state
        
        else:
            for n, d in self.couplings.nodes.data():
                if d['qnums'] == list(state):
                    return n
        
        raise ValueError(f"No state in cell with quantum numbers {state}")


    def get_qnums(self, state: Union[QState, int]):
        """Gets the quantum numbers for a given states.
        
        Works by either retrieving quantum numbers for an integer state,
        or transparently passing through a state already defined by its
        quantum numbers.

        Parameters
        ----------
        state : Union[QState, int]
            The state the get the quantum numbers for.

        Returns
        -------
        list(int)
            The quantum numbers of the state. If `state` is already a 
            list of quantum numbers, just returns the value of `state`

        Raises
        ------
        ValueError
            If the integer provided is outside the basis of the `Cell`.
        """
        if isinstance(state, int):
            try:
                qnums = self.couplings.nodes[state]["qnums"]
            except KeyError:
                raise ValueError(f"Cell basis is smaller that state {state}")
        else:
            qnums = state
    
        return self._validate_qnums(qnums)
