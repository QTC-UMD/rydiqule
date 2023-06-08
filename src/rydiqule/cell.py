"""
Physical Cell objects for use in solvers.
"""

import scipy
import numpy as np
import warnings

import scipy.constants
from scipy.constants import Boltzmann

# rydiqule imports
from .sensor import Sensor, ScannableParameter
from .sensor_utils import scale_dipole, calc_eta, calc_kappa
from .atom_utils import ATOMS

from typing import Literal, Optional, Sequence, List, Tuple, Callable

a0 = scipy.constants.physical_constants["Bohr radius"][0]

AtomFlags = Literal['H', 'Li6', 'Li7', 'Na', 'K39', 'K40', 'K41', 'Rb85', 'Rb87', 'Cs']
QState = Sequence  # TODO: consider using named tuples here
States = Tuple[int, ...]


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

    # these are not optional for a Cell
    eta: float
    kappa: float

    def __init__(self, atom_flag: AtomFlags, *atomic_states: QState,
                 gamma_transit: Optional[float] = None, cell_length: float = 0,
                 beam_area: float = 1e-6, beam_diam: Optional[float] = None,
                 temp: float = 300.0) -> None:
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
            of the system. 
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

        self.add_transit_broadening(gamma_transit)


    def add_states(self):
        """
        Deprecated.
        
        Use \"atomic_state\" keyword argument of the constructor instead.
        """
        raise NotImplementedError(("Adding states after a cell has been created is "
                                   "no longer supported. Please specify states with "
                                   "the \"atomic_states\" keyword argument in the constructor."))


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
            node_info: dict = {"qnums": state}

            # add energy to node
            if self.basis_size == 0:
                node_info["energy"] = 0
            else:
                ground_state = self.couplings.nodes[0]["qnums"]
                # Mrad/s
                node_info["energy"] = self.atom.getTransitionFrequency(*ground_state[:3],
                                                                       *state[:3])*1e-6*2.0*np.pi

            # add state lifetime to node
            if self.basis_size == 0:
                gamma_lifetime = 0
            else:
                gamma_lifetime = 1E-6/self.atom.getStateLifetime(*state[0:3])
            node_info['gamma_lifetime'] = gamma_lifetime

            self.couplings.add_node(self.basis_size, **node_info)

            self.basis_size += 1

        self._add_decay_to_graph()
        self._get_probe_info()


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
        >>> cell = rq.Cell("Rb85", *rq.D2_states(5), state1, state2) #D2 states gets the states for the Rubidium 85 D2 line
        >>> print(cell.states_list())
        [[5, 0, 0.5, 0.5], [5, 1, 1.5, 0.5], [50, 2, 2.5, 2.5], [51, 2, 2.5, 2.5]]

        """ # noqa
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
        >>> cell = rq.Cell("Rb85", *rq.D2_states("Rb85"), state1, state2) #uses the D2 line of Rb85
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


    def _get_probe_info(self, q_optical: Literal[-1,0,1] = 0) -> None:
        """
        Internal helper method to get information about the probing transition.

        For this function, the probe transition is defined as the transition between
        the ground state and first excited state.

        Parameters
        ----------
        q_optical : int, optional
            polarization of probing optical field in spherical basis. Must be -1, 0, 1.
            Defaults to 0 for linear polarization.

        """
        gState = self.states_list()[0]
        iState = self.states_list()[1]

        self.probe_elem = self.atom.getDipoleMatrixElement(*gState, *iState, q_optical)
        self.probe_freq = np.abs(self.atom.getTransitionFrequency(*gState[:3], *iState[:3]))

        self.density = self.atom.getNumberDensity(self.temp)
        self.kappa = calc_kappa(self.probe_freq, self.probe_elem, self.density)
        self.eta = calc_eta(self.probe_freq, self.probe_elem, self.beam_area)


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
            phase: Optional[ScannableParameter] = 0, kvec: Tuple[float,float,float] = (0,0,0),
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
        rabi_frequency: float, optional
            The rabi frequency, in Mrad/s, of the coupling
            field. If specified, `e_field`, `beam_power`, and `beam_waist` cannot
            be specified.
        detuning: float, optional
            Field detuning, in Mrad/s, of a coupling
            in the RWA. If specified, RWA is assumed, otherwise RWA not assumed,
            and transition frequency will be calculated based on atomic properties.
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

        >>> cell = rq.Cell("Rb85", *rq.D2_states("Rb85"))
        >>> cell.add_coupling(states=(0,1), detuning=1, rabi_frequency=2)
        >>> print(dict(cell.couplings.edges))
        {(0, 0): {'gamma_transit': 0.41172855461658464, 'label': '(0,0)'}, (0, 1): {'rabi_frequency': 2, 'detuning': 1, 
        'phase': 0, 'kvec': (0, 0, 0), 'label': '(0,1)'}, (1, 0): {'gamma_transition': 37.829349995476726, 
        'label': '(1,0)', 'gamma_transit': 0.41172855461658464}}
        
        Here we see implicitly calling this overloaded function through :meth:`~.Sensor.add_couplings`.
        
        >>> cell = rq.Cell("Rb85", *rq.D2_states("Rb85"))
        >>> c = {"states":(0,1), "detuning":1, "rabi_frequency":2}
        >>> cell.add_couplings(c)
        >>> print(dict(cell.couplings.edges))
        {(0, 0): {'gamma_transit': 0.41172855461658464, 'label': '(0,0)'}, (0, 1): {'rabi_frequency': 2, 'detuning': 1, 
        'phase': 0, 'kvec': (0, 0, 0), 'label': '(0,1)'}, (1, 0): {'gamma_transition': 37.829349995476726, 
        'label': '(1,0)', 'gamma_transit': 0.41172855461658464}}

        `e_field` can be specified in stead of `rabi_frequency`,
        but a `rabi_frequency` will still be added to the system based on the `e_field`,
        rather than `e_field` directly.
        
        >>> cell = rq.Cell("Rb85", *rq.D2_states("Rb85"))
        >>> c = {"states":(0,1), "detuning":1, "e_field":6}
        >>> cell.add_couplings(c)
        >>> print(cell.couplings.edges(data='e_field'))
        >>> print(cell.couplings.edges(data='rabi_frequency'))
        [(0, 0, None), (0, 1, None), (1, 0, None)]
        [(0, 0, None), (0, 1, -1.172912676105507), (1, 0, None)]
        
        As can `beam_power` and `beam_waist`,
        with similar behavior regarding how information is stored.

        >>> cell = rq.Cell("Rb85", *rq.D2_states("Rb85"))
        >>> c = {"states":(0,1), "detuning":1, "beam_power":1, "beam_waist":1}
        >>> cell.add_couplings(c)
        >>> print(cell.couplings.edges(data='beam_power'))
        >>> print(cell.couplings.edges(data='rabi_frequency'))
        [(0, 0, None), (0, 1, None), (1, 0, None)]
        [(0, 0, None), (0, 1, 4.28138982322625), (1, 0, None)]

        """ # noqa
        states = self._states_valid(states)
        state1 = self.states_list()[states[0]]
        state2 = self.states_list()[states[1]]
        
        suppress_dipole_warn = extra_kwargs.pop("suppress_dipole_warn", False)
        
        try:
            dipole_moment = self.atom.getDipoleMatrixElement(*state1,*state2, q)
        except ValueError:
            msg = f"Transition between states {state1} and {state2} not electric-dipole allowed."\
                "Solutions may be innacurate. "\
                "Suppress this by passing \"suppress_dipole_warn=True\" to Cell.add_coupling"
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
                msg = """Cell does not support explicit definietion of transition_frequency,
                it is calculated based on atomic properties."""
                raise ValueError(msg)
            else:
                transition_frequency = self.atom.getTransitionFrequency(*state2[:3],
                                                                        *state1[:3])*1e-6*2.0*np.pi

        super().add_coupling(states=states, rabi_frequency=passed_rabi,
                             detuning=detuning, transition_frequency=transition_frequency,
                             phase=phase, kvec=kvec, time_dependence=time_dependence, label=label,
                             **extra_kwargs)
