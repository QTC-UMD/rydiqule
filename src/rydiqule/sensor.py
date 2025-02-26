"""
Sensor objects that control solvers.
"""

import numpy as np
import networkx as nx
import warnings

import itertools

from .sensor_utils import (ScannableParameter, CouplingDict, State, States, StateSpec, StateSpecs, TimeFunc,
                           match_states, _squeeze_dims, expand_statespec, state_tuple_to_str)
from .exceptions import RydiquleError, CouplingNotAllowedError
from .exceptions import RWAWarning, PopulationNotConservedWarning, RydiquleWarning, debug_state

from typing import List, Tuple, Dict, Literal, Callable, Optional, Union, Sequence, Iterable, Sized, cast
# generic type for working around homogeneous Dict[key,val] hints in zip_parameters
_ZP = Dict[Union[States,str], str]

BASE_SCANNABLE_KEYS = ["detuning",
                       "rabi_frequency",
                       "phase",
                       "e_shift"]
"""Reference list of all coherent coupling keys that support rydiqules stacking convention.
Note that all decoherence keys (keys beginning with `gamma_`) are supported, but handled separately.
"""

BASE_EDGE_KEYS = ["states",
                   "detuning",
                   "rabi_frequency",
                   "transition_frequency",
                   "phase",
                   "kvec",
                   "time_dependence",
                   "label",
                   "dipole_moment",
                   "coherent_cc"]
"""Reference list of all keys that can be specified with values in a coheroaddenct coupling.
Subclasses which inherit from :class:`~.Sensor` should override the `valid_parameters` attribute,
NOT this list. The `valid_parameters` attribute is initialized as a copy of `BASE_EDGE_KEYS`."""

PROTECTED_LABELS = ["gamma", "kvec"]

class Sensor():
    """
    Class that contains minimum information necessary to run the solvers.

    Consider this class the theorist's interface to the solvers.
    It requires nearly complete, explicit specification of inputs.
    This allows for very fine control of the solvers,
    including the ability to solve systems that are not entirely physical.

    """

    eta: Optional[float] = None
    """Noise density prefactor, in units of root(Hz).
    Must be specified when using :class:`Sensor`.
    Automatically calculated when using :class:`Cell`."""

    kappa: Optional[float] = None
    """Differential prefactor, in units of (rad/s)/m.
    Must be specified when using :class:`Sensor`.
    Automatically calculated when using :class:`Cell`."""

    _probe_tuple: Optional[StateSpecs] = None

    _vP: Optional[float] = None

    probe_freq: Optional[float] = None
    """Probing transition frequency, in rad/s."""

    cell_length: Optional[float] = None
    """Optical path length of the medium, in meters."""

    beam_area: Optional[float] = None
    """Cross-sectional area of the probing beam, in square meters."""

    v_th: Optional[float] = None
    """Thermal velocity of the atoms in vapor cell, in meters per second."""

    temp: Optional[float] = None
    """Temperature of the vapor cell, in Kelvin."""

    atom_mass: Optional[float] = None
    """Mass of an atom in the vapor cell, in kilograms."""

    def __init__(self, states: Union[int, Sequence[StateSpec]],
                 *couplings: CouplingDict,
                 vP: Optional[float] = None) -> None:
        """
        Initializes the Sensor with the specified basis .

        Can be specified as either an integer number of states (which will automatically
        label the states `[0,...,basis_size]`) or list of valid state specifications.


        Parameters
        ---------
        states: int or list of statespec
            The specification of the basis size and labelling for a new `Sensor`. Can be
            either a integer or a list of valid state specifications. If specified as an integer
            `n`, the created `Sensor` will have `n` states labelled as `0,...n`. Valid state
            specifications are tuples containing either numbers or strings, optionally a list of
            the same.
            In the case of a list, the tuple will be converted into a list of tuples with each
            corresponding to one element in the list element element of the specification. See
            the `Examples` section for examples on how to specify groups of states. Note that
            with only a single statespec, it must be passed as an element of a list
        *couplings : tuple(dict)
            Couplings dictionaries to pass to :meth:`~.add_couplings` on sensor construction.
        vP: float, optional
            Most probable speed of the 3D Maxwell-Boltzmann distribution of the ensemble.
            Calculated as :math:`\\sqrt{2kT/m}` and is provided in units of m/s.
            This parameter is only necessary to perform Doppler-broadened solves.

        Raises
        ------
        RydiquleError
            If `basis` is not an integer or iterable.
        RydiquleError
            If any of the state label specifications of basis are the wrong type.

        Examples
        --------
        Providing an integer will define a sensor with the given basis size, labelled with
        ascending integers.

        >>> s = rq.Sensor(3)
        >>> s.states
        [0, 1, 2]

        States can also be defined with a list of integers:

        >>> s = rq.Sensor([0, 1, 2])
        >>> s.states
        [0, 1, 2]

        States can also be strings

        >>> s = rq.Sensor([0, 'e1', 'e2'])
        >>> s.states
        [0, 'e1', 'e2']

        States can be defined with tuples. These can be thought of as quantum numbers, although
        no physics around quantum numbers exist in `Sensor`, so the values are completely
        general.

        >>> s = rq.Sensor([(1,-1),(1,1)])
        >>> s.states
        [(1, -1), (1, 1)]

        States can be specified in groups with a "state specification", which will
        expand lists of quantum numbers

        >>> statespec = (1,[-1,0,1])
        >>> s = rq.Sensor([(0,0), statespec])
        >>> s.states
        [(0, 0), (1, -1), (1, 0), (1, 1)]

        """
        #if its an int, expand to a list
        if isinstance(states, int):
            basis = list(range(states))

        elif isinstance(states, list):
            basis: List[State] = []
            for  statespec in states:
                basis += expand_statespec(statespec)
        else:
            raise RydiquleError("'states' must be specified by a list of states or an integer defining their range")


        if len(basis) != len(set(basis)):
            raise RydiquleError(f"All state labels must be unique, got {states}")

        self.valid_parameters = BASE_EDGE_KEYS.copy()
        self.scannable_parameters = BASE_SCANNABLE_KEYS.copy()
        self.protected_labels = PROTECTED_LABELS.copy()
        self.couplings: nx.DiGraph = nx.DiGraph()
        self.couplings.add_nodes_from(basis)

        self._zip_labels: List = []

        if vP is not None:
            self.vP = vP

        if len(couplings) > 0:
            self.add_couplings(*couplings)

        if debug_state():
            print(f'Sensor initialized with {len(basis):d} states!')

    #add as a property to enforce
    @property
    def probe_tuple(self) -> Optional[StateSpecs]:
        """Coupling edge that corresponds to the probing field.
        Defaults to `None` and gets set to the first coupling
        added to the system with :meth:`~.Sensor.add_coupling`.
        Can be modified directly."""

        return self._probe_tuple


    @probe_tuple.setter
    def probe_tuple(self, probe_tuple: StateSpecs):
        state1_list = self.states_with_spec(probe_tuple[0])
        state2_list = self.states_with_spec(probe_tuple[1])

        if not (len(state1_list) > 0 and len(state2_list) > 0):
            raise ValueError(f"Probe tuple specification {probe_tuple} contains invalid state specs")

        self._probe_tuple = probe_tuple

    @property
    def vP(self) -> float:
        """Most probable speed of the 3D Maxwell-Boltzmann distribution.

        This is defined as :math:`\\sqrt{2kT/m}` and is given in units of m/s.

        This must be defined manually when performing Doppler-broadened solves.
        Accessing it before definition will raise an error.
        """
        if self._vP is None:
            raise RydiquleError("You must specify the attribute 'vP' before doing a Doppler broadened solve. " +
                                "This is done with Sensor's `vP` keyword argument " +
                                "or by setting the attribute after Sensor creation.")
        return self._vP

    @vP.setter
    def vP(self, vP: float):

        if not vP > 0.0:
            raise ValueError('Most probable speed must be positive')

        self._vP = vP


    def set_experiment_values(self, probe_freq:float, kappa: float,
                              eta: Optional[float] = None,
                              cell_length: Optional[float] = None,
                              beam_area: Optional[float] = None,
                              v_th: Optional[float] = None,
                              temp: Optional[float] = None,
                              atom_mass: Optional[float] = None
                              ):
        """Sets attributes needed for observable calculations.

        Parameters
        ----------
        probe_tuple: tuple of StateSpec
            Coupling that corresponds to the probing field. If `None`, corresponding Sensor
            attribute remains unchanged. Defaults to `None`.
        probe_freq: float
            Frequency of the probing transition, in Mrad/s. If `None`, corresponding Sensor
            attribute remains unchanged. Defaults to `None`.
        kappa: float
            Numerical prefactor that defines susceptibility, in (rad/s)/m.
            If `None`, corresponding Sensor  attribute remains unchanged. Defaults to `None`.
            See :func:`~.get_susceptibility` and :class:`Cell.kappa` for details.
        eta: float
            Noise-density prefactor, in root(Hz). If `None`, corresponding Sensor
            attribute remains unchanged. Defaults to `None`.
            See :class:`Cell.eta` for details.
        cell_length: float, optional
            The optical path length through the medium, in meters. If `None`, corresponding Sensor
            attribute remains unchanged. Defaults to `None`.
        beam_area: float, optional
            The cross-sectional area of the beam, in m^2. If `None`, corresponding Sensor
            attribute remains unchanged. Defaults to `None`.
        v_th: float, optional
            Thermal velocity of the Maxwell-Boltzmann distribution: v_th = (k_B*T/m)^(1/2) in units of m/s.
            Defaults to `None`.
        temp: float, optional
            Temperature of the vapor cell in units of Kelvin. Defaults to `None`.
        atom_mass: float, optional.
            Mass of a sensing atom in kg. Defaults to `None`.
        """
        self.probe_freq = probe_freq
        self.kappa = kappa

        if cell_length is not None:
            self.cell_length = cell_length
        if beam_area is not None:
            self.beam_area = beam_area
        if eta is not None:
            self.eta = eta

        if v_th is not None:
            self.v_th = v_th
        if temp is not None:
            self.temp = temp
        if atom_mass is not None:
            self.atom_mass = atom_mass


    @property
    def basis_size(self) -> int:
        """Property to return the number of nodes on the Sensor graph.

        Returns
        -------
        int
            The number of nodes on the graph, corresponding to the basis size for the system.
        """
        return len(self.couplings)


    @property
    def states(self) -> List[State]:
        """Property which gets a list of labels for the sensor in the order defined in
        :meth:`~.Sensor.__init__`. This is also the order corresponding the rows and columns
        in the system Hamiltonian and decoherence matrix.

        Returns
        -------
        list
            List of states of the system defined the constructor, in the order corresponding to
            rows and columns of the Hamiltonian.
        """

        return list(self.couplings.nodes())


    def add_energy_shift(self, statespec: StateSpec, shift: ScannableParameter, **kwargs):
        """Add an energy shift to a single state or a group of states.

        `statespec` can be provided either as a single state in the Sensor or as a valid
        state specification matching a group of states. When `statespec` matches a single state,
        :meth:`~.Sensor.add_single_energy_shift` method will be dispatched. In the case of
        a multi-state specification, the :meth:`~.Sensor.add_energy_shift_group` method will be
        dispatched applying an individual shift to all states with labels matching the
        specification provided.

        Note that an energy shifts are applied to the underlying graph as a self-edge connecting a
        node to itsself, not as data on the node itsself.

        Additional arguments for either dispatched function are passed normally via `**kwargs`

        Parameters
        ----------
        state_spec : StateSpec
            Integer or string label matching a state in the `Sensor`, or state specification
            matchingone or more states in the `Sensor`. The number of states this corresponds
            to will affect which internal function is dispatched.
        shift : float or array-like
            The energy shift to apply to the matching state or states in Mrad/s. Note that if it
            corresponds to multiple states, the `prefactors` argument of
            :meth:`~.Sensor.add_energy_shift_group` will be multiplied by this value for the
            corresponding state.

        Raises
        ------
        RydiquleError
            If the state provided does not match any state in the `Sensor`.

        Examples
        --------
        The basic use of `add_energy_shift` is to add terms to the diagonal of the hamiltonian.

        >>> s = rq.Sensor(3)
        >>> s.add_energy_shift(1, 1)
        >>> s.add_energy_shift(2, 2.5)
        >>> print(s.couplings.edges(data=True))
        [(1, 1, {'e_shift': 1, 'label': '1'}), (2, 2, {'e_shift': 2.5, 'label': '2'})]
        >>> print(s.get_hamiltonian())
        [[0. +0.j 0. +0.j 0. +0.j]
         [0. +0.j 1. +0.j 0. +0.j]
         [0. +0.j 0. +0.j 2.5+0.j]]

        `add_energy_shift` can be used with state specifications.

        >>> s = rq.Sensor([(0,0), (1,[-1,0,1])])
        >>> prefactors = {(1,i):i for i in [-1,0,1]}
        >>> s.add_energy_shift((1, [-1,0,1]), 0.1, prefactors=prefactors)
        >>> print(s.couplings.edges(data="e_shift"))
        [((1, -1), (1, -1), -0.1), ((1, 0), (1, 0), 0.0), ((1, 1), (1, 1), 0.1)]
        >>> print(s.get_hamiltonian())
        [[ 0. +0.j  0. +0.j  0. +0.j  0. +0.j]
         [ 0. +0.j -0.1+0.j  0. +0.j  0. +0.j]
         [ 0. +0.j  0. +0.j  0. +0.j  0. +0.j]
         [ 0. +0.j  0. +0.j  0. +0.j  0.1+0.j]]

        """
        states_list = self.states_with_spec(statespec)

        if len(states_list) == 1:
            self.add_single_energy_shift(states_list[0], shift, **kwargs)
        elif len(states_list) > 1:
            self.add_energy_shift_group(states_list, shift, **kwargs)
        else:
            raise RydiquleError(f"State specification {statespec} does not correspond to any states.")


    def add_single_energy_shift(self, state: State, shift: ScannableParameter, label=None):
        """Add an energy shift to a state.

        First perfoms validation that the provided `state` is actually a node in the graph, then
        adds the shift specified by `shift` to a self-loop edge keyed with `"e_shift"`. This value
        will be added to the corresponding diagonal term when the hamiltonian is generated.

        Parameters
        ----------
        state : int, str, or tuple
            The label corresponding to the atomic state to which the shift will be added.
        shift : float or list-like of float
            The magnitude of the energy shift, in Mrad/s

        Raises
        ------
        RydiquleError
            If the supplied `state` is not in the system.
        """
        if label is None:
            label = str(state)
        if not self.couplings.has_node(state):
            raise RydiquleError(f"state {state} is not a node on the graph")

        self._remove_edge_data((state, state), kind="coherent")
        self.couplings.add_edge(state, state, e_shift=shift, label=label)
        if debug_state():
            print(f'   Added energy shift for {state}')


    def add_energy_shifts(self, shifts: dict):
        """Wrapper for :meth:`Sensor.add_energy shift b`.

        Shifts are specified with the `shifts` dictionary, which is keyed with states and
        has values corresponding to the energy shift applied to the state in Mrad/s. Error
        handling and validation is done with the :meth:`~.Sensor.add_energy_shift` function.

        Parameters
        ----------
        shifts : dict
            Dictionary keyed with states with values corresponding to the energy shift, in Mrad/s,
            of the corresponding state.
        """

        try:
            shifts_items = shifts.items()
        except AttributeError as err:
            raise RydiquleError("Shifts parameters must be a dictionary-like object") from err

        for state, shift in shifts_items:
            self.add_energy_shift(state, shift)


    def add_energy_shift_group(self, states: List[State], shift:ScannableParameter,
                               prefactors:Optional[dict]=None,
                               zip_label:Optional[str]=None):
        """Add energy shifts to a group of states, optionally with a modifying prefactor for each.

        Given a list of states, calls :meth:`~.Sensor.add_single_energy_shift` on each one with the
        provided energy shift. Shifts are modified by a multiplicative factor defined by the
        `prefactors` dictionary. The dictionary is keyed with states that are elements of `states`
        with entries corresponding to a factor multiplied by the base `shift` argument for each
        state. When energy shifts are array-like, the `e_shift` attribute corresponding to each
        self-edge will be zipped with :meth:`~.Sensor.zip_parameters`.

        Parameters
        ----------
        states : list of states
            List of states to include in the group.
        shift : float or array-like
            The base value of the energy shift to apply the states. Will be modified by entries
            of the `prefactors` dictionary.
        prefactors : dict or `None`, optional
            Dictionary of values by which to multiply the base `shift` parameter for each
            each state. Keys are elements of the `states` list, entries are the corresponding
            factor by which to multiply `shift` for that state. If `None`, all prefactors are
            set to 1. If not `None`, the prefactors for any nonspecified values will be set
            to zero. Default is `None`.
        zip_label : str or `None`, optional
            Label passed to :meth:`~.Sensor.zip_parameters` when the shift is provided as an
            array-like when all states in the group are zipped together. Defaults to `None`.

        Raises
        ------
        RydiquleError
            If the supplied energy shift is not a float and cannot be interpreted as a numpy

        Examples
        --------
        >>> s = rq.Sensor(['g','e1','e2'])
        >>> factors = {'e1':1, 'e2':2}
        >>> s.add_energy_shift_group(["e1","e2"], 0.1, prefactors=factors)
        >>> print(s.couplings.edges(data='e_shift'))
        [('e1', 'e1', 0.1), ('e2', 'e2', 0.2)]
        >>> print(s.get_hamiltonian())
        [[0. +0.j 0. +0.j 0. +0.j]
         [0. +0.j 0.1+0.j 0. +0.j]
         [0. +0.j 0. +0.j 0.2+0.j]]

        """
        if prefactors is None:
            prefactors = {state: 1.0 for state in states}

        for state in states:
            self.add_single_energy_shift(state, shift * prefactors.get(state, 0.0))

        if hasattr(shift, "__len__"):
            try:
                shift=np.array(shift, dtype=np.float64)
            except (ValueError, TypeError):
                raise RydiquleError(f"Shift type {type(shift)} cannot be interpreted as an array")

        if isinstance(shift, np.ndarray) and len(states) > 1:
            zip_dict: _ZP = {(s,s):"e_shift" for s in states}
            self.zip_parameters(zip_dict, zip_label=zip_label)


    def add_coupling(self, states: StateSpecs, **kwargs):
        """Add a coupling between states or groups of states.

        Wraps the :meth:`~.Sensor.add_single_coupling` and :meth:`~.Sensor.add_coupling_group`
        functions, and dispatches to the appropriate one depending on the number of states in the
        `states` argument. Additional keyword arguments will be passed unmodified to the
        relevant method. See documentation of those functions for details on keyword argument
        options.

        If each state specification in `states` correspond to a single state, the
        corresponding states will be passed to :meth:`~.Sensor.add_single_coupling`. If
        either or both specifications correspond to multiple states, the corresponding lists
        will be passed as the `states1` and `states2` lists in :meth:`~.Sensor.add_coupling_group`.

        If this is the first time `add_coupling` has been called for this `Sensor`, sets the
        `probe_tuple` attribute to the `states` specification , which is used as the default,
        for calculating observable values, in a :class:`~.sensor_soution.Solution` after solving.
        For this reason, this function is preferred over :meth:`~.Sensor.add_single_coupling`
        and :meth:`~.Sensor.add_coupling_group` outside of special circumstances. If couplings
        are added with either of the specific dispatched functions, `probe_tuple` should be set
        manually.

        Parameters
        ----------
        states : tuple of Statespecs
            The states or state manifolds of the coupling. If both are integers or state
            specifications matching a single state in the `Sensor`, :meth:`~.Sensor.add_single_coupling`
            is dispatched. If either argument is a string pattern matching multiple states,
            :meth:`~.Sensor.add_coupling_group` is dispatched.
        **kwargs
            Additional keyword argumets passed to the relevant function. See the documentation for
            :meth:`~.Sensor.add_single_coupling` and :meth:`~.Sensor.add_coupling_group` for
            details on valid keyword arguments.

        Notes
        -----
        ..note:
            Outside of specific use cases for users well-versed in the `rydiqule` code base, this
            method is preferred over :meth:`~.Sensor.add_single_coupling` and
            :meth:`~.Sensor.add_coupling_group` since it appropriately handles necessary
            backend bookeeping.

        Examples
        --------
        Couplings are added identically regardless of how states are labelled.

        >>> s = rq.Sensor(2)
        >>> s.add_coupling((0,1), detuning=1, rabi_frequency=2)
        >>> print(s.get_hamiltonian())
        [[ 0.+0.j  1.+0.j]
         [ 1.-0.j -1.+0.j]]

        >>> s = rq.Sensor(['g','e'])
        >>> s.add_coupling(('g','e'), detuning=1, rabi_frequency=2)
        >>> print(s.get_hamiltonian())
        [[ 0.+0.j  1.+0.j]
         [ 1.-0.j -1.+0.j]]

        Couplings can have list-like parameters, in which case the resulting rydiqule will
        compute hamiltonians for all values simultaneously. Here 101 x 21 = 2,121 2x2 Hamiltonians
        are generated simultaneously, with one for every combination of parameters, and arranged
        into a single array.

        >>> s = rq.Sensor(2)
        >>> det=np.linspace(-10, 10, 101)
        >>> rabi = np.linspace(-1, 1, 21)
        >>> s.add_coupling((0,1), detuning=det, rabi_frequency=rabi, label="laser")
        >>> print(s.get_hamiltonian().shape)
        (101, 21, 2, 2)

        Couplings can be be defined between manifolds of states with state specifications.
        The values for rabi frequencies of individual states are modified by the
        `coupling_coefficients` keyword argument. To avoid cumbersome numbers of nested
        brackets, it is advisable to name manifolds with variables. Note that `StateSpec`s
        can be expanded with :func:`~.sensor_utils.expand_statespec` for this purpose.

        >>> g = (0,0) #statespec for ground
        >>> excited = (1,[-1,0,1]) #statespec for excited
        >>> [e1,e2,e3] = rq.sensor_utils.expand_statespec(excited)
        >>> cc = {
        ...     (g, e1): 0.25,
        ...     (g, e2): 0.5,
        ...     (g, e3): 0.25,
        ... } # coupling coefficiens
        >>> s = rq.Sensor([g, excited])
        >>> s.add_coupling((g, excited), rabi_frequency=10, detuning=1, coupling_coefficients=cc, label="laser")
        >>> print(s.get_hamiltonian())
        [[ 0.  +0.j  1.25+0.j  2.5 +0.j  1.25+0.j]
         [ 1.25-0.j -1.  +0.j  0.  +0.j  0.  +0.j]
         [ 2.5 -0.j  0.  +0.j -1.  +0.j  0.  +0.j]
         [ 1.25-0.j  0.  +0.j  0.  +0.j -1.  +0.j]]

        This function sets the `probe_tuple` for the first call, but not subsequent calls. This
        makes it preferred over :meth:`~.Sensor.add_single_coupling` and
        :meth:`~.Sensor.add_coupling_group`, which do not have this behavior.

        >>> g = (0,0) #statespec for ground
        >>> e1 = (1,[-1,0,1]) #statespec for 1st excited
        >>> e2 = (2,0)
        >>> s = rq.Sensor([g, e1, e2])
        >>> print(s.probe_tuple)
        None
        >>> s.add_coupling((g, e1), rabi_frequency=10, detuning=1, label="red")
        >>> print(s.probe_tuple)
        ((0, 0), (1, [-1, 0, 1]))
        >>> s.add_coupling((e1,e2), rabi_frequency=1, detuning=2, label="blue")
        >>> print(s.probe_tuple)
        ((0, 0), (1, [-1, 0, 1]))

        For state manifolds, list-like parameters are automatically zipped. See
        :meth:`Sensor.zip_parameters` for more details on the mechanics of zipping parameters.

        >>> g = (0,0) #statespec for ground
        >>> e1 = (1,[-1,0,1]) #statespec for 1st excited
        >>> s = rq.Sensor([g, e1])
        >>> det = np.linspace(-1,1,11)
        >>> s.add_coupling((g, e1), rabi_frequency=10, detuning=det, label="red")
        >>> print(s.couplings.edges)
        [((0, 0), (1, -1)), ((0, 0), (1, 0)), ((0, 0), (1, 1))]
        >>> print(s._zip_labels)
        ['red_detuning']
        >>> print(s.get_hamiltonian().shape)
        (11, 4, 4)

        """
        #define as probe if this is the first coupling added
        if len(nx.get_edge_attributes(self.couplings, "rabi_frequency"))==0 and self.probe_tuple is None:
            self.probe_tuple = states

        #get relevant states from state specifications
        try:
            states_list1 = self.states_with_spec(states[0])
            states_list2 = self.states_with_spec(states[1])
        except RydiquleError as err:
            raise RydiquleError(f"Invalid State specifications {states}") from err

        if len(states_list1) == len(states_list2) == 1:
            self.add_single_coupling((states_list1[0], states_list2[0]), **kwargs)

        elif len(states_list1) > 1 or len(states_list2) > 1:
            self.add_coupling_group(states_list1, states_list2, **kwargs)

        else:
            raise RydiquleError("Each state specification must match at least one state.")


    def add_single_coupling(
            self, states: States, rabi_frequency: Optional[ScannableParameter] = None,
            detuning: Optional[ScannableParameter] = None,
            transition_frequency: Optional[float] = None,
            phase: Optional[ScannableParameter] = None,
            kvec: Sequence[float] = (0,0,0),
            time_dependence: Optional[TimeFunc] = None,
            label: Optional[str] = None, coherent_cc: Optional[float] = None,
            **extra_kwargs) -> None:
        """
        Adds a single coupling of states to the system.

        One or more of these paramters can be a list or array-like of values to represent
        a laser that can take on a set of discrete values during a field scan.
        Designed to be a user-facing wrapper for :meth:`~._add_coupling` with arguments
        for states and coupling parameters.

        Note that unlike :meth:`~.Sensor.add_coupling`, this function does not set the
        `probe_tuple` attribute, so if used to add the first coupling, `probe_tuple` must
        be set manually.

        Parameters
        ----------
        states : tuple of States
            The pair of states of the sensor which the state couples. Must be a tuple
            of length 2, where each element is a string, integer, or tuple corresponding to a state in the
            Sensor as defined in the constructor. Tuple order indicates which state to has higher
            energy; the second state is always assumed to have higher energy.
        rabi_frequency : float or complex, or list-like of float or complex
            The rabi frequency of the field being added. Defined in units of Mrad/s. List-like
            values will invoke Rydiqule's stacking convention when relevant quantities are calculated.
        detuning : float or list-like of floats, optional
            The frequency difference between the transition frequency and the field frequency in
            units of Mrad/s. List-like values will invoke Rydiqule's stacking convention when relevant
            quantities are calculated. If specified, the coupling is treated with the rotating-wave
            approximation rather than in the lab frame, and `transition_frequency` is ignored if present.
            A positive number always indicates a blue detuning, and a negative number indicates a blue
            detuning.
        transition_frequency : float, optional
            The transition frequency between a particular pair of states. Must be a positive number.
            Only used directly in calculations if `detuning` is `None`, ignored otherwise.
            Note that on its own, it only defines the spacing between two energy levels and not the
            field itsself. To define a field, the `time_dependence` argument must be specified, or else
            the off-diagonal terms to drive transitions will not be generated in the Hamiltonian matrix.
        phase : float, optional
            Static phase offset in the rotating frame.
            Cannot be used outside the rotating frame, ie when detuning is not defined.
            Default is undefined, which is interpreted as 0 for couplings in the rotating frame.
        kvec : iterable, optional 
            A three-element iterable that defines the k-vector of a particular coupling field.
            It should have units of Mrad/m, such that :math:`vP*k_vec` gives the most probable doppler shift
            along each axis.
            Note that the `vP` class attribute must be defined to perform doppler-broadened solves.
            If equal to `(0,0,0)`, solvers will ignore doppler shifts on this field.
            Defaults to `(0,0,0)`.
        time_dependence : scalar function, optional
            A scalar function specifying a time-dependent field. The time dependence function is defined
            as a python funtion that returns a unitless value as a function of time (in microseconds)
            that is multiplied by the `rabi_frequency` parameter to get a  field strength scaled to units
            of Mrad/s.
        coherent_cc: float, optional
            Addtional information regarding coupling strength for the `states` coupling. Does
            **not** modify the `rabi_frequency` before adding to the graph. Rather, when the
            Hamiltonians and physical observables in :class:`~.sensor_solution.Solution` are
            computed, first multiplies the `rabi_frequency` by this value. The `rabi_frequency`
            can be thought of as a "base" field power, while this is a modification based on the
            coupling strength. If `None`, the `coherent_cc` added to the graph will be set to 1.
            Defaults to `None`.
        label : str or None, optional
            Name of the coupling. This does not change any calculations, but can be used
            to help track individual couplings, and will be reflected in the output of
            :meth:`~.Sensor.axis_labels`, and to specify zipping for :meth:`~.Sensor.zip_couplings`.
            If `None`, the label is generated as the value of `states` cast to a string with
            whitespace removed. Defaults to `None`.

        Raises
        ------
        RydiquleError
            If `states` cannot be interpreted as a tuple.
        RydiquleError
            If `states` does not have a length of 2.
        RydiquleError
            If the states specified in the `states` argument are not in the basis of the
            `Sensor`
        RydiquleError
            If both `rabi_frequency` and `dipole_moment` are specified or if
            neither are specified.
        RydiquleError
            If both detuning and transition_frequency are specified or if
            neither are specified.
        RydiquleError
            If a coupling is added in the non-rotating frame (detuning=None)
            and no time dependence function is specified.
        RydiquleError
            If `kvec` is not a three element sequence of floats.

        Warns
        -----
        RWAWarning
            Raised if large `transition_frequency` is passed,
            which can lead to very long time-dependent solves
            and is often not intended.
        RydiquleWarning
            Raised if 'kvec' is likely incorrectly defined (field k-vector in Mrad/m).
            Either by incorrect units or as most probable velocity vector (rq v1 convention).
            Triggers if the implied wavelength is less than 210 nm or greater than 2095 nm.

        Examples
        --------
        >>> s = rq.Sensor(2)
        >>> s.add_single_coupling((0,1), detuning=1, rabi_frequency=2)
        >>> print(s.get_hamiltonian())
        [[ 0.+0.j  1.+0.j]
         [ 1.-0.j -1.+0.j]]

        >>> s = rq.Sensor(['g','e'])
        >>> s.add_single_coupling(('g','e'), detuning=1, rabi_frequency=2)
        >>> print(s.get_hamiltonian())
        [[ 0.+0.j  1.+0.j]
         [ 1.-0.j -1.+0.j]]

        >>> s = rq.Sensor(2)
        >>> s.add_single_coupling((0,1), detuning=np.linspace(-10, 10, 101), rabi_frequency=2, label="laser")
        >>> print(s.get_hamiltonian().shape)
        (101, 2, 2)

        The `coherent_cc` attribute does not modify the `rabi_frequency` that is stored on the graph, but
        rather in the computed hamiltonian

        >>> s = rq.Sensor(2)
        >>> s.add_single_coupling((0,1), detuning=1, rabi_frequency=2, coherent_cc=1/3)
        >>> print(s.couplings.edges.data("rabi_frequency")) #shows the rabi_frequency on the graph is 2
        [(0, 1, 2)]
        >>> print(s.get_hamiltonian())
        [[ 0.      +0.j  0.333333+0.j]
         [ 0.333333-0.j -1.      +0.j]]

        >>> s = rq.Sensor(2)
        >>> step = lambda t: 1 if t>=1 else 0
        >>> s.add_single_coupling((0,1), transition_frequency=1000, rabi_frequency=2, time_dependence=step)
        >>> print(s.get_hamiltonian())
        [[   0.+0.j    0.+0.j]
         [   0.+0.j 1000.+0.j]]
        >>> print(s.get_time_hamiltonian_components()[0])
        [array([[0.+0.j, 2.+0.j],
               [2.-0.j, 0.+0.j]])]

        >>> s = rq.Sensor(2, vP=10)
        >>> kp = 25*np.array([1,0,0])
        >>> s.add_single_coupling((0,1), detuning=1, rabi_frequency=2, kvec=kp)
        >>> s.get_hamiltonian()
        array([[ 0.+0.j,  1.+0.j],
               [ 1.-0.j, -1.+0.j]])

        """
        #ensure states are unique
        if states[0] == states[1]:
            raise RydiquleError(f'{states}: Coherent coupling must couple different states.')

        if coherent_cc is None:
            coherent_cc = 1.0

        if 'suppress_rwa_warn' in extra_kwargs:
            warnings.warn(("The 'suppress_rwa_warn' kwarg is Deprecated. "
                           "Use warnings.simplefilter('ignore', rq.RWAWarning)"),
                           FutureWarning)

        if (phase is None) and (detuning is not None):
            phase = 0
        elif (phase is not None) and (transition_frequency is not None):
            raise RydiquleError(f"{states}: Cannot specify rotating frame phase offset "
                                "for coupling not in the rotating frame. "
                                "Incorporate phase offsets into the time-dependence.")

        if detuning is None and time_dependence is None:
            raise RydiquleError(f"Got rotating frame but no time dependence for coupling {states}")
        
        if len(kvec) != 3:
            raise RydiquleError('kvec must be a three-element sequence of floats')
        k_mag_sq = np.sum(np.asarray(kvec)**2)
        if not np.isclose(k_mag_sq, 0,0) and (k_mag_sq < 9 or k_mag_sq > 900):
            # implied lambda is > 2095nm or < 210nm
            warnings.warn((f"Coupling {states} has kvec = {kvec} " +
                           f"with |kvec|={np.sqrt(k_mag_sq):.3g}. " +
                           "This is likely unphysical as 'kvec' has been redefined " +
                           "to be the field k-vector, not the most probable velocity vector."),
                          RydiquleWarning)

        field_params = dict(
            states=states,
            rabi_frequency=rabi_frequency,
            detuning=detuning,
            transition_frequency=transition_frequency,
            phase=phase,
            kvec=kvec,
            time_dependence=time_dependence,
            label=label,
            coherent_cc=coherent_cc
        )
        field_params_trimmed = {k:v for k,v in field_params.items() if v is not None}

        full_edge_data = {
            param: np.array(val)
            if param in self.scannable_parameters and hasattr(val, "__len__")
            else val
            for (param, val) in {**field_params_trimmed, **extra_kwargs}.items()
        }

        if not (detuning is not None) ^ (transition_frequency is not None):
            raise RydiquleError(f"{states}: Please specify \'detuning\' for a field under the RWA"
                                " or \'transition_frequency\' for a coupling without the approximation,"
                                " but not both.")

        if transition_frequency is not None:
            if transition_frequency < 0:
                raise RydiquleError(f"{states}: \'transition_frequency\' must be positive.")
            elif transition_frequency > 5000:
                msg = (f"{states}: Not using the rotating wave approximation"
                    " for large transition frequencies can result in "
                    "prohibitively long computation times. Specify detuning to use "
                    "the rotating wave approximation or suppress with "
                    "\'warnings.simplefilter('ignore', rq.RWAWarning)\'")

                warnings.warn(msg, RWAWarning)

        self._add_coherent_data(**full_edge_data)
        if debug_state():
            print(f' Added coupling for {states}')


    def add_couplings(self, *couplings: CouplingDict,
                        **extra_kwargs) -> None:
        """
        Add any number of couplings between pairs of states.

        Acts as an alternative to calling :meth:`~.Sensor.add_coupling`
        individually for each pair of states. Can be used interchangably up to preference,
        and all of keyword :meth:`~.Sensor.add_coupling` are supported dictionary
        keys for dictionaries passed to this function.

        Note that since this function wraps :meth:`~.Sensor.add_coupling`, the first
        element of `couplings` will be used to set `probe_tuple`.

        Parameters
        ----------
        couplings : tuple of dicts
            Any number of dictionaries, each specifying the parameters of a single field
            coupling 2 states. For more details on the keys of each dictionry see the arguments
            for :meth:`~.Sensor.add_coupling`. Equivalent to passing each dictiories keys and
            values to :meth:`~.Sensor.add_coupling` individually.
        **extra_kwargs : dict
            Additional keyword-only arguments to pass to the relevant `add_coupling` method.
            The same arguments will be passed to each call of :meth:`~.Sensor.add_coupling`.
            Often used for warning suppression.
            Can also be used to define a common coupling parameter for each coupling.

        Examples
        --------
        >>> s = rq.Sensor(3)
        >>> blue = {"states":(0,1), "rabi_frequency":1, "detuning":2}
        >>> red = {"states":(1,2), "rabi_frequency":3, "detuning":4}
        >>> s.add_couplings(blue, red)
        >>> print(s)
        <class 'rydiqule.sensor.Sensor'> object with 3 states and 2 coherent couplings.
        States: [0, 1, 2]
        Coherent Couplings:
            (0,1): {rabi_frequency: 1, detuning: 2, phase: 0, kvec: (0, 0, 0), coherent_cc: 1.0, label: (0,1)}
            (1,2): {rabi_frequency: 3, detuning: 4, phase: 0, kvec: (0, 0, 0), coherent_cc: 1.0, label: (1,2)}
        Decoherent Couplings:
            None
        Energy Shifts:
            None

        """
        for c in couplings:
            self.add_coupling(**c, **extra_kwargs)


    def add_coupling_group(self, states1: List[State], states2: List[State], label:str,
                           rabi_frequency: Optional[ScannableParameter]=None,
                           detuning: Optional[ScannableParameter]=None,
                           transition_frequency: Optional[float]=None,
                           coupling_coefficients: Optional[dict]=None,
                           time_dependence: Union[Callable, Dict[States, Optional[Callable]], None]=None,
                           **kwargs):
        """Adds a group of couplings to a Sensor.

        Given 2 lists of states, iterates over each combination of states in the two lists,
        add performs the :meth:`~.Sensor.add_single_coupling` on that pair of states. All
        additional parameters are passed directly to the `add_single_coupling` function.

        Additionally, a multiplicative factor can be applied to the rabi frequency of each coupling
        (i.e. Clebsch-Gordon coefficients). These factors are provided by the `cc`(coupling
        coefficient) parameter as a dictionary
        with keys corresponding to state pairs in the groups,
        and the value being the multiplicative factor applied to `rabi_frequency`
        when the Hamiltonian is generated. Note that these `cc` values cannot be arrays.
        The corrollary for state energy (e.g. for `detuning` or `transition_frequency`)
        is handled via :meth:`~.Sensor.add_energy_shifts`. If no dictionary is supplied for
        for `coherent_cc`, *all* coupling coefficients are set to 1.0, effectively meaning that
        the base `rabi_frequency` supplied is what is added to the Hamiltonian. If a
        dictionary is supplied, any couplings whose coefficient is not specified by the dictionary
        will be left off the graph.

        If any of the parameters are specified as arrays, the associated
        couplings will be applied to all couplings and automatically zipped together with
        the label specified by `label`. For the purposes of axis labelling, the parameters
        will be zipped in the order`rabi_frequency`, `detuning`, `transition_frequency`.

        Note that unlike :meth:`~.Sensor.add_coupling`, this function does not set the
        `probe_tuple` attribute, so if used to add the first coupling, `probe_tuple` must
        be set manually.

        Parameters
        ----------
        states1 : list[str or int]
            List of states in the lower energy group of states. Must be integers or string
            values which correspond to states in the Sensor.
        states2 : list[str or int]
            List of states in the higher energy group of states. Must be integers or string
            values which correspond to states in the Sensor.
        label : str
            Required string label denoting what the group of couplings is called. Used to
            apply a label in :meth:`~.Sensor.zip_parameters`.
        rabi_frequency : ScannableParameter, optional
            Floating point value or list of values for the base rabi frequency of the coupling
            group. Multiplied by values specified in `cc` for individual couplings,
            often accounting for variations in dipole moment. Default is None.
        detuning : ScannableParameter, optional
            Base detuning for the coupling group. If specified, every coupling in the group will
            be treated in the rotating frame. Can be modified through energy level shifts on
            individual states specified by :meth:`~.Sensor.add_energy_shift`. Default is None.
        transition_frequency : float, optional
            Base transition frequency for the coupling group.
            Individual states can be shifted via :meth:`~.Sensor.add_energy_shift`.
            Default is None.
        coupling_coefficients : dict, optional
            Individual coupling coeffients passed to the :meth:`~.Sensor.add_single_coupling`
            method. If provided, defined by a dictionary keyed with tuples of states corresponding
            to couplings in this group, with values equal to the coupling coeffient to be passed
            to the `add_single_coupling` call for that coupling. If any entries are absent in the
            provided dictionary, they are assumed to not be coupled, and no coupling will be added
            for that transition. If `None`, defaults to a dictionary containing every coupling in
            coupling in the group with `None` for all values (defaulting to 1.0 when passed to
            `add_single coupling`). Defaults to `None`.
        time_dependence : scalar function or dict of scalar functions, optional
            Time-dependendent scalar factor that is multiplied by the rabi frequency in Hamiltonian
            generation. Can be specified as a single function, in which case the function will be
            used as the `time_dependence` argument for each coupling in the group
            (see :meth:`~.Sensor.add_single_coupling`),  Can also bespecified as a dictionary
            mapping state pairs in the coupling to individual functionswhich will be applied to
            the associated coupling in the same manner. In the case of a dictionary specification,
            each unspecified coupling will default to `time_dependence=None`.

        Raises
        ------
        RydiquleError
            If `states1` and `states2` only have one state. Use :meth:`~.Sensor.add_coupling` instead.

        Note
        ----
        .. note::
            The :meth:`~.Sensor.add_coupling` is typically preferred over this method, since it allows
            for shorthand specification of groups, and sets the :attr:`Sensor.probe_tuple` attribute.

        .. note::
            If a :class:`~.CouplingNotAllowedError` is raised while adding the individual couplings
            for the group, couplings that raised the error will be ignored.


        Examples
        --------
        Energy shifts added to remove degenerate energy levels.
        If no clebsch-gordon coefficients are supplied, ALL default to 1

        >>> s = rq.Sensor(['a1', 'a2', 'b1', 'b2'])
        >>> s.add_energy_shifts({'a2':0.1, 'b2':0.1})
        >>> s.add_coupling_group(['a1','a2'], ['b1','b2'], detuning=1, rabi_frequency=1, label='example')
        >>> s.get_hamiltonian()
        array([[ 0. +0.j,  0. +0.j,  0.5+0.j,  0.5+0.j],
               [ 0. +0.j,  0.1+0.j,  0.5+0.j,  0.5+0.j],
               [ 0.5-0.j,  0.5-0.j, -1. +0.j,  0. +0.j],
               [ 0.5-0.j,  0.5-0.j,  0. +0.j, -0.9+0.j]])

        If the cc dictionary is specified, any unspecified terms are skipped on the graph. Note that
        although `(0,3)` is in the coupling group, it is omitted from the graph since it is
        not in `coupling_coeffiecients`.

        >>> s = rq.Sensor(4)
        >>> cc = {(0,1):0.5, (0,2):0.5}
        >>> s.add_coupling_group([0],[1,2,3], detuning=1, rabi_frequency=1, coupling_coefficients=cc, label='foo')
        >>> print(s)
        <class 'rydiqule.sensor.Sensor'> object with 4 states and 2 coherent couplings.
        States: [0, 1, 2, 3]
        Coherent Couplings:
            (0,1): {rabi_frequency: 1, detuning: 1, phase: 0, kvec: (0, 0, 0), label: foo_0, coherent_cc: 0.5}
            (0,2): {rabi_frequency: 1, detuning: 1, phase: 0, kvec: (0, 0, 0), label: foo_1, coherent_cc: 0.5}
        Decoherent Couplings:
            None
        Energy Shifts:
            None

        For list-like parameters, the couplings are treated as originating from a single laser and
        that parameter is zipped across all couplings in the group.

        >>> g = (0,0) #statespec for ground
        >>> e1 = (1,[-1,0,1]) #statespec for 1st excited
        >>> s = rq.Sensor([g, e1])
        >>> det = np.linspace(-1,1,11)
        >>> s.add_coupling_group([(0,0)], [(1,-1), (1,0), (1,1)],
        ...                       rabi_frequency=10, detuning=det, label="red")
        >>> print(s)
        <class 'rydiqule.sensor.Sensor'> object with 4 states and 3 coherent couplings.
        States: [(0, 0), (1, -1), (1, 0), (1, 1)]
        Coherent Couplings:
            ((0, 0),(1, -1)): {rabi_frequency: 10, detuning: <parameter with 11 values>, phase: 0, kvec: (0, 0, 0), label: red_0, coherent_cc: 1.0, red_detuning: detuning}
            ((0, 0),(1, 0)): {rabi_frequency: 10, detuning: <parameter with 11 values>, phase: 0, kvec: (0, 0, 0), label: red_1, coherent_cc: 1.0, red_detuning: detuning}
            ((0, 0),(1, 1)): {rabi_frequency: 10, detuning: <parameter with 11 values>, phase: 0, kvec: (0, 0, 0), label: red_2, coherent_cc: 1.0, red_detuning: detuning}
        Decoherent Couplings:
            None
        Energy Shifts:
            None
        Zip Labels:
            ['red_detuning']
        >>> print(s.get_hamiltonian().shape)
        (11, 4, 4)

        """
        labels: List[str] = []
        if coupling_coefficients is None:
            coupling_coefficients = {(s1,s2):None for s1, s2 in itertools.product(states1, states2)}

        if callable(time_dependence) or time_dependence is None:
            time_dependence_full = {(s1, s2): time_dependence for s1, s2 in itertools.product(states1, states2)}
        else:
            time_dependence_full = time_dependence

        if len(states1) == len(states2) == 1:
            raise RydiquleError("Both states groups only have one state. Use add_coupling instead.")

        if hasattr(rabi_frequency, "__len__"):
            # must be numpy array to multiply scaling factors
            rabi_frequency = np.asarray(rabi_frequency)

        #iterate over combinations of states, and add a coupling for each
        for s1, s2 in itertools.product(states1, states2):
            label_full = label + "_" + str(len(labels))

            #skip if no coupling coefficient is supplied
            try:
                coherent_cc = coupling_coefficients[(s1, s2)]
            except KeyError:
                continue

            #get the relevant time_dependence
            try:
                single_time_dependence = time_dependence_full.get((s1,s2))
            except AttributeError:
                raise RydiquleError("'time_dependence' must be a callable or dict of callables")

            try:
                self.add_single_coupling((s1, s2), rabi_frequency=rabi_frequency, detuning=detuning,
                                transition_frequency=transition_frequency, label=label_full,
                                time_dependence=single_time_dependence, coherent_cc=coherent_cc,
                                **kwargs)
                labels.append(label_full)

            except CouplingNotAllowedError:
                if debug_state():
                    print(f'\tCoupling {(s1, s2)} in \'{label:s}\' skipped as not allowed')
                continue


        scannable_group_params = ['rabi_frequency', 'detuning']
        for param in scannable_group_params:

            # if param is array-like, zip all couplings together on that param
            if hasattr(locals()[param], "__len__"): #this is a little janky but probably fine
                zip_labels: _ZP = {l:param for l in labels}
                self.zip_parameters(zip_labels, zip_label=label+"_"+param)


    def _add_coherent_data(self, states: States, **field_params) -> None:
        """
        Function for internal use which will ensure the supplied couplings is valid,
        add the field to self.couplings.

        Exists to abstract away some of the internally necessary bookkeeping functionality from
        user-facing classes.

        Parameters
        ----------
            states : tuple
                The integer pair of states to be coupled.
            **field_params : dict
                The dictionry of couplings parameters. For details
                on the keys of the dictionry see :meth:`~.Sensor.add_coupling`.

        """
        states = self._states_valid(states)

        # if label not provided, set to default value of states tuple as a str
        if field_params.get("label") is None:
            field_params["label"] = state_tuple_to_str(states)

        # remove all coherent data in both directions on the graph
        # this prevents adding coherent couplings in both directions between 2 nodes
        self._remove_edge_data(states, 'coherent')
        self._remove_edge_data(states[::-1], 'coherent')  # pyright: ignore[reportArgumentType]

        coupling_labels = [l for _,_,l in self.couplings.edges(data="label")]
        if isinstance(field_params.get("label"), str) and field_params["label"] in coupling_labels + self._zip_labels:
            raise ValueError(f"Label {field_params['label']} is already on a sensor coupling or zip")

        self.couplings.add_edge(*states, **field_params)


    def zip_parameters(self, parameters: Dict[Union[States, str], str],  zip_label: Optional[str]=None):
        """
        Define 2 scannable parameters as "zipped" so they are scanned in parallel.

        Zipped parameters will share an axis when quantities relevant to the equations of
        motion, such as the `gamma_matrix` and `hamiltonian` are generated. So for 2 list-like
        parameters, the first elements in each are solved at the same time, then the second, etc
        Note that calling
        this function does not affect internal quanties directly, but flags them to be zipped
        at calculation time for relevant quantities.

        Internally, adds the `label` value to the internal list of zipped parameter labels, and
        adds a flag in the form of `<label>:<parameter_name>` to each edge of the graph.

        Parameters
        ----------
        parameters : dict
            Parameter labels to scan together. Parameters are specified with a dictionary keyed by
            the either pair of states defining the coupling (e.g. `(0,1)`) or a previously
            specified label (e.g. `"probe"`) with items corresponding to the
            respective parameter name (e.g. `"detuning"`).
        zip_label : optional, str
            String label shorthand for the zipped parameters. The label for the axis of these
            parameters in :meth:`~.Sensor.axis_labels()`. Does not affect functionality of the
            Sensor. If `None` (the default), the label used will be `"zip_" + <number>`, where <number>
            is the one index beyond the current length of the zip_parameters list.

        Raises
        ------
        RydiquleError
            If fewer than 2 labels are provided.
        RydiquleError
            If any of the 2 labels are the same.
        RydiquleError
            If the label contains the substring `"gamma"`, as this is used internally
            for decoherence matrix generation.
        RydiquleError
            If any elements of `labels` are not labels of couplings in the sensor.
        RydiquleError
            If any of the parameters specified by labels are already zipped.
        RydiquleError
            If any of the parameters specified are not list-like.
        RydiquleError
            If all list-like parameters are not the same length.

        Notes
        -----
        .. note::
            This function should be called last after all Sensor couplings and dephasings
            have been added. Changing a coupling that has already been zipped removes it from
            the `self.zipped_parameters` list.

        .. note::
            Modifying the `Sensor._zip_labels` attribute directly can break some functionality
            and should be avoided. Use this function or :meth:`~.Sensor.unzip_parameters` instead.

        .. note::
            When defining the zip strings for states labelled with strings, be sure to additional
            `'` or `"` characters on either side of the labels, as demonstrated in the second
            example below.

        Examples
        --------
        >>> s = rq.Sensor(3)
        >>> det = np.linspace(-1,1,11)
        >>> s.add_coupling(states=(0,1), detuning=det, rabi_frequency=1, label="probe")
        >>> s.add_coupling(states=(1,2), detuning=det, rabi_frequency=1)
        >>> s.zip_parameters({"probe":"detuning", (1,2):"detuning"}, zip_label="detunings")
        >>> print(s._zip_labels) #NOT modifying directly
        ['detunings']
        >>> print(s.couplings.edges(data="detunings"))
        [(0, 1, 'detuning'), (1, 2, 'detuning')]
        >>> print(s.get_hamiltonian().shape)#zipped parameters share an axis
        (11, 3, 3)

        Especially when states are labelled with tuples, specifying zips parameters with the
        states they couple can be cumbersome. In this case, it can be useful to either assign
        variables to the tuples defining the states, or to label the couplings.

        >>> g = (0,0)
        >>> e1, e2 = (1,-1), (1, 1)
        >>> s = rq.Sensor([g, e1, e2])
        >>> arr = np.linspace(-1,1,11)
        >>> s.add_coupling((g,e1), detuning=arr, rabi_frequency=1, label="probe")
        >>> s.add_coupling((e1,e2), detuning=arr, rabi_frequency=1, label="coupling")
        >>> s.zip_parameters({((0,0),(1,-1)):"detuning", ((1,-1), (1, 1)):"detuning"}, zip_label="foo") #clunky
        >>> print(s._zip_labels)
        ['foo']
        >>> s.unzip_parameters("foo")
        >>> s.zip_parameters({"probe":"detuning", "coupling":"detuning"}, zip_label="bar") #readable
        >>> print(s._zip_labels)
        ['bar']

        For maximum flexibility, any parameters specified as arrays with matching lengths can
        be zipped. This should be used with care, as some parameter combinations can be
        nonsensical.

        >>> s = rq.Sensor(3)
        >>> arr = np.linspace(-1,1,11)
        >>> s.add_energy_shift(0, 0.5*arr)
        >>> s.add_coupling(states=(0,1), detuning=arr, rabi_frequency=1, label="probe")
        >>> s.add_coupling(states=(1,2), detuning=arr, rabi_frequency=1)
        >>> s.add_decoherence((1,0), 0.1*arr)
        >>> s.zip_parameters({(0,0):"e_shift", "probe":"detuning", (1,2):"detuning", (1,0):"gamma"}, zip_label="foo")
        >>> print(s._zip_labels) #NOT modifying directly
        ['foo']
        >>> print(s.couplings.edges(data="foo"))
        [(0, 0, 'e_shift'), (0, 1, 'detuning'), (1, 2, 'detuning'), (1, 0, 'gamma')]
        >>> print(s.get_hamiltonian().shape)
        (11, 3, 3)

        """
        #give a dummy label if not provided
        if zip_label is None:
            zip_label = "zip_" + str(len(self._zip_labels))

        #check for protected label
        for protected_label in self.protected_labels:
            if zip_label.find(protected_label) > -1:
                raise RydiquleError(f"Label {zip_label} contains protected string {protected_label}")

        #check for protected label
        for protected_label in self.scannable_parameters:
            if zip_label == protected_label:
                raise RydiquleError(f"Label {zip_label} is protected and cannot be a zip label")

        #ensure zip label does not already exist
        if zip_label in self._zip_labels:
            raise RydiquleError(f"Parameters already zipped with label {zip_label}. "
                                "Zip labels must be unique.")

        # check for at least 2 labels
        if len(parameters) < 2:
            raise RydiquleError(("Please provide at least 2 parameter labels "
                                 f"to zip (only provided {len(parameters)})"))

        #check that all labels are unique
        if len(parameters) != len(set(parameters)): #set will be shorter if there are duplicates
            raise RydiquleError("parameters cannot be zipped to themselves")

        #check if any labels are already zipped
        if any([l1==l2 for l1, l2 in itertools.product(self._zip_labels, parameters)]):
            raise RydiquleError("Parameter already zipped!")

        #ensure provided labels are valid for zipping
        previous_len = 0
        for coupling, p in parameters.items():

            #make sure the label exists
            try:
                states = self._coupling_with_label(coupling)
            except RydiquleError:
                raise RydiquleError(f"{coupling} is not a label of any coupling in this sensor")

            #make sure parameter exists and is an array
            try:
                parameter_val = self.couplings.edges[states][p]
            except KeyError as err:
                raise RydiquleError(f"Coupling {coupling} has no parameter {p}") from err

            #make sure array-defined parameters are the same length
            try:
                current_len = len(parameter_val)
            except TypeError:
                raise RydiquleError(f"Parameter {coupling}:{p} is not an array and cannot be zipped")

            if previous_len > 0 and current_len != previous_len:

                raise RydiquleError(
                    f"Got length {current_len} for parameter \"{coupling}\":\" {p}\", "
                    f"but should be length {previous_len}")

            previous_len = current_len

            self.couplings.edges[states][zip_label] = p


        self._zip_labels.append(zip_label)


    def zip_zips(self, *zip_labels: str, new_label: Optional[str]=None):
        """Combine multiple parameter zips into a single zip.

        Given any number of labels of zips in the sensor, combines them so that they will
        all share a single axis in the stack. Note that this will override all previous
        zips in `zip_labels`, and they cannot be recovered.

        Parameters
        ----------
        new_label : string, optional
            Label for the new zip that will replace the ones provided in `zip_labels`.
            If None, will be generated by joining all the strings of `zip_labels` with
            a "_" character, by default None.

        Raises
        ------
        RydiquleError
            If any of `zip_labels` do not exist in the `Sensor`.
        RydiquleError
            If `new_label` contains a protected substring (such as "gamma").
        RydiquleError
            If any of `zip_labels` are the same.
        RydiquleError
            If any of the dimensions of the axes specified by `zip_labels` do not match.

        Examples
        --------
        >>> s = rq.Sensor(5)
        >>> det = np.linspace(-1, 1, 11)
        >>> s.add_coupling((0, [1,2]), rabi_frequency=1, detuning=det, label="foo")
        >>> s.add_coupling((0, [3,4]), rabi_frequency=1, detuning=det, label="bar")
        >>> print(s.get_hamiltonian().shape)
        (11, 11, 5, 5)
        >>> print(s.axis_labels())
        ['bar_detuning', 'foo_detuning']
        >>> s.zip_zips("foo_detuning", "bar_detuning", new_label="foobar_detuning")
        >>> print(s.get_hamiltonian().shape)
        (11, 5, 5)
        >>> print(s.axis_labels())
        ['foobar_detuning']

        """

        #set default label
        if new_label is None:
            new_label = "_".join([l for l in zip_labels])

        #Check that all zips are actually in the system
        for l in zip_labels:
            if l not in self._zip_labels:
                raise RydiquleError(f"No zip labeled {l}")

        # Check the label does not contain protected substrings
        for protected_label in self.protected_labels:
            if new_label.find(protected_label) > -1:
                raise RydiquleError(f"Label {new_label} contains protected substring \"{protected_label}\"")

        #check that all labels are unique
        if len(zip_labels) != len(set(zip_labels)): #set will be shorter if there are duplicates
            raise RydiquleError("Zips cannot be zipped to themselves")

        # Check that all zips have the same dimensionality
        axis_labels = self.axis_labels()
        stack_shape = self._stack_shape()

        axis_lengths = np.array(
            [ax_size for ax_size, ax_label in zip(stack_shape, axis_labels)
             if ax_label in zip_labels]
        )
        if not np.all(axis_lengths==axis_lengths[0]):
            raise RydiquleError(f"Got mismatching dimensions {axis_lengths} for zips")

        for z_label in zip_labels:
            #generate a list of all couplings that are part of zip z_label
            couplings = [(s1, s2, param)
                         for s1, s2, param in self.couplings.edges(data=z_label)
                         if param is not None]

            for s1, s2, param in couplings:
                del self.couplings[s1][s2][z_label]
                self.couplings[s1][s2][new_label] = param

            self._zip_labels.remove(z_label)

        self._zip_labels.append(new_label)


    def unzip_parameters(self, zip_label: str, verbose: Optional[bool]=True):

        """
        Remove a set of zipped parameters from the internal zip_labels list.

        If an element of the internal `_zip_labels` array matches the label provided,
        removes it from `_zip_labels`. If no such element is present
        in `_zip_labels`, does nothing, and prints a message (disabled with `verbose=False`)

        Parameters
        ----------
        zip_label : str
            The string label corresponding the key to be deleted in the `_zip_labels`
            attribute.

        verbose : bool
            Whether to print a message if the unzip fails due to the specified `zip_label`
            not being a zip in the sensor. If `True` prints a message to std out if `zip_label`
            is not an element of the internal `self._zip_labels`. Otherwise, fails silently.
            Can be used if unzipping as part of an automated script.

        Notes
        -----
        .. note::
            This function should always be used rather than modifying the `_zip_labels`
            attribute directly.

        Examples
        --------
        >>> s = rq.Sensor(3)
        >>> det = np.linspace(-1,1,11)
        >>> s.add_coupling(states=(0,1), detuning=det, rabi_frequency=1, label="probe")
        >>> s.add_coupling(states=(1,2), detuning=det, rabi_frequency=1)
        >>> s.zip_parameters({"probe":"detuning", (1,2):"detuning"}, zip_label="demo1")
        >>> print(s._zip_labels) #NOT modifying directly
        ['demo1']
        >>> print(s.couplings.edges(data="demo1"))
        [(0, 1, 'detuning'), (1, 2, 'detuning')]
        >>> s.unzip_parameters("demo1")
        >>> print(s._zip_labels) #NOT modifying directly
        []
        >>> print(s.couplings.edges(data="demo1"))
        [(0, 1, None), (1, 2, None)]

        If the labels provided are not a match, a message is printed and nothing is altered.
        In the case where simulations are scripted and the printed message is annoying, the
        print behaviour can be modified with `verbose=False`, potentially useful for scripting
        cases where the desired behavior is to silently countinue over non-existant zip labels.

        >>> s = rq.Sensor(3)
        >>> det = np.linspace(-1,1,11)
        >>> s.add_coupling(states=(0,1), detuning=det, rabi_frequency=1, label="probe")
        >>> s.add_coupling(states=(1,2), detuning=det, rabi_frequency=1)
        >>> s.zip_parameters({"probe":"detuning", (1,2):"detuning"})
        >>> print(s._zip_labels) #NOT modifying directly
        ['zip_0']
        >>> print(s.couplings.edges(data="zip_0"))
        [(0, 1, 'detuning'), (1, 2, 'detuning')]
        >>> s.unzip_parameters("zipp0")
        No label matching zipp0, no action taken
        >>> print(s._zip_labels) #NOT modifying directly
        ['zip_0']
        >>> print(s.couplings.edges(data="zip_0"))
        [(0, 1, 'detuning'), (1, 2, 'detuning')]

        """
        try:
            self._zip_labels.remove(zip_label)
        except ValueError:
            if verbose:
                print(f"No label matching {zip_label}, no action taken")
                return

        for edge in self.couplings.edges():
            try:
                del self.couplings.edges[edge][zip_label]
            except KeyError:
                pass


    def add_decoherence(self, statespecs: StateSpecs, gamma: ScannableParameter, **kwargs):
        """Add a coupling between states or groups of states.

        Wraps the :meth:`~.Sensor.add_single_decoherence` and :meth:`~.Sensor.add_decoherence_group`
        functions, and dispatches to the appropriate one depending on the formatting of the `states`
        argument. Additional keyword arguments will be passed unmodified to the relevant
        method. See documentation of those functions for details on keyword argument options.

        Parameters
        ----------
        states : tuple of StateSpec
            The states or state manifolds of the decoherent coupling. If both are integers or
            string patterns matching a single state in the `Sensor`,
            :meth:`~.Sensor.add_single_decoherence` is dispatched. If either argument is a
            specification matching multiple states, :meth:`~.Sensor.add_decoherence_group` is
            dispatched.
        gamma : float or Sequence
            The decoherence rate, in Mrad/s.
        **kwargs :
            Additional keyword arguments passed to the appropriate function. See documentation for
            :meth:`~.Sensor.add_single_decoherence` and :meth:`~.Sensor.add_decoherence_group` for
            more details on valid keyword arguments.

        Examples
        --------
        >>> s = rq.Sensor(3)
        >>> s.add_coupling(states=(0,1), detuning=1, rabi_frequency=1)
        >>> s.add_coupling(states=(1,2), detuning=1, rabi_frequency=1)
        >>> s.add_decoherence((2,0), 0.1, label="misc")
        >>> print(s.decoherence_matrix())
        [[0.  0.  0. ]
         [0.  0.  0. ]
         [0.1 0.  0. ]]

        To add multiple decoherence effects to the same term, use a different label for each.

        >>> s = rq.Sensor(3)
        >>> s.add_coupling(states=(0,1), detuning=1, rabi_frequency=1)
        >>> s.add_coupling(states=(1,2), detuning=1, rabi_frequency=1)
        >>> s.add_decoherence((2,0), 0.1, label='foo')
        >>> s.add_decoherence((2,0), 0.15, label='bar')
        >>> print(s.decoherence_matrix())
        [[0.   0.   0.  ]
         [0.   0.   0.  ]
         [0.25 0.   0.  ]]

        Just like coherent coupling parameters, decoherence values can be passed as list-like objects
        and scanned. This adjusts the hamiltonian shape for clear broadcasting.

        >>> s = rq.Sensor(3)
        >>> gamma = np.linspace(0,0.5,11)
        >>> s.add_coupling(states=(0,1), detuning=1, rabi_frequency=1)
        >>> s.add_coupling(states=(1,2), detuning=1, rabi_frequency=1)
        >>> s.add_decoherence((2,0), gamma)
        >>> print(s.decoherence_matrix().shape)
        (11, 3, 3)
        >>> print(s.get_hamiltonian().shape)
        (11, 3, 3)

        Upper and lower states can also be regex strings matched against states in the Sensor just like for
        coherent couplings.

        >>> s = rq.Sensor(['g','e1','e2'])
        >>> s.add_coupling(states=('g','e1'), detuning=1, rabi_frequency=1)
        >>> s.add_coupling(states=('e1','e2'), detuning=1, rabi_frequency=1)
        >>> gamma = np.linspace(0, 0.3, 3)
        >>> cc = {('e1','g'):0.25, ('e2','g'):0.75}
        >>> s.add_decoherence((['e1','e2'],'g'), gamma, label="test", coupling_coefficients=cc)
        >>> print(s.decoherence_matrix())
        [[[0.     0.     0.    ]
          [0.     0.     0.    ]
          [0.     0.     0.    ]]
        <BLANKLINE>
         [[0.     0.     0.    ]
          [0.0375 0.     0.    ]
          [0.1125 0.     0.    ]]
        <BLANKLINE>
         [[0.     0.     0.    ]
          [0.075  0.     0.    ]
          [0.225  0.     0.    ]]]

        Also just like coherent couplings, decoherent coupling can be defined over manifolds
        using state specifications. The interface is identical.

        >>> g = (0,[-1,1])
        >>> e = (1,[-1,1])
        >>> cc = {
        ...    ((1,-1),(0,-1)):1,
        ...    ((1,1),(0,-1)):2,
        ...    ((1,-1),(0,1)):1,
        ...    ((1,1),(0,1)):2,
        ... }
        >>> s = rq.Sensor([g,e])
        >>> s.add_coupling((g,e), detuning=1, rabi_frequency=1, label='foo')
        >>> s.add_decoherence((e,g), 0.1, coupling_coefficients=cc, label='bar')
        >>> print(s.decoherence_matrix())
        [[0.  0.  0.  0. ]
         [0.  0.  0.  0. ]
         [0.1 0.1 0.  0. ]
         [0.2 0.2 0.  0. ]]

        """
        #pass adding edges if gamma is 0
        if np.all(gamma==np.zeros_like(gamma)):
            pass

        #get relevant integer states from string label patterns
        try:
            states_list1 = self.states_with_spec(statespecs[0])
            states_list2 = self.states_with_spec(statespecs[1])
        except ValueError:
            raise ValueError("states1 and states2 must be valid state specifications")

        if len(states_list1) == len(states_list2) == 1:
            self.add_single_decoherence((states_list1[0], states_list2[0]), gamma, **kwargs)
            return

        self.add_decoherence_group(states_list1, states_list2, gamma, **kwargs)


    def add_single_decoherence(self, states: States, gamma: ScannableParameter,
                               decoherent_cc: float=1.0, label: Optional[str] = None):
        """
        Add decoherent coupling to the graph between two states.

        If `gamma` is list-like, the array generated by :meth:`~.Sensor.decoherence_matrix` will
        contain decoherence matrices for every combination of decoherence values provided. This
        functionality mirrors hamiltonian generation when parameters of
        :meth:`~.Sensor.add_coupling` are list-like. Note that if `gamma` is 0 or an array
        of zeros, the associated edge key will be left off the graph.

        Parameters
        ----------
        states : tuple of State
            Length-2 tuple of integers corresponding to the two states. The first
            value is the number of state out of which population decays, and the
            second is the number of the state into which population decays.
        gamma : float or sequence
            The decay rate, in Mrad/s.
        decoherent_cc : float
            The value by which `gamma` is multiplied before it is added to the graph. Typically only used
            by :meth:`~.Sensor.add_decoherence_group`, but made transparent for scripting purposes.
            Defaults to 1.0
        label : str or None, optional
            Optional label for the decay. If `None`, decay will be stored on
            the graph edge as `"gamma"`. Otherwise, will cast as a string and decay will be stored
            on the graph edge as `"gamma_"+label`.

        Notes
        -----
        .. note::
            Adding a decoherece with a particular label (including `None`) will override an existing
            decoherent transition with that label.

        Examples
        --------
        >>> s = rq.Sensor(3)
        >>> s.add_coupling(states=(0,1), detuning=1, rabi_frequency=1)
        >>> s.add_coupling(states=(1,2), detuning=1, rabi_frequency=1)
        >>> s.add_single_decoherence((2,0), 0.1, label="misc")
        >>> print(s.decoherence_matrix())
        [[0.  0.  0. ]
         [0.  0.  0. ]
         [0.1 0.  0. ]]

        To add multiple decoherence effects to the same term, provida a differnt label for each.

        >>> s = rq.Sensor(3)
        >>> s.add_coupling(states=(0,1), detuning=1, rabi_frequency=1)
        >>> s.add_coupling(states=(1,2), detuning=1, rabi_frequency=1)
        >>> s.add_single_decoherence((2,0), 0.1, label='foo')
        >>> s.add_single_decoherence((2,0), 0.15, label='bar')
        >>> print(s.decoherence_matrix())
        [[0.   0.   0.  ]
         [0.   0.   0.  ]
         [0.25 0.   0.  ]]

        Decoherence values can also be scanned. Here decoherece from states 2->0 is scanned
        between 0 and 0.5 for 11 values. We can also see how the Hamiltonian shape accounts
        for this to allow for clean broadcasting, indicating that the hamiltonian is identical
        across all decoherence values.

        >>> s = rq.Sensor(3)
        >>> gamma = np.linspace(0,0.5,11)
        >>> s.add_coupling(states=(0,1), detuning=1, rabi_frequency=1)
        >>> s.add_coupling(states=(1,2), detuning=1, rabi_frequency=1)
        >>> s.add_single_decoherence((2,0), gamma)
        >>> print(s.decoherence_matrix().shape)
        (11, 3, 3)
        >>> print(s.get_hamiltonian().shape)
        (11, 3, 3)

        """
        if label is None:
            label_full = "gamma"
        else:
            label_full = "gamma_" + str(label)

        states = self._states_valid(states)
        # coerce gamma to numpy array if a sequence
        if isinstance(gamma, Sized):
            gamma = np.array(gamma)

        gamma_full = decoherent_cc*gamma
        if np.all(gamma_full==0.0):
            return

        self.couplings.add_edge(*states, **{label_full:gamma_full})
        if debug_state():
            print(f'  Added decoherence for {states}')

        # if edge doesn't have a label (ie decoherence only), add a default label
        if self.couplings.edges[states].get("label") is None:
            self.couplings.edges[states]["label"] = state_tuple_to_str(states)


    def add_decoherence_group(self, states1: List[State], states2: List[State],
                              gamma: ScannableParameter, label: str,
                              coupling_coefficients: Optional[Dict[States,float]] = None):
        """Adds a group of dechorences to the Sensor.

        Given 2 lists of states, adds a single coupling across each combination of states
        between the first and second lists. Then, if gamma is a array-like of values, automatically
        performs :meth:`~.Sensor.zip_parameters` on all decoherences added as part of this
        funtion so they share an axis when :meth:`~.Sensor.decoherence_matrix` is called.

        Scaling multiplicative factors for `gamma` must be applied per pair of states
        using `decoherent_cc`, a dictionary of coefficients determining coupling strengths.
        If a pair is not in `decoherent_cc`, it is assumed to have a coupling coefficient of
        zero, and will be omitted from the graph. If `decoherent_cc` is `None`, all
        couplings are assumed to have a relative strength of 1.

        Parameters
        ----------
        states1 : List of State
            The list of states out of which population is decaying. Each element of the list
            must be a state in this `Sensor`.
        states2 : List of State
             The list of states into which population is decaying. Each element of the list
            must be a state in this `Sensor`.
        label : str
            Required string label denoting what the group of dephasings is called. Used to
            apply a label to the zip.
        gamma : ScannableParameter
            Base decoherence rate between the two groups of states, in units of Mrad/s.
            Mutliplied by the corresponding values in the `decoherent_coupling` dictionary.
        coupling_coefficients : dict, optional
            Coefficiants describing the relative coupling strengths for decoherences in the group.
            Treated as modifications to the "base" dephasing rate specified by the `gamma`
            argument. The gamma of individual decoherences will be the `gamma` argument
            multiplied by the corresponding value in this dictionary.
            If `None`, all couplings in the group are assumed to have a coefficient of 1.0
            if specified, all unspecified coupling pairs are ignored.

        Raises
        ------
        ValueError
            If the either of the states strings provided cannot be parsed as a regex pattern
        ValueError
            If `states1` and `states2` only have one state. Use :meth:`~.Sensor.add_decoherence` instead.

        Examples
        --------
        >>> s = rq.Sensor(['g','e1','e2'])
        >>> s.add_coupling(states=('g','e1'), detuning=1, rabi_frequency=1)
        >>> s.add_coupling(states=('e1','e2'), detuning=1, rabi_frequency=1)
        >>> cc = {('e1','g'):0.25, ('e2','g'):0.75}
        >>> s.add_decoherence_group(['e1','e2'],['g'], 0.1, "test", coupling_coefficients=cc)
        >>> print(s.decoherence_matrix())
        [[0.    0.    0.   ]
         [0.025 0.    0.   ]
         [0.075 0.    0.   ]]

        Unlike :meth:`Sensor.add_decoherence`, this function does not accept state specifications.
        Upper and lower states must be passed as lists. As this tends to be a little clunkier,
        :meth:`Sensor.add_decoherence` is usually preferred.

        >>> g = (0,[-1,1])
        >>> e = (1, [-1,1])
        >>> list_g = [(0, -1), (0, 1)]
        >>> list_e = [(1, -1), (1, 1)]
        >>> cc = {
        ...    ((1,1),(0,1)): 0.4,
        ...    ((1,1),(0,-1)): 0.1,
        ...    ((1,-1),(0,1)): 0.1,
        ...    ((1,-1),(0,-1)): 0.4
        ... }
        >>> s = rq.Sensor([g,e])
        >>> print(s.states)
        [(0, -1), (0, 1), (1, -1), (1, 1)]
        >>> s.add_decoherence_group(list_e, list_g, 0.1, "foo", coupling_coefficients=cc)
        >>> print(s.decoherence_matrix())
        [[0.   0.   0.   0.  ]
         [0.   0.   0.   0.  ]
         [0.04 0.01 0.   0.  ]
         [0.01 0.04 0.   0.  ]]

        """
        labels = []
        if coupling_coefficients is None:
            coupling_coefficients = {(s1,s2):1 for s1, s2 in itertools.product(states1, states2)}

        if len(states1) == 1 and len(states2) == 1:
            raise ValueError('Both states groups only have one state. Use add_decoherence instead')

        if isinstance(gamma, Sized):
            # must cast sequences to numpy array to multiply by scalars
            gamma = np.asarray(gamma)

        for s1, s2 in itertools.product(states1, states2):
            coupling_label = state_tuple_to_str((s1,s2))

            # apply CG with default of 0 if not specified
            gamma_cc = coupling_coefficients.get((s1,s2))
            # skip coupling if coefficient not supplied
            if gamma_cc is None:
                continue

            self.add_decoherence((s1,s2), gamma_cc * gamma, label=label)
            labels.append(coupling_label)

        if hasattr(gamma, "__len__"):
            #reconstruct the label that will be added to the graph in add_single decoherence
            if label is None:
                gamma_label = "gamma"
            else:
                gamma_label = "gamma_" + str(label)
            zip_labels: _ZP = {l:gamma_label for l in labels}
            self.zip_parameters(zip_labels, zip_label=label)


    def add_transit_broadening(self, gamma_transit: ScannableParameter,
                               repop: Optional[Union[Dict[State, float], List[State]]] = None,
                               label: str = "transit"):
        """
        Adds transit broadening by adding a decoherence from each node to ground.

        For each state n, adds a decoherent transition from n to each state in the
        keys of the `repop` dictionary using the :meth:`~.Sensor.add_decoherence`
        method with provided label (`"transit"`by default)
        See :meth:`~.Sensor.add_decoherence` for more details on labeling.

        If an array of transit values are provided, they will be automatically zipped together
        into a single scanning element.

        Parameters
        ----------
        gamma_transit: float or sequence
            The transit broadening rate in Mrad/s.
        repop: dict, optional
            Dictionary of states for transit to repopulate in to.
            The keys represent tshe state labels. The values represent
            the fractional amount that goes to that state.
            If the sum of value does not equal 1, population will not be conserved.
            Default is to repopulate everything into the ground state (either state 0
            or the first state in the basis passed to the :meth:`~.Sensor.__init__` method).
            If `None`, all population decays to the ground state, defined as the first state
            in the state list passed to the constuctor. Defaults to `None`.
        label: str, optional
            Label to be passed to :meth:`~.Sensor.add_decoherence`. Defaults to "transit"

        Warns
        -----
        PopulationNotConservedWarning
            If the values of the `repop` parameter do not sum to 1, thus meaning
            population will not be conserved.

        Examples
        --------
        >>> s = rq.Sensor(3)
        >>> s.add_transit_broadening(0.1)
        >>> print(s.couplings.edges(data=True))
        [(0, 0, {'gamma_transit': 0.1, 'label': '(0,0)'}),
         (1, 0, {'gamma_transit': 0.1, 'label': '(1,0)'}),
         (2, 0, {'gamma_transit': 0.1, 'label': '(2,0)'})]
        >>> print(s.decoherence_matrix())
        [[0.1 0.  0. ]
         [0.1 0.  0. ]
         [0.1 0.  0. ]]

        >>> s = rq.Sensor(['g', 'e1', 'e2'])
        >>> repop = {'g':0.75, 'e1': 0.25}
        >>> s.add_transit_broadening(0.2, repop=repop)
        >>> print(s.decoherence_matrix())
        [[0.15 0.05 0.  ]
         [0.15 0.05 0.  ]
         [0.15 0.05 0.  ]]

        """
        #all decay to ground state
        if repop is None:
            ground_state = self.states[0]
            repop = {ground_state: 1.0}
        #decay evenly across ground manifold
        elif isinstance(repop, list):
            ground_list = sum([self.states_with_spec(s) for s in repop], start=[])
            repop = {s:(1/len(ground_list)) for s in ground_list}

        if not isinstance(repop, dict):
            raise ValueError("'repop' argument must be 'None', list of ground statespecs, or dict")

        if isinstance(gamma_transit, Sized):
            # needed for multiplying branching ratios
            gamma_transit = np.asarray(gamma_transit)

        if not np.isclose(sum(repop.values()), 1.0):
            warnings.warn(('Repopulation branching ratios do not sum to 1!'
                           ' Population will not be conserved.'),
                           PopulationNotConservedWarning)

        for t, br in repop.items():
            for i in self.states:
                self.add_single_decoherence((i, t), gamma=gamma_transit*br, label="transit")

        if hasattr(gamma_transit, "__len__"):
            # need to zip together all the transit rates
            transit_parameters: _ZP = {l:"gamma_transit"
                                  for s1,s2,l in cast(Iterable[Tuple[State,State,str]],
                                                      self.couplings.edges(data="label"))
                                  if self.couplings.edges[s1,s2].get("gamma_transit") is not None}
            self.zip_parameters(transit_parameters, zip_label=label)


    def add_self_broadening(self, state: State, gamma: ScannableParameter,
                            label: str = "self",
                            decoherent_cc: Optional[Dict[States, float]] = None):
        """
        Specify self-broadening (such as collisional broadening) of a level.

        Equivalent to calling :meth:`~.Sensor.add_decoherence` and specifying both
        states to be the same, with the "self" label. For more complicated systems,
        it may be useful to further specify the source of self-broadening as, for
        example, "collisional" for easier bookkeeping and to ensure no values
        are overwritten.

        Parameters
        ----------
        state: State
            State or states to which the broadening will be added.
            Using a regular expression allows for specifying self broadening of a group of states.
            In this case, `mult-factor` is used to define relative amplitudes.
        gamma: float or sequence
            The broadening width to be added in Mrad/s.
        label: str, optional
            Optional label for the state. By default, decay will be stored on
            the graph edge as `"gamma_self"`. Otherwise, will cast as a string
            and decay will be stored on the graph edge as `"gamma_"+label`
        mult-factor: dict
            Dictionary mapping of the scaling factors to apply to the self broadening
            of each state in a group specified via regular expression.

        Notes
        -----
        .. note::
            Just as with the :meth:`~.Sensor.add_decoherence` function, adding a decoherence
            value with a label that already exists will overwrite an existing decoherent
            transition with that label. The "self" label is applied to this function
            automatically to help avoid an unwanted overwrite.

        Examples
        --------
        >>> s = rq.Sensor(3)
        >>> s.add_self_broadening(1, 0.1)
        >>> print(s.couplings.edges(data=True))
        [(1, 1, {'gamma_self': 0.1, 'label': '(1,1)'})]
        >>> print(s.decoherence_matrix())
        [[0.  0.  0. ]
         [0.  0.1 0. ]
         [0.  0.  0. ]]

        """
        states_list = self.states_with_spec(state)

        #handle the case of a single state or null spec
        if len(states_list) == 0:
            raise RydiquleError(f'No states matching {state}')
        elif len(states_list) == 1:
            self.add_single_decoherence((states_list[0], states_list[0]), gamma, label=label)
            return

        # handle group of states
        if decoherent_cc is None:
            decoherent_cc = {(s,s):1 for s in states_list}

        self.add_self_broadening_group(states_list, gamma, label=label, decoherent_cc=decoherent_cc)


    def add_self_broadening_group(self, states: List[State], gamma: ScannableParameter,
                                  label: str='self', decoherent_cc: Optional[dict]=None):
        """Specify self-broadening (such as collisional broadening) of a group of states.

        Equivalent to calling :meth:`~.Sensor.add_decoherence_group` and specifying both
        state groups to be the same, with the "self" label. For more complicated systems,
        it may be useful to use `label` to label the source of self-broadening as, for
        example, "collisional" for easier bookkeeping and to ensure no values
        are overwritten.

        Note that this function applies decohernce terms to every combiniation of states
        in the group, not just from each state to itsself.

        Parameters
        ----------
        states: list of State
            List of states to which the self-broadening is applied.
        gamma: ScannableParameter
            The broadening width to be added in Mrad/s.
        label: str, optional
            Optional label for the state. By default, decay will be stored on
            the graph edge as `"gamma_self"`. Otherwise, will cast as a string
            and decay will be stored on the graph edge as `"gamma_"+label`
        decoherent_cc: dict, optional
            Clebsch-Gordon-like coefficients for how gamma scales to different pairs of states
            within the group. Unspecified pairs are assumed to have coefficients of 0.
            Default value is None, which applies 0 to all coefficients.
        """

        if decoherent_cc is None:
            decoherent_cc = {(s,s):1.0 for s in states}

        self.add_decoherence_group(states, states, gamma, label, decoherent_cc)


    def decoherence_matrix(self) -> np.ndarray:
        """
        Build a decoherence matrix out of the decoherence terms of the graph.

        For each edge, sums all parameters with a key that begins with "gamma",
        and places it on the appropriate location in an adjacency matrix for the
        `couplings` graph.

        Returns
        -------
        numpy.ndarray
            The decoherence matrix stack of the system.

        Examples
        --------
        >>> s = rq.Sensor(3)
        >>> s.add_decoherence((1,0), 0.2, label="foo")
        >>> s.add_decoherence((1,0), 0.1, label="bar")
        >>> s.add_decoherence((2,0), 0.05)
        >>> s.add_decoherence((2,1), 0.05)
        >>> print(s.couplings.edges(data=True))
        [(1, 0, {'gamma_foo': 0.2, 'label': '(1,0)', 'gamma_bar': 0.1}), (2, 0, {'gamma': 0.05, 'label': '(2,0)'}), (2, 1, {'gamma': 0.05, 'label': '(2,1)'})]
        >>> print(s.decoherence_matrix())
        [[0.   0.   0.  ]
         [0.3  0.   0.  ]
         [0.05 0.05 0.  ]]

        Decoherences can be stacked just like any parameters of the Hamiltonian:

        >>> s = rq.Sensor(3)
        >>> gamma = np.linspace(0,0.5, 11)
        >>> s.add_decoherence((1,0), gamma)
        >>> print(s.decoherence_matrix().shape)
        (11, 3, 3)

        Defining decoherences between states labelled with string values works just like coherent couplings:

        >>> s = rq.Sensor(['g', 'e1', 'e2'])
        >>> s.add_decoherence(('e1', 'g'), 0.1)
        >>> s.add_decoherence(('e2', 'g'),0.1)
        >>> print(s.decoherence_matrix())
        [[0.  0.  0. ]
         [0.1 0.  0. ]
         [0.1 0.  0. ]]

        """
        self._expand_dims()

        int_states = {state: i for (i, state) in enumerate(self.states)}

        stack_shape = self._stack_shape()
        for states, param, arr, _ in self.variable_parameters(apply_mesh=True):
            self.couplings.edges[states][param] = arr

        gamma_shape = (*stack_shape, self.basis_size, self.basis_size)
        gamma_matrix = np.zeros(gamma_shape, np.float64)

        # get a list of all unique parameter labels containing "gamma"
        labels_lists = [list(d.keys()) for _,_,d in self.couplings.edges(data=True)]
        all_labels = list(set(sum(labels_lists, start=[])))
        decoherence_labels = [l for l in all_labels if "gamma" in l]

        # for each unique decoherence name,
        for label in decoherence_labels:
            for states, f in self.couplings_with(label).items():

                states_n = tuple([int_states[s] for s in states])
                idx = (...,*states_n)
                gamma_matrix[idx] += f[label]

        _squeeze_dims(self.couplings)
        return gamma_matrix


    def axis_labels(self) -> List[str]:
        """
        Get a list of axis labels for stacked hamiltonians.

        The axes of a hamiltonian
        stack are defined as the axes preceding the usual hamiltonian, which are always
        the last 2. These axes only exist if one of the parametes used to define
        a Hamiltonian are lists.

        Be default, labels which have been zipped using :meth:`~.Sensor.zip_parameters`
        will be combined into a single label, as this is how :meth:`~.Sensor.get_hamiltonian`
        treats these axes.

        The ordering of axis labels is as follows:

         - Zipped parameter (shared axes) appear before single parameters.
         - Zipped parameters are ordered alphabetically by label.
         - Single axes are sorted first by lower state, then by upper state, then
           alphabetically by parameter.

        Returns
        -------
        list of str
            Strings corresponding to the label of each axis on a stack
            of multiple hamiltonians.

        Examples
        --------
        There are no preceding axes if there are no list-like parameters.

        >>> s = rq.Sensor(3)
        >>> blue = {"states":(0,1), "rabi_frequency":1, "detuning":2}
        >>> red = {"states":(1,2), "rabi_frequency":3, "detuning":4}
        >>> s.add_couplings(blue, red)
        >>> print(s.get_hamiltonian().shape)
        (3, 3)
        >>> print(s.axis_labels())
        []

        Adding list-like parameters expands the hamiltonian

        >>> s = rq.Sensor(3)
        >>> det = np.linspace(-10, 10, 11)
        >>> blue = {"states":(0,1), "rabi_frequency":1, "detuning":det, "label":"blue"}
        >>> red = {"states":(1,2), "rabi_frequency":3, "detuning":det}
        >>> s.add_couplings(blue, red)
        >>> print(s.get_hamiltonian().shape)
        (11, 11, 3, 3)
        >>> print(s.axis_labels())
        ['blue_detuning', '(1,2)_detuning']

        The ordering of labels doesn't change if string state names are used. For single couplings,
        the ordering of axes is determined purely by the ordering of the states, regardless
        of coupling labels or string names of states.

        >>> s = rq.Sensor(['g', 'e1', 'e2'])
        >>> det = np.linspace(-10, 10, 11)
        >>> blue = {"states":('g','e1'), "rabi_frequency":1, "detuning":det, "label":"blue"}
        >>> red = {"states":('e1','e2'), "rabi_frequency":3, "detuning":det}
        >>> s.add_couplings(blue, red)
        >>> print(s.get_hamiltonian().shape)
        (11, 11, 3, 3)
        >>> print(s.axis_labels())
        ['blue_detuning', '(e1,e2)_detuning']

        Zipping parameters combines labels onto a single axis, since their Hamiltonians now
        lie on a single axis of the stack. The name of that axis will be the label provided to
        :meth:`~.Sensor.zipped_parameters`. Note that this will default to 'zip_<int>'. Here
        the axis of length 7 (axis 1) corresponds to the rabi frequencies and the axis of shape
        11 (axis 0) corresponds to the zipped detunings

        >>> s = rq.Sensor(3)
        >>> s.add_coupling(states=(0,1), detuning=np.arange(11), rabi_frequency=np.linspace(-3, 3, 7))
        >>> s.add_coupling(states=(1,2), detuning=0.5*np.arange(11), rabi_frequency=1)
        >>> s.zip_parameters({(0,1):"detuning", (1,2):"detuning"}, zip_label="detunings")
        >>> print(s.get_hamiltonian().shape)
        (11, 7, 3, 3)
        >>> print(s.axis_labels())
        ['detunings', '(0,1)_rabi_frequency']

        """
        parameter_groups = self.group_variable_parameters()
        axis_labels = ['' for _ in parameter_groups]

        for i,group in enumerate(parameter_groups):

            #if 1 entry, combine label+parameter name
            if len(group)==1:
                states, parameter, _, _ = group[0]
                label=self.couplings.edges[states]["label"]
                axis_labels[i] = label+"_"+parameter

            #if multiple, just extract zip (last entry in any list, here we just pick 1st)
            elif len(group) > 1:
                assert group[0][-1] is not None
                axis_labels[i] = group[0][-1]

            #something has gone horribly wrong
            else:
                raise ValueError(f"parameter {i} has no data")

        return axis_labels


    def variable_parameters(self, apply_mesh:bool = False,
                            ) -> List[Tuple[States, str, np.ndarray, Optional[str]]]:
        """
        Property to retrieve the values of parameters that were stored on the graph as arrays.

        Values are returned as a list of tuples in the standard order of pythons default sorting,
        applied first to the tuple indicating states and then to the key of the parameter itself.
        This means that couplings are sorted first by lower state, then by upper state, then
        alphabetically by the name of the parameter.To determine order, all state labels treated
        as their integer position in the basis as determined by ordering in the constructor
        :meth:`~.Sensor.__init__`.

        Returns
        -------
        list of tuples
            A list of tuples corresponding to the parameters of the systems that are variable
            (i.e. stored as an array). They are ordered according to states,
            then according to variable name.
            Tuple entries of the list take the form `(states, param_name, value)`

        Examples
        --------
        >>> s = rq.Sensor(3)
        >>> vals = np.linspace(-1,2,3)
        >>> s.add_coupling(states=(1,2), rabi_frequency=vals, detuning=1)
        >>> s.add_coupling(states=(0,1), rabi_frequency=vals, detuning=vals)
        >>> print(s.variable_parameters())
        [((0, 1), 'detuning', array([-1. ,  0.5,  2. ]), None),
         ((0, 1), 'rabi_frequency', array([-1. ,  0.5,  2. ]), None),
         ((1, 2), 'rabi_frequency', array([-1. ,  0.5,  2. ]), None)]

        The order is important; in the unzipped case, it will sort as though all state labels
        were cast to strings, meaning integers will always be treated as first.

        >>> s = rq.Sensor([0, 'e1', 'e2'])
        >>> det1 = np.linspace(-1, 1, 3)
        >>> det2 = np.linspace(-1, 1, 5)
        >>> blue = {"states":(0,'e1'), "rabi_frequency":1, "detuning":det1}
        >>> red = {"states":('e1','e2'), "rabi_frequency":3, "detuning":det2}
        >>> s.add_couplings(blue, red)
        >>> print(s.variable_parameters())
        [((0, 'e1'), 'detuning', array([-1.,  0.,  1.]), None),
         (('e1', 'e2'), 'detuning', array([-1. , -0.5,  0. ,  0.5,  1. ]), None)]
        >>> print(f"Axis Labels: {s.axis_labels()}")
        Axis Labels: ['(0,e1)_detuning', '(e1,e2)_detuning']

        """
        parameter_list: List[Tuple[States, str, np.ndarray, Optional[str]]] = []

        states: States
        for states, edge_data in self.couplings.edges.items():

            key: str
            for key, value in sorted(edge_data.items()):
                if not key.startswith("gamma") and key not in self.scannable_parameters:
                    continue

                if hasattr(value, "__len__"):

                    #test all key-value pairs for zip parameters
                    zip_label=None
                    for zip_label_test, zip_parameter_test in edge_data.items():
                        if zip_label_test in self._zip_labels and zip_parameter_test==key:
                            zip_label=zip_label_test
                            break

                    parameter_list.append((states, key, np.array(value), zip_label))

        parameter_list.sort(key=self.variable_parameter_sort)

        #no need to do remaining calculations if theres no extra stuff
        if not apply_mesh:
            return parameter_list

        #collect the index of each parameter
        zip_labels=[l if l is not None
                    else i
                    for i,(_,_,_,l) in enumerate(parameter_list)]
        zip_labels_unique = [l for i,l in enumerate(zip_labels) if l not in zip_labels[:i]]
        axis_indeces = [zip_labels_unique.index(l) for l in zip_labels]
        try:
            n_dim = max(axis_indeces)+1
        except ValueError:
            n_dim = 0

        #apply meshgrid to parameters
        #basically a manual implementation of np.meshgrid(..., indexing='ij', sparse=True)
        #but with some stuff sharing an axis
        if apply_mesh:
            for i,(_,_,values,_) in enumerate(parameter_list):

                arr_shape = np.ones(n_dim, dtype=int)
                arr_shape[axis_indeces[i]] = values.size
                parameter_list[i][2].shape = tuple(arr_shape)

        return parameter_list


    def group_variable_parameters(self, apply_mesh: bool = False,
                                  ) -> List[List[Tuple[States, str, np.ndarray, Optional[str]]]]:

        variable_parameters = self.variable_parameters(apply_mesh)
        #collect the index of each parameter
        zip_labels=[l if l is not None
                    else i
                    for i,(_,_,_,l) in enumerate(variable_parameters)]
        zip_labels_unique = [l for i,l in enumerate(zip_labels) if l not in zip_labels[:i]]
        axis_indeces = [zip_labels_unique.index(l) for l in zip_labels]
        try:
            n_dim = max(axis_indeces)+1
        except ValueError:
            n_dim = 0

        grouped_parameters: List[List[Tuple[States, str, np.ndarray, Optional[str]]]] = [[] for _ in range(n_dim)]
        for p,i in zip(variable_parameters, axis_indeces):
            grouped_parameters[i].append(p)

        return grouped_parameters


    def get_parameter_mesh(self) -> List[np.ndarray]:
        """
        Returns the parameter mesh of the sensor.

        The parameter mesh is the flattened grid of variable parameters
        in all the couplings of a sensor.
        Wraps `numpy.meshgrid` with the `indexing` argument
        always `"ij"` for matrix indexing.

        Returns
        -------
        list of numpy.ndarray
            list of mesh grids for every variable parameter

        Examples
        --------
        >>> s = rq.Sensor(3)
        >>> rabi1 = np.linspace(-1,1,11)
        >>> rabi2 = np.linspace(-2,2,21)
        >>> s.add_coupling(states=(0,1), rabi_frequency=rabi1, detuning=1)
        >>> s.add_coupling(states=(1,2), rabi_frequency=rabi2, detuning=1)
        >>> for p in s.get_parameter_mesh():
        ...     print(p.shape)
        (11, 1)
        (1, 21)

        """
        parameter_mesh = [v for _,_,v,_ in self.variable_parameters(apply_mesh=True)]

        return parameter_mesh


    def get_hamiltonian(self) -> np.ndarray:
        """
        Creates the Hamiltonians from the couplings defined by the fields.

        They will only be the steady state hamiltonians, i.e. will only contain
        terms which do not vary with time. Implicitly creates hamiltonians in "stacks"
        by creating a grid of all supported coupling parameters which are lists.
        This grid of parameters will not contain rabi-frequency parameters which
        vary with time and are defined as list-like. Rather, the associated axis
        will be of length 1, with the scanning over this value handled by the
        :meth:`~Sensor.get_time_couplings` function.

        For m list-like parameters x1,x2,...,xm with shapes N1,N2,...,Nm, and basis
        size n, the output will be shape `(N1,N2,...,Nm, n, n)`. The dimensions
        N1,N2,...Nm are labeled by the output of :meth:`~.Sensor.axis_labels`.

        If any parameters have been zipped with the :meth:`~.Sensor._zip_parameters`
        method, those parameters will share an axis in the final hamiltonian stack.
        In this case, if axis N1 and N2 above are the same shape and zipped, the final
        Hamiltonian will be of shape `(N1,...,Nm, n, n)`.

        In the case where the basis of the `Sensor` was explicitly defined with a list
        of states, the ordering of rows and coulumns in the hamiltonian corresponds to the
        ordering of states passed in the basis.

        See rydiqule's conventions for matrix stacking for more details.

        Returns
        -------
        np.ndarray
            The complex hamiltonian stack for the sensor.

        Examples
        --------
        >>> s = rq.Sensor(3)
        >>> det = np.linspace(-1,1,11)
        >>> blue = {"states":(0,1), "rabi_frequency":1, "detuning":det}
        >>> red = {"states":(1,2), "rabi_frequency":3, "detuning":det}
        >>> s.add_couplings(red, blue)
        >>> print(s.get_hamiltonian().shape)
        (11, 11, 3, 3)

        Time dependent couplings are handled separately. The axis that contains array-like
        parameters with time dependence is length 1 in the steady-state Hamiltonian.

        >>> s = rq.Sensor(3)
        >>> rabi = np.linspace(-1,1,11)
        >>> step = lambda t: 0 if t<1 else 1
        >>> blue = {"states":(0,1), "rabi_frequency":rabi, "detuning":1}
        >>> red = {"states":(1,2), "rabi_frequency":rabi, "detuning":0, 'time_dependence': step}
        >>> s.add_couplings(red, blue)
        >>> print(s.get_hamiltonian().shape)
        (11, 1, 3, 3)

        Zipping parameters means they share an axis in the Hamiltonian.

        >>> s = rq.Sensor(3)
        >>> s.add_coupling(states=(0,1), detuning=np.arange(11), rabi_frequency=2)
        >>> s.add_coupling(states=(1,2), detuning=0.5*np.arange(11), rabi_frequency=1)
        >>> s.zip_parameters({(0,1):"detuning", (1,2):"detuning"})
        >>> H = s.get_hamiltonian()
        >>> print(H.shape)
        (11, 3, 3)

        If the basis is provided as a list of string labels, the ordering of Hamiltonian rows
        and columns will correspond to the order of states provided.

        >>> s = rq.Sensor(['g', 'e1', 'e2'])
        >>> s.add_coupling(('g', 'e1'), detuning=1, rabi_frequency=1)
        >>> s.add_coupling(('e1', 'e2'), detuning=1.5, rabi_frequency=1)
        >>> print(s.get_hamiltonian())
        [[ 0. +0.j  0.5+0.j  0. +0.j]
         [ 0.5-0.j -1. +0.j  0.5+0.j]
         [ 0. +0.j  0.5-0.j -2.5+0.j]]

        """
        #adjust array parameters to be the appropriate shape
        self._expand_dims()

        #dictionary of state ordering used to determine indeces in ham
        int_states = {state: i for (i, state) in enumerate(self.states)}

        stack_shape = self._stack_shape(time_dependence='steady')

        hamiltonian_shape = (*stack_shape, self.basis_size, self.basis_size)
        # returns diagonal elements of Hamiltonian
        transition_frequencies = self.get_transition_frequencies()
        #define hamiltonian and place terms from above on diagonal
        hamiltonian = np.zeros(hamiltonian_shape, np.complex128)
        np.einsum("...ii->...i", hamiltonian)[:] = transition_frequencies

        for states, f in self.couplings_with('time_dependence', method='not any').items():

            if 'rabi_frequency' not in f:
                continue

            #convert the state label to an index of position in ham
            states_n = tuple([int_states[s] for s in states])
            idx = (...,*states_n)
            conj_idx = (...,*states_n[::-1])

            #get the coupling coeffiecient to multiply the rabi frequency by from the graph
            cc = self.couplings.edges[states].get('coherent_cc', 1.0)
            # factor of 1/2 accounts for implicit rotating wave approximation
            hamiltonian[idx] = cc * f['rabi_frequency']*np.exp(1j*f['phase'])/2
            hamiltonian[conj_idx] = np.conj(hamiltonian[idx])

        #add the numerical diagonal shifts
        for state1, state2, shift in nx.selfloop_edges(self.couplings, data='e_shift', default=0):

            shift_state = int_states[state1]
            idx = (..., shift_state, shift_state)

            hamiltonian[idx] += shift

        #restore parameters to 1d arrays
        _squeeze_dims(self.couplings)

        return hamiltonian


    def get_time_hamiltonian_components(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Get time-dependent components of the hamiltonian.

        Returns the list of matrices of all couplings in the system defined with
        a `time_dependence` key.
        The ouput will be two lists of matricies representing terms of the hamiltonian which
        are dependent on each time-dependent coupling.
        The lists will be of length M and shape `(*l_time, n, n)`,
        where M is the number of time-dependent couplings, `l_time` is time-dependent stack shape
        (possibly all ones), and `n` is the basis size. Each matrix will have terms equal to the rabi frequency
        (or half the rabi frequency under RWA) in positions that correspond to the associated transition.
        For example, in the case where there is a `time_dependence` function defined for the `(2,3)` transition
        with a rabi frequency of 1, the associated time coupling matrix will be all zeros,
        with a 1 in the `(2,3)` and `(3,2)` positions.

        Typically, this function is called internally and multiplied by the output
        of the :meth:`~.Sensor.get_time_dependence` function.

        Returns
        -------
        list of numpy.ndarray
            The list of M `(*l,n,n)` matrices representing the
            real-valued time-dependent portion of the hamiltonian. For `0 <= i <= M`,
            the ith value along the first axis is the portion of the matrix which
            will be multiplied by the output of the ith `time_dependence` function.

        list of numpy.ndarray
            The list of M `(*l,n,n)` matrices representing the
            imaginary-valued time-dependent portion of the hamiltonian. For `0 <= i <= M`,
            the ith value along the first axis is the portion of the matrix which
            will be multiplied by the output of the ith `time_dependence` function.

        Examples
        --------
        >>> s = rq.Sensor(3)
        >>> step = lambda t: 0 if t<1 else 1
        >>> wave = lambda t: np.sin(2000*np.pi*t)
        >>> f1 = {"states": (0,1), "transition_frequency":10, "rabi_frequency": 1, "time_dependence":wave}
        >>> f2 = {"states": (1,2), "transition_frequency":10, "rabi_frequency": 2, "time_dependence":step}
        >>> s.add_couplings(f1, f2)
        >>> time_hams, time_hams_i = s.get_time_hamiltonian_components()
        >>> for H in time_hams:
        ...     print(H)
        [[0.+0.j 1.+0.j 0.+0.j]
         [1.-0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j]]
        [[0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 2.+0.j]
         [0.+0.j 2.-0.j 0.+0.j]]

        To handle stacking across the steady-state and time hamiltonians, the dimensions are
        matched in a way that broadcasting works in a numpy-friendly way

        >>> s = rq.Sensor(3)
        >>> rabi = np.linspace(-1,1,11)
        >>> step = lambda t: 0 if t<1 else 1
        >>> blue = {"states":(0,1), "rabi_frequency":rabi, "detuning":1}
        >>> red = {"states":(1,2), "rabi_frequency":rabi, "detuning":0, 'time_dependence': step}
        >>> s.add_couplings(red, blue)
        >>> time_hams, time_hams_i = s.get_time_hamiltonian_components()
        >>> print(s.get_hamiltonian().shape)
        (11, 1, 3, 3)
        >>> print(time_hams[0].shape)
        (1, 11, 3, 3)
        >>> print(time_hams_i[0].shape)
        (1, 11, 3, 3)

        """
        #save to re-add later
        self._expand_dims()

        stack_shape = self._stack_shape(time_dependence='time')

        #dictionary of state ordering used to determine indeces in ham
        int_states = {state: i for (i, state) in enumerate(self.states)}

        for states, param, arr, _ in self.variable_parameters(apply_mesh=True):
            self.couplings.edges[states][param] = arr

        hamiltonian_shape = (*stack_shape, self.basis_size, self.basis_size)

        matrix_list = []
        matrix_list_i = []

        #loop over time-dependent couplings
        for states, f in self.couplings_with("time_dependence").items():

            if 'rabi_frequency' not in f:
                continue

            time_hamiltonian = np.zeros(hamiltonian_shape, dtype='complex')
            time_hamiltonian_i = np.zeros(hamiltonian_shape, dtype='complex')

            #convert state label to an index of position in the ham
            states_n = tuple([int_states[s] for s in states])
            idx = (...,*states_n)
            conj_idx = (...,*states_n[::-1])

            cc = f.get("coherent_cc", 1.0)

            time_hamiltonian[idx] = cc * f['rabi_frequency']
            time_hamiltonian_i[idx] = 1j * time_hamiltonian[idx]

            if 'transition_frequency' not in f:
                # add factors of 1/2 and phase for field in rotating frame
                time_hamiltonian[idx] *= np.exp(1j*f["phase"])/2
                time_hamiltonian_i[idx] *= np.exp(1j*f["phase"])/2

            # set hermitian conjugate components
            time_hamiltonian[conj_idx] = np.conj(time_hamiltonian[idx])
            time_hamiltonian_i[conj_idx] = np.conj(time_hamiltonian_i[idx])

            matrix_list.append(time_hamiltonian)
            matrix_list_i.append(time_hamiltonian_i)

        _squeeze_dims(self.couplings)

        return matrix_list, matrix_list_i


    def get_time_dependence(self) -> List[TimeFunc]:
        """
        Function which returns a list of the `time_dependence` functions.

        The list is returned with in the order that matches with the time hamiltonians from
        :meth:`~.Sensor.get_time_couplings` such that the ith element of of the return of this
        functions corresponds with the ith Hamiltonian terms returned by that function.

        Returns
        -------
        list
            List of scalar functions, representing all couplings specified with a `time_dependence`.


        Examples
        --------
        >>> s = rq.Sensor(3)
        >>> step = lambda t: 0 if t<1 else 1
        >>> wave = lambda t: np.sin(2000*np.pi*t)
        >>> f1 = {"states": (0,1), "transition_frequency":10, "rabi_frequency": 1, "time_dependence":wave}
        >>> f2 = {"states": (1,2), "transition_frequency":10, "rabi_frequency": 2, "time_dependence":step}
        >>> s.add_couplings(f1, f2)
        >>> print(s.get_time_dependence())
        [<function <lambda> at ...>, <function <lambda> at ...>]

        """
        time_dependence = []
        for (i,j),f in self.couplings_with("time_dependence").items():
            time_dependence.append(f['time_dependence'])
        return time_dependence


    def get_time_hamiltonian(self, t: float) -> np.ndarray:
        """
        Get the system hamiltonian at a specific time, t.

        This sums the steady-state hamiltonians with the time-dependent parts,
        evaluated at a specific time, t.
        If there is no time dependence in the system, function is equivalent to
        :meth:`get_hamiltonian`.

        Parameters
        ----------
        t: float
            Time to evaluate the time-dependence function at when building the hamiltonians

        Returns
        -------
        numpy.ndarray
            System hamiltonian, evaluated at time t.
        """

        hams_steady = self.get_hamiltonian()
        hamiltonians_time_r, hamiltonians_time_i = self.get_time_hamiltonian_components()
        time_functions = self.get_time_dependence()
        # pre-allocate results
        hamiltonians_time = np.zeros_like(hamiltonians_time_i)
        for i, (func, htr, hti) in enumerate(zip(time_functions, hamiltonians_time_r, hamiltonians_time_i)):
            f0 = func(t)
            hamiltonians_time[i] += f0.real*htr + f0.imag*hti
        # collapse all time function dependence
        hamiltonians_total = hams_steady + np.sum(hamiltonians_time, axis=0)
        return hamiltonians_total


    def get_hamiltonian_diagonal(self, values: dict, no_stack: bool=False) -> np.ndarray:
        """
        Apply addition and subtraction logic corresponding to the direction of the couplings.

        For a given state `n`, the path from ground will be traced to `n`.
        For each edge along this path,
        values will be added where the path direction and coupling direction match,
        and subtracting values where they do not.
        The sum of all such values along the path is the `n` th term in the output array.

        Designed for internal functions which help generate hamiltonians. Most commonly used
        to calculate total detunings for ranges of couplings under the RWA

        Parameters
        ----------
        values : dict
            Key-value pairs where the keys correspond to transitions
            (agnostic to ordering of states) and values corresponding to the values
            to which the logic will be applied.
        no_stack : bool, optional
            Whether to ignore variable parameters in the system and
            use only basic math operations rather than reshape the output. Typically
            only `True` for calculating doppler shifts.

        Returns
        -------
        numpy.ndarray
            The digonal of the hamiltonian of the system of shape `(*l,n)`,
            where `l` is the shape of the hamiltonian stack for the sensor.

        """
        if no_stack:
            diag = np.zeros(self.basis_size)
        else:
            diag = np.zeros((*self._stack_shape(time_dependence="steady"),
                             self.basis_size),
                             dtype=np.complex128)

        subgraphs = self.get_rotating_frames()
        int_states = {state: i for (i, state) in enumerate(self.states)} #ref list of states/idxs

        for paths in subgraphs.values():

            for base_node, path in paths.items():
                # print(base_node, path)

                term = 0

                for j in range(1, len(path)):
                    n, sign = path[j]
                    n_prev, _ = path[j-1]
                    # get the jth couplings along the path
                    # remove frame signs
                    field = (n_prev, n)
                    # get the sign from the rotating frame

                    if sign < 0:
                        # Since it is getting an existing edge from the undirected graph,
                        # we are guaranteed either field or field[::-1] being on the graph
                        field = field[::-1]

                    # sum to the cumulative term along the path from ground
                    term = term + values.get(field,0)*sign
                i = int_states[base_node]
                diag[..., i] = term

        return diag


    def get_rotating_frames(self) -> dict:
        """
        Determines the rotating frames for the disconnected subgraphs.

        Each returned path gives the states traversed,
        and the sign gives the direction of the coupling.
        If the sign is negative, the coupling is going to a lower energy state.
        Choice of frame depends on graph distance to lowest indexed node on subgraph,
        ties broken by lowest indexed path traversed first.

        Returns
        -------
        dict
            Dictionary keyed by disconnected subgraphs,
            values are path dictionaries for each node of the subgraph.
            Each path shows the node indexes traversed,
            where a negative sign denotes a transition to a lower energy state.
        """

        coherent_edges = [
            states for states in self.couplings.edges
            if "rabi_frequency" in self.couplings.edges[states]
        ]

        coherent_graph = self.couplings.edge_subgraph(coherent_edges)

        connected_levels = nx.weakly_connected_components(coherent_graph)
        subgraphs: dict = {coherent_graph.subgraph(ls):{} for ls in connected_levels}

        for g in subgraphs:
            # min sets lowest state in graph as "ground"
            source_node = list(g.nodes)[0]
            paths = nx.shortest_path(nx.to_undirected(g), source=source_node)

            path_and_sign = {}
            for node, path in paths.items():

                path_sign = [1 for _ in path]
                for j in range(1, len(path)):
                    # get the jth couplings along the path and assume the sign as positive
                    field = (path[j-1], path[j])
                    if field not in coherent_graph.edges:
                        # switch the sign if the arrow points in the opposite direction
                        # This corresponds to moving to a lower energy state
                        path_sign[j] = -1
                # print("path", path)
                # print("sign", path_sign)
                path_and_sign[node] = [ps for ps in zip(path, path_sign)]

            subgraphs[g] = path_and_sign

        return subgraphs


    def get_transition_frequencies(self) -> np.ndarray:
        """
        Gets an array of the diagonal elements of the Hamiltonian from the field detunings.

        Wraps the :meth:`~.Sensor.get_hamiltonian_diagonal` function using both
        transition frequencies and detunings. Primarily for internal use.

        Returns
        -------
        numpy.ndarray
            N-D array of the hamiltonian diagonal. For an n-level system with stack shape `*l`,
            will be shape `(*l, n)`

        """
        detuning_dict = self.get_value_dictionary("detuning")
        # enforces detuning convention that positive detuning == blue detuning
        for key, val in detuning_dict.items():
            detuning_dict[key] = -val
        transition_frequency_dict = self.get_value_dictionary("transition_frequency")
        freq_dict = {**detuning_dict, **transition_frequency_dict}

        return self.get_hamiltonian_diagonal(freq_dict)


    def get_value_dictionary(self, key: str) -> dict:
        """
        Get subset of dictionary coupling parameters.

        Return a dictionary of key value pairs where the keys are couplings added
        to the system and the values are the value of the parameter specified by key.
        Produces an output that can be passed directly to :meth:`~.get_hamiltonian_diagonal`.
        Only couplings whose parameter dictionaries contain "key" will be in the
        returned dictionary.

        Parameters
        ----------
        key : str
            String value of the parameter name to build the dictionary.
            For example, `get_value_dictionary("detuning")` will return a dictionary with keys
            corresponding to transitions and values corresponding to detuning
            for each transition which has a detuning.

        Returns
        -------
        dict
            Coupling dictionary with couplings as keys and corresponding
            values set by input key.

        Examples
        --------
        >>> s = rq.Sensor(4)
        >>> f1 = {"states": (0,1), "detuning": 2, "rabi_frequency": 1}
        >>> f2 = {"states": (1,2), "detuning": 3, "rabi_frequency": 2}
        >>> step = lambda t: 1 if t>1 else 0
        >>> f3 = {"states": (2,3), "rabi_frequency": 3, "transition_frequency": 3, "time_dependence":step}
        >>> s.add_couplings(f1, f2, f3)
        >>> print(s.get_value_dictionary("detuning"))
        {(0, 1): 2, (1, 2): 3}

        """
        couplings_with_key = self.couplings_with(key)
        return {states:params[key] for states, params in couplings_with_key.items()}


    def set_gamma_matrix(self, gamma_matrix: np.ndarray):
        """
        Set the decoherence matrix for the system.

        Works by first removing all existing decoherent data from graph edges, then individually
        adding all nonzero terms of a provided gamma matrix to the corresponding graph edges.
        Can be used to set all decoherence attributes to edges simultaneously,
        but :meth:`~.add_decoherence` is preferred.

        Unlike :meth:`~.add_decoherence`, does not support scanning multiple decoherence values,
        rather should be used to set the decoherences of the system to individual static values.

        Parameters
        ----------
        gamma_matrix : numpy.ndarray
            Array of shape `(basis_size, basis_size)`.
            Element `(i,j)` describes the decoherence rate, in Mrad/s,
            from state `i` to state `j`.

        Raises
        ------
        RydiquleError
            If `gamma_matrix` is not a numpy array.
        ValueError
            If `gamma_matrix` is not a square matrix of the appropriate size
        ValueError
            If the shape of `gamma_matrix` is not compatible with `self.basis_size`.

        Examples
        --------
        >>> s = rq.Sensor(2)
        >>> f1 = {"states": (0,1), "detuning":1, "rabi_frequency": 1}
        >>> s.add_couplings(f1)
        >>> gamma = np.array([[.1,0],[.1,0]])
        >>> s.set_gamma_matrix(gamma)
        >>> print(s.decoherence_matrix())
        [[0.1 0. ]
         [0.1 0. ]]

        """
        if not isinstance(gamma_matrix, np.ndarray):
            raise RydiquleError(f'gamma_matrix must be a numpy array, not type {type(gamma_matrix)}')

        if gamma_matrix.shape != (self.basis_size,self.basis_size):
            raise RydiquleError((f'gamma_matrix has shape {gamma_matrix.shape}, '
                                 f'must be {(self.basis_size,self.basis_size)}'))

        for states, gamma in np.ndenumerate(gamma_matrix):
            states = cast(Tuple[int, int], states)  # cast for mypy
            #remove existing decoherence data
            if self.couplings.has_edge(*states):

                remove_keys = []
                for key in self.couplings.edges[states].keys():
                    if key.startswith("gamma_"):
                        remove_keys.append(key)

                for key in remove_keys:
                    del self.couplings.edges[states][key]

            #add new decoherence
            if gamma != 0:
                #exclude gamma==0; its implicitly put there in decoherence_matrix()
                self.add_decoherence(states, gamma)


    def get_doppler_shifts(self) -> np.ndarray:
        """
        Returns the Hamiltonian with only detunings set to the most probable doppler shift values for
        each spatial dimension.

        Determining if a float should be treated as zero is done using :obj:`numpy.isclose`,
        which has default absolute tolerance of `1e-08`.

        Returns
        -------
        numpy.ndarray
            Array of shape (used_spatial_dim,n,n), Hamiltonians
            with only the doppler shifts present along each non-zero spatial dimension
            specified by the fields' "kvec" parameter.

        """
        spatial_dim = 3

        kvecs = self.get_value_dictionary('kvec')
        # collect shifts for each spatial dimension that is non-zero
        s_kvecs = [{k:v[i]*self.vP for k,v in kvecs.items() if ~np.isclose(v[i],0)}
                   for i in range(spatial_dim)]
        if not any(s_kvecs):
            raise RydiquleError(('You must specify at least one non-zero '
                                 'kvector to do doppler averaging.'))
        # get hamiltonian diagonal for each non-zero spatial dimension
        frequencies = np.array([self.get_hamiltonian_diagonal(s_kvec, no_stack=True)
                                for s_kvec in s_kvecs if s_kvec])
        # expand to full hamiltonians
        doppler_hamiltonians = np.eye(self.basis_size) * frequencies[:,np.newaxis,:]

        assert self.spatial_dim() == doppler_hamiltonians.shape[0], \
            'Spatial dimension inconsistency'

        return doppler_hamiltonians


    def couplings_with(self, *keys: str,
                       method: Literal['all','any', 'not any'] = "all"
                       ) -> Dict[States, CouplingDict]:
        """
        Returns a version of self.couplings with only the keys specified.

        Can be specified with a several criteria, including all, none, or any of the keys
        specified.

        Parameters
        ----------
        keys(tuple of str): tuple of strings which should be one the valid
            parameter names for a state. See :meth:`~.add_coupling` for which
            names are valid for a Sensor object.
        method : {'all','any', 'not any'}
            Method to see if a given field matches the keys
            given. Choosing "all" will return couplings
            which have keys matching all of the values provided in the keys
            argument, while coosing "any", will return all couplings with keys
            matching at least one of the values specified by keys. For example,
            `sensor.couplings_with("rabi_frequency")` returns a dictionary of
            all couplings for which a rabi_frequency was specified.
            `sensor.couplings_with("rabi_frequency", "detuning", method="all")`
            returns all couplings for which both rabi_frequency and detuning
            are specified.
            'sensor.couplings_with("rabi_frequency", "detuning", method="any")`
            returns all couplings for which either rabi_frequency or detuning
            are specified.
            Defaults to "all".

        Returns
        -------
        dict
            A copy of the `sensor.couplings` dictionary with only couplings containing
            the specified parameter keys.

        Examples
        --------
        Can be used, for example, to return couplings in the roating wave approximation.

        >>> s = rq.Sensor(3)
        >>> sinusoid = lambda t: 0 if t<1 else sin(100*t)
        >>> f2 = {"states": (0,1), "detuning": 1, "rabi_frequency":2}
        >>> f1 = {"states": (1,2), "transition_frequency":100, "rabi_frequency":1, "time_dependence": sinusoid}
        >>> s.add_couplings(f1, f2)
        >>> gamma = np.array([[.2,0,0],
        ...                  [.1,0,0],
        ...                  [0.05,0,0]])
        >>> s.set_gamma_matrix(gamma)
        >>> print(s.couplings_with("detuning"))
        {(0, 1): {'rabi_frequency': 2, 'detuning': 1, 'phase': 0, 'kvec': (0, 0, 0), 'coherent_cc': 1.0, 'label': '(0,1)'}}
        """
        def notAll(x):
            return not all(x)

        def notAny(x):
            return not any(x)

        methods = {"any":any, "all":all, "not any": notAny, "not all": notAll}

        return {s:p
                for s,p in self.couplings.edges.items()
                if methods[method]([k in p for k in keys])
                }


    def states_with_spec(self, statespec: StateSpec) -> List[State]:
        """
        Return a list of all states in the sensor matching the `state_spec` pattern.

        A state is considered a "match" if, for each element of the state, the corresponding
        element of `statespec` is either exactly the floating point or string value, or a list
        containing that element of state. In this way, groups of states can be specified more
        tersely than a complete list of all states.

        Parameters
        ----------
        statespec:
            The StateSpec against which state labels in sensor are to be matched

        Returns
        -------
        list of State
            All the states in the sensor matching the given specification.

        Examples
        --------
        >>> states = [
        ...    (0,0),
        ...    (1,-1),
        ...    (1,0),
        ...    (1,1)
        ... ]
        >>> s = rq.Sensor(states)
        >>> s.states_with_spec((1,[-1,0,1]))
        [(1, -1), (1, 0), (1, 1)]

        """
        return match_states(statespec, self.states)


    def get_couplings(self) -> Dict[States, CouplingDict]:
        """
        Returns the couplings of the system as a dictionary

        Deprecating in favor of calling the couplings.edges attribute directly.

        Returns
        -------
        dict
            A dictionary of key-value pairs with the keys corresponding to levels of
            transition, and the values being dictionaries of coupling attributes.

        """
        return {s:p for s,p in self.couplings.edges.items()}


    def spatial_dim(self) -> int:
        """
        Returns the number of spatial dimensions doppler averaging will occur over.

        Determining if a float should be treated as zero is done using :obj:`numpy.isclose`,
        which has default absolute tolerance of `1e-08`.

        Returns
        -------
        int
            Number of dimensions, between 0 and 3,
            where 0 means no doppler averaging kvectors have been specified
            or are too small to be calculates.

        Examples
        --------
        No spatial dimesions specified

        >>> s = rq.Sensor(2)
        >>> s.add_coupling((0,1), detuning = 1, rabi_freqency=1)
        >>> print(s.spatial_dim())
        0

        One spatial dimension specified

        >>> s = rq.Sensor(2)
        >>> s.add_coupling((0,1), detuning = 1, rabi_freqency=1, kvec=(0,0,4))
        >>> print(s.spatial_dim())
        1

        Multiple spatial dimensions can exist in a single coupling or
        across multiple couplings

        >>> s = rq.Sensor(2)
        >>> s.add_coupling((0,1), detuning = 1, rabi_freqency=1, kvec=(3,0,3))
        >>> print(s.spatial_dim())
        2

        >>> s = rq.Sensor(3)
        >>> s.add_coupling((0,1), detuning = 1, rabi_freqency=1, kvec=(3,0,3))
        >>> s.add_coupling((1,2), detuning = 2, rabi_freqency=2, kvec=(0,4,0))
        >>> print(s.spatial_dim())
        3
        """
        k_vector_dim = np.zeros(3,dtype=bool)

        for key, field in self.couplings.edges.items():
            if 'kvec' in field:
                k_vector_dim = k_vector_dim | ~np.isclose(field['kvec'],0)

        return np.sum(k_vector_dim)


    def _states_valid(self, states: Sequence) -> States:
        """
        Confirms that the provided states are in a valid format.

        Typically used internally to validate states added. If provided as
        a form other than a tuple, first casts to a tuple for consistent
        indexing.

        Checks that `states` contains 2 elements, can be interpreted as a tuple,
        and that both states lie inside the basis.

        Parameters
        ----------
        states : iterable
            iterable of to validate. Should be a pair of integers that can
            be cast to a tuple.

        Returns
        -------
        tuple
            Length 2 tuple of validated state labels.

        Raises
        ------
        RydiquleError
            If `states` has more than two elements.
        TypeError
            If `states` cannot be converted to a tuple.
        RydiquleError
            If either state in `states` is outside the basis.
        """
        try:
            tpl = tuple(states)
        except TypeError as err:
            raise RydiquleError(
                f'states argument of type {type(states)} cannot be interpreted as a tuple') from err
        if len(tpl) != 2:
            raise RydiquleError(
                f'A field must couple exactly 2 states, but {len(tpl)} are specified in {states}')
        for i in tpl:
            if i not in self.states:
                raise RydiquleError((f'State specification {i} is not a state in the basis'))

        return cast(States, tpl)  # cast for mypy


    def _stack_shape(self, time_dependence: Literal['steady', 'time', 'all']="all"
                     ) -> Tuple[int, ...]:
        """
        Internal function to get the shape of the tuple preceding the two hamiltonian
        axes in :meth:`~.get_hamiltonian()`

        """
        variable_parameters = self.variable_parameters(apply_mesh=True)
        stack_shape_full = np.array(np.broadcast_shapes(*[p.shape for _,_,p,_ in variable_parameters]))
        time_couplings = self.couplings_with("time_dependence").keys()

        steady_idx = []
        time_idx = []

        for states, param, val, zip_label in variable_parameters:
            #find the axis of nontrivial dimension
            axis = [i > 1 for i in val.shape].index(True)

            if states in time_couplings and param=="rabi_frequency":
                time_idx.append(axis)
            else:
                steady_idx.append(axis)

        final_shape = np.ones_like(stack_shape_full)
        if time_dependence in ["steady", "all"]:
            final_shape[steady_idx] = stack_shape_full[steady_idx]
        if time_dependence in ["time", "all"]:
            final_shape[time_idx] = stack_shape_full[time_idx]

        return tuple(final_shape)


    def dm_basis(self) -> np.ndarray:
        """
        Generate basis labels of density matrix components.

        The basis corresponds to the elements in the solution.
        This is not the complex basis of the sensor class, but rather the real basis
        of a solution after calling one of `rydiqule`'s solvers. This means that the
        ground state population has been removed and it has been transformed to the real basis.

        Returns
        -------
        numpy.ndarray
            Array of string labels corresponding to the solving basis.
            Is a 1-D array of length `n**2-1`.

        Examples
        --------
        >>> s = rq.Sensor(3)
        >>> print(s.dm_basis())
        ['01_real' '02_real' '01_imag' '11_real' '12_real' '02_imag' '12_imag'
         '22_real']

        """
        dm_basis = [f'{j:d}{i:d}_imag' if i > j else f'{i:d}{j:d}_real'
                    for i in range(self.basis_size)
                    for j in range(self.basis_size)][1:]  # indexing removes ground state label
        return np.array(dm_basis)


    def _remove_edge_data(self, states: States, kind: str):
        """
        Helper function to remove all data that was added with a :meth:`~.Sensor.add_coupling`
        call or :meth:`~.Sensor.add_decoherence` call.
        Needed to ensure that two nodes do not have coherent couplings pointing both ways
        and to invalidate existing zip parameter couplings.

        Parameters
        ----------
        states : tuple
            Edge from which to remove data.
        kind : str
            What type of data to remove. Valid options are `coherent` coherent couplings
            or the incoherent key to be cleared (must start with `gamma`).

        Raises
        ------
        RydiquleError
            If `kind` is not `'coherent'` and doesn't begin with `'gamma'`
        """

        if states not in self.couplings.edges:
            return

        if kind != 'coherent' and not kind.startswith('gamma'):
            msg = ("If clearing incoherent data,"
                   " must provide key to clear that starts with `gamma`, not {}")
            raise RydiquleError(msg.format(kind))

        #get rid of zips containing this coupling
        zip_labels_to_delete = []
        for param, value in self.couplings.edges[states].items():
            if param in self._zip_labels:
                self._zip_labels.remove(param)
                zip_labels_to_delete.append(param)

        for label in zip_labels_to_delete:
            for s1, s2, parameter in cast(Iterable[Tuple[State,State,Optional[str]]],
                                          self.couplings.edges(data=label)):
                if parameter is not None:
                    del self.couplings.edges[s1, s2][label]

        # delete undesired keys from the edge
        for key in list(self.couplings.edges[states]):  # list() prevents generator persistence
            if key == 'label':
                pass
            elif kind == 'coherent' and not key.startswith('gamma'):
                del self.couplings.edges[states][key]
            elif kind == key:
                del self.couplings.edges[states][key]

        # delete edge outright if it only has a label
        if not sum(k != 'label' for k in self.couplings.edges[states].keys()):
            self.couplings.remove_edge(*states)


    def _coupling_with_label(self, label: Union[str, States]) -> States:
        """
        Helper function to return the pair of states corresponding to a particular label string.
        For internal use.
        """
        #if already a coupling just return as-is
        if isinstance(label, tuple):
            if label in self.couplings.edges():
                return label
            else:
                raise RydiquleError(f"{label} is not a coupling in the sensor")

        label_map = {key:(state1, state2)
                     for state1, state2, key in cast(Iterable[Tuple[State,State,str]],
                         self.couplings.edges(data="label"))}
        if label in label_map.keys():
            return label_map[label]
        else:
            raise RydiquleError(f"No coupling with label {label}")


    def int_states_map(self, invert: bool = False) -> Union[Dict[State, int], Dict[int, State]]:
        """Get a dictionary mapping between state labels and their corresponding integer ordering.
        Can be returned with `key:value` pairs defined either by `label:int` or `int:label`,
        controlled via optional `invert` argument.

        Parameters
        ----------
        invert : bool, optional
            Whether to switch the role of keys and values. Labels are keys if `False`, and
            values if `True`, by default False

        Returns
        -------
        dict
            Dictionary mapping between state labels and integer ordering

        """

        states = {state:n for state, n in zip(self.states, range(self.basis_size))}
        if invert:
            states = {v:k for k,v in states.items()}

        return states

    def variable_parameter_sort(self, par : tuple) -> tuple:
        """Assistance function which determines the sorting order of elements parameters in sensor.

        Called in :meth:`~.Sensor.variable_parameters` to ensure a consistent sort order. Provided as
        the `key` parameter in python's `sorted()` function before parameters are returned.

        Sorts first by `zip_label`, then by `states`, then by `parameter`. Ensures all parameters
        zipped with one another are grouped together in a list.  Zipped parameters will always come
        first. From there, parameters are sorted alphabetically by zip_label (including case), then
        by state pair (as determined by ordering in the sensor, NOT alphabetically), then
        alphabetically by parameter.

        Parameters
        ----------
        par : tuple
            4-element list of information on each parameter. Consists of `(states, parameter, value, zip_label)`

        Returns
        -------
        tuple
            3-element tuple that defines a particular parameter's position in the final sorting order.

        """
        (states, parameter, _, zip_label) = par #unpack parameter
        states_int = tuple(self.int_states_map()[i] for i in states)
        zip_label_str = "none" if zip_label is None else ("_"+zip_label) #empty string if no zip label
        return (zip_label_str, states_int, parameter)


    def _expand_dims(self):
        """Converts the 1-D arrays in the sensor into shapes that allows for rydiqule stacking.
        """
        for states, param, arr, _ in self.variable_parameters(apply_mesh=True):
            self.couplings.edges[states][param] = arr


    def __str__(self):
        """Overload of __str__ allowing a clean way to view all info in a :class:`~.Sensor`.

        Returns
        -------
        str
            Tidy string representation of a sensor, showing all states and couplings.

        """

        n_coh = np.sum([1 for (s1,s2) in self.couplings.edges if "rabi_frequency" in self.couplings.edges[(s1,s2)]])
        out_str = f"{self.__class__} object with {len(self.couplings)} states and {n_coh} coherent couplings.\n"

        out_str += f"States: {self.states}\n"
        out_str += "Coherent Couplings: "

        #add coherent couplings
        coh_couplings = [(s1, s2, e) for (s1, s2, e) in self.couplings.edges.data() if "rabi_frequency" in e]
        if len(coh_couplings) == 0:
            coh_couplings_str = "\n    None"
        else:
            coh_couplings.sort(key=self.__sort_couplings)
            coh_couplings_str = "".join(["\n" + "    " + _format_coupling_str(c) for c in coh_couplings])
        out_str += coh_couplings_str

        #add decoherent couplings
        decoh_couplings_str = "\nDecoherent Couplings:"
        decoh_couplings = [(s1, s2, _extract_gamma_keys(e))
                           for (s1, s2, e) in self.couplings.edges.data()
                           if "rabi_frequency" not in e and len(_extract_gamma_keys(e)) > 0]
        if len(decoh_couplings) == 0:
            decoh_couplings_str += "\n    None"
        else:
            decoh_couplings.sort(key=self.__sort_couplings)
            decoh_couplings_str += "".join(["\n" + "    " + _format_coupling_str(c) for c in decoh_couplings if len(c[2]) > 0])
        out_str += decoh_couplings_str

        #add energy shift
        e_shifts_str = "\nEnergy Shifts:"
        self_loops = [(state, e) for (state, _, e) in nx.selfloop_edges(self.couplings, data="e_shift") if e is not None]
        if len(self_loops) == 0:
            e_shifts_str += "\n    None"
        else:
            self_loops.sort(key=self.__sort_couplings)
            e_shifts_str += "".join(["\n" + "    " + str(state) + ": " + str(e) for (state, e) in self_loops])
        out_str += e_shifts_str

        #add zips if present
        if len(self._zip_labels) >  0:
            out_str += "\nZip Labels:\n    " + str(self._zip_labels)

        return out_str  #+ decoh_couplings_str


    def __sort_couplings(self, coupling):
        """Helper function for __str__ to sort couplings by states
        """
        if len(coupling) == 2:
            return self.states.index(coupling[0])
        return (self.states.index(coupling[0]), self.states.index(coupling[1]))


def _format_coupling_str(coupling: tuple):
    """Helper function to format the data in a coupling into a string for __str__.
    """
    states = f"({coupling[0]},{coupling[1]})"
    params = "".join([f"{k}: {_format_param_str(v)}, " for k,v in coupling[2].items()])
    return states + ": " + "{" + params[:-2] +"}"

def _extract_gamma_keys(d: Dict):
    """Helper function to get all entries in a dict whose keys start with "gamma"
    """
    return {key:value for key, value in d.items() if key.startswith("gamma")}

def _format_param_str(param):
    """Helper function to format the data in a parameter into a string for __str__.
    Mostly just turns arrays into strings indicating number of elements.
    """
    return str(param) if not (hasattr(param, "size") and param.size>1) else f"<parameter with {param.size} values>"
