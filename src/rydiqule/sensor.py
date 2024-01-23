"""
Sensor objects that control solvers.
"""

import numpy as np
import networkx as nx
import re
import warnings

import itertools

from .sensor_utils import _combine_parameter_labels
from .sensor_utils import ScannableParameter, CouplingDict, State, States, TimeFunc

from typing import List, Tuple, Dict, Literal, Callable, Optional, Union, Sequence, cast

BASE_SCANNABLE_KEYS = ["detuning", 
                       "rabi_frequency", 
                       "phase", 
                       "transition_frequency",
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
                   "dipole_moment"]
"""Reference list of all keys that can be specified with values in a coherenct coupling.
Subclasses which inherit from :class:`~.Sensor` should override the `valid_parameters` attribute,
NOT this list. The `valid_parameters` attribute is initialized as a copy of `BASE_EDGE_KEYS`."""

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

    probe_tuple: Optional[States] = None
    """Coupling edge that corresponds to the probing field.
    Defaults to `(0,1)` in :class:`Cell`."""

    probe_freq: Optional[float] = None
    """Probing transition frequency, in rad/s."""

    cell_length: Optional[float] = None
    """Optical path length of the medium, in meters."""

    beam_area: Optional[float] = None
    """Cross-sectional area of the probing beam, in square meters."""

    def __init__(self, basis: Union[int, List[Union[int, str]]],
                 *couplings: CouplingDict) -> None:
        """
        Initializes the Sensor with the specified basis .

        Can be specified as either an integer number of states (which will automatically 
        label the states `[0,...,basis_size]`) or list of state labels. 

        Parameters
        ---------
        basis: int or list of int, str
            The specification of the basis size and labelling for a new `Sensor`. Can be specified 
            by either a integer or a list. If specified as an integer `n`, the created `Sensor` 
            will have `n` states labelled as `0,...n`. In the case of a list, a number of states 
            equal to the length of the list will be created in the sensor, indexed by the integer 
            or string values of the nodes.
        *couplings : tuple(dict) 
            Couplings dictionaries to pass to :meth:`~.add_couplings` on sensor construction.
        
        Raises
        ------
        TypeError
            If `basis` is not an integer or iterable.
        TypeError
            If any of the state label specifications of basis are the wrong type.

        Examples
        --------
        Providing an integer will define a sensor with the given basis size, labelled with
        ascending integers.

        >>> s = rq.Sensor(3)
        >>> print(s.states)
        [0, 1, 2]

        States can also be defined with a list of integers:

        >>> s = rq.Sensor([0, 1, 2])
        >>> print(s.states) 
        [0, 1, 2]

        States can also be strings

        >>> s = rq.Sensor(['g', 'e1', 'e2'])
        >>> print(s.states)
        ['g', 'e1', 'e2']

        Using `None` in a list will default to using the integer correspoinding to the state:

        >>> s = rq.Sensor(['g', None, None])
        >>> print(s.states)
        ['g', 1, 2]
        
        """
        #if its an int, expand to variables
        if isinstance(basis, int):
            basis = list(range(basis))

        try:        
            valid_node = [isinstance(n, (str,int)) for n in basis]
        except TypeError:
            raise TypeError("'basis' must be an integer or iterable")
        
        #ensure node types are valid
        if not all(valid_node):
            raise ValueError("Nodes in \'basis\' must be integers or strings")
        
        #ensure unique labels
        if len(basis) != len(set(basis)):
            raise ValueError("All state labels must be unique")
        
        self.valid_parameters = BASE_EDGE_KEYS.copy()
        self.couplings = nx.DiGraph()
        self.couplings.add_nodes_from(basis)

        if len(couplings) > 0:
            self.add_couplings(*couplings)

        self._zipped_parameters: Dict = {}


    def set_experiment_values(self, probe_tuple: Tuple[int,int],
                              probe_freq: float,
                              kappa: float,
                              eta: Optional[float] = None,
                              cell_length: Optional[float] = None,
                              beam_area: Optional[float] = None,
                              ):
        """Sets attributes needed for observable calculations.
        
        Parameters
        ----------
        probe_tuple: tuple of int
            Coupling that corresponds to the probing field.
        probe_freq: float
            Frequency of the probing transition, in Mrad/s.
        kappa: float
            Numerical prefactor that defines susceptibility, in (rad/s)/m.
            See :func:`~.get_susceptibility` and :class:`Cell.kappa` for details.
        eta: float
            Noise-density prefactor, in root(Hz).
            See :class:`Cell.eta` for details.
        cell_length: float, optional
            The optical path length through the medium, in meters.
        beam_area: float, optional
            The cross-sectional area of the beam, in m^2.        
        """

        self.probe_tuple = self._states_valid(probe_tuple)
        self.probe_freq = probe_freq
        self.cell_length = cell_length
        self.beam_area = beam_area
        self.eta = eta
        self.kappa = kappa
    

    @property
    def basis_size(self):
        """Property to return the number of nodes on the Sensor graph. 

        Returns
        -------
        int
            The number of nodes on the graph, corresponding to the basis size for the system.
        """
        return len(self.couplings)
    

    @property
    def states(self):
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
    

    def add_energy_shift(self, state: State, shift: ScannableParameter):
        """Add an energy shift to a state.

        First perfoms validation that the provided `state` is actually a node in the graph, then
        adds the shift specified by `shift` to a self-loop edge keyed with `"e_shift"`. This value
        will be added to the corresponding diagonal term when the hamiltonian is generated. If
        the provided node

        Parameters
        ----------
        state : str or int
            The label corresponding to the atomic state to which the shift will be added.
        shift : float or list-like of float
            The magnitude of the energy shift, in Mrad/s

        Raises
        ------
        KeyError
            If the supplied `state` is not in the system.
        """
        if not self.couplings.has_node(state):
            raise KeyError(f"state {state} is not a node on the graph")
        
        self._remove_edge_data((state, state), kind="coherent")
        self.couplings.add_edge(state, state, e_shift=shift)
            
    
    def add_energy_shifts(self, shifts:dict):
        """Add multiple energy shifts to different nodes.

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
        except AttributeError:
            raise ValueError("Shifts parameters must be a dictionary-like object")
        
        for state, shift in shifts_items:
            self.add_energy_shift(state, shift)
        

    def add_coupling(
            self, states: States, rabi_frequency: Optional[ScannableParameter] = None,
            detuning: Optional[ScannableParameter] = None,
            transition_frequency: Optional[float] = None,
            phase: ScannableParameter = 0,
            kvec: Tuple[float,float,float] = (0,0,0),
            time_dependence: Optional[Callable[[float],float]] = None,
            label: Optional[str] = None,
            **extra_kwargs) -> None:
        """
        Adds a single coupling of states to the system.

        One or more of these paramters can be a list or array-like of values to represent
        a laser that can take on a set of discrete values during a field scan.
        Designed to be a user-facing wrapper for :meth:`~._add_coupling` with arguments
        for states and coupling parameters.

        Parameters
        ----------
        states : tuple of ints or strings of length 2
            The pair of states of the sensor which the state couples. Must be a tuple
            of length 2, where each element is a string or integer corresponding to a state in the
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
        transition_frequency : float or list-like of floats, optional 
            The transition frequency between a particular pair of states. Must be a positive number.
            List-like values will invoke Rydiqule's stacking convention when relevant quantities are calculated. 
            Only used directly in calculations if `detuning` is `None`, ignored otherwise. 
            Note that on its own, it only defines the spacing between two energy levels and not the
            field itsself. To define a field, the `time_dependence` argument must be specified, or else 
            the off-diagonal terms to drive transitions will not be generated in the Hamiltonian matrix.
        phase : float, optional
            The relative phase of the field in radians. Defaults to zero.
        kvec : iterable, optional 
            A three-element iterable that defines the atomic doppler shift on a particular coupling
            field. It should have magntiude equal to the doppler shift (in the units of Mrad/s) of a
            n atom moving at the Maxwell-Boltzmann distribution most probable speed, `vP=np.sqrt(2*kB*T/m)`. 
            I.E. `np.linalg.norm(kvec)=2*np.pi/lambda*vP`. If equal to `(0,0,0)`, solvers will ignore 
            doppler shifts on this field.  Defaults to `(0,0,0)`.
        time_dependence : scalar function, optional
            A scalar function specifying a time-dependent field. The time dependence function is defined 
            as a python funtion that returns a unitless value as a function of time (in microseconds) 
            that is multiplied by the `rabi_frequency` parameter to get a  field strength scaled to units 
            of Mrad/s.
        label : str or None, optional
            Name of the coupling. This does not change any calculations, but can be used 
            to help track individual couplings, and will be reflected in the output of 
            :meth:`~.Sensor.axis_labels`, and to specify zipping for :meth:`~.Sensor.zip_couplings`. 
            If `None`, the label is generated as the value of `states` cast to a string with 
            whitespace removed. Defaults to `None`.

        Raises
        ------
        ValueError
            If `states` cannot be interpreted as a tuple.
        ValueError
            If `states` does not have a length of 2.
        ValueError
            If the state numbers specified by `states` are beyond the basis size.
            For example, calling this function with `states=(3,4)`
            will raise this error if the basis size is equal to 3.
        ValueError
            If both `rabi_frequency` and `dipole_moment` are specified or if
            neither are specified.
        ValueError
            If both detuning and transition_frequency are specified or if
            neither are specified.

        Examples
        --------
        >>> s = rq.Sensor(2)
        >>> s.add_coupling(states=(0,1), detuning=1, rabi_frequency=2)

        >>> s = rq.Sensor(['g', 'e'])
        >>> s.add_coupling(('g', 'e'), detuning=1, rabi_frequency=1)

        >>> s = rq.Sensor(2)
        >>> s.add_coupling(states=(0,1), detuning=np.linspace(-10, 10, 101), rabi_frequency=2, label="laser")

        >>> s = rq.Sensor(2)
        >>> step = lambda t: 1 if t>=1 else 0
        >>> s.add_coupling(states=(0,1), transition_frequency=1000, rabi_frequency=2, time_dependence=step)

        >>> s = rq.Sensor(2)
        >>> kp = 250*np.array([1,0,0])
        >>> s.add_coupling(states=(0,1), detuning=1, rabi_frequency=2, kvec=kp)

        """
        #ensure states are unique
        if states[0] == states[1]:
            raise ValueError(f'{states}: Coherent coupling must couple different states.')

        suppress_rwa = extra_kwargs.pop("suppress_rwa_warn", False)

        field_params = {k:v for k,v in locals().items()
                        if k in self.valid_parameters
                        and v is not None}
        
        if not (detuning is not None) ^ (transition_frequency is not None):
            raise ValueError(f"{states}: Please specify \'detuning\' for a field under the RWA"
                            " or \'transition_frequency\' for a coupling without the approximation,"
                            " but not both.")
        
        if transition_frequency is not None:
            if transition_frequency < 0:
                raise ValueError(f"{states}: \'transition_frequency\' must be positive.")
            elif transition_frequency > 5000 and not suppress_rwa:
                msg = (f"{states}: Not using the rotating wave approximation"
                    " for large transition frequencies can result in "
                    "prohibitively long computation times. Specify detuning to use "
                    "the rotating wave approximation or pass \"suppress_rwa_warn=True\" "
                    "to add_coupling() to suppress this warning.")

                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    warnings.warn(msg, stacklevel=2)

        self._add_coupling(**field_params, **extra_kwargs)


    def add_couplings(self, *couplings: CouplingDict,
                        **extra_kwargs) -> None:
        """
        Add any number of couplings between pairs of states.
        
        Acts as an alternative to calling :meth:`~.Sensor.add_coupling`
        individually for each pair of states.
        Can be used interchangably up to preference,
        and all of keyword :meth:`~.Sensor.add_coupling` are supported dictionary
        keys for dictionaries passed to this function.

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

        Raises
        ------
        ValueError
            If the `states` parameter is missing.

        Examples
        --------
        >>> s = rq.Sensor(3)
        >>> blue = {"states":(0,1), "rabi_frequency":1, "detuning":2}
        >>> red = {"states":(1,2), "rabi_frequency":3, "detuning":4}
        >>> s.add_couplings(blue, red)
        >>> print(s.couplings.edges(data=True))
        [(0, 1, {'rabi_frequency': 1, 'detuning': 2, 'phase': 0, 'kvec': (0, 0, 0)}),
        (1, 2, {'rabi_frequency': 3, 'detuning': 4, 'phase': 0, 'kvec': (0, 0, 0)})]

        """
        for c in couplings:
            c_copy = c.copy()
            try:
                states = self._states_valid(c_copy.pop('states'))
            except KeyError:
                raise ValueError("\'states\' parameter must be specified for any field")

            self.add_coupling(states=states, **c_copy, **extra_kwargs)


    def _add_coupling(self, states: States, **field_params) -> None:
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
            field_params["label"] = str(states).replace(" ","")

        # remove all coherent data in both directions on the graph
        # this prevents adding coherent couplings in both directions between 2 nodes
        self._remove_edge_data(states, 'coherent')
        self._remove_edge_data(states[::-1], 'coherent')

        self.couplings.add_edge(*states, **field_params)


    def zip_parameters(self, *parameters: str,  zip_label: Optional[str]=None):
        """
        Define 2 scannable parameters as "zipped" so they are scanned in parallel.

        Zipped parameters will share an axis when quantities relevant to the equations of
        motion, such as the `gamma_matrix` and `hamiltonian` are generated. Note that calling
        this function does not affect internal quanties directly, but adds their labels together
        in the internal `self._zipped_parameters` dict, and they are zipped at calculation time
        for `hamiltonian` and `decoherence_matrix`.

        Parameters
        ----------
        parameters : str
            Parameter labels to scan together. Parameter labels are strings of the form 
            `"<coupling_label>_<parameter_name>"`, such as `"(0,1)_detuning"`.
            Must be at least 2 labels to zip. Note that couplings are specified in the 
            :meth:`~.Sensor.add_coupling` function. If unspecified in this function, the 
            pair of states in the coupling cast to a string will be used.

        
        zip_label : optional, str
            String label shorthand for the zipped parameters. The label for the axis of these
            parameters in :meth:`~.Sensor.axis_labels()`. Does not affect functionality of the
            Sensor. If unspecified, the label used will be `"zip_" + <number>`.

        Raises
        ------
        ValueError
            If fewer than 2 labels are provided.
        ValueError
            If any of the 2 labels are the same.
        ValueError
            If any elements of `labels` are not labels of couplings in the sensor.
        ValueError
            If any of the parameters specified by labels are already zipped.
        ValueError
            If any of the parameters specified are not list-like.
        ValueError
            If all list-like parameters are not the same length.

        Notes
        -----
        .. note::
            This function should be called last after all Sensor couplings and dephasings
            have been added. Changing a coupling that has already been zipped removes it from
            the `self.zipped_parameters` list. 
            
        .. note::
            Modifying the `Sensor.zipped_parameters` attribute directly can break some functionality
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
        >>> s.zip_parameters("probe_detuning", "(1,2)_detuning", zip_label="demo_zip")
        >>> print(s._zipped_parameters) #NOT modifying directly
        {'demo_zip': ['(1,2)_detuning', 'probe_detuning']}

        Make sure to add the appropriate additional string markings when the states are strings. 

        >>> s = rq.Sensor(['g', 'e1', 'e2'])
        >>> det = np.linspace(-1,1,11)
        >>> s.add_coupling(states=('g','e1'), detuning=det, rabi_frequency=1, label="probe")
        >>> s.add_coupling(states=('e1','e2'), detuning=det, rabi_frequency=1)
        >>> s.zip_parameters("probe_detuning", "('e1','e2')_detuning", zip_label="demo_zip")
        >>> print(s._zipped_parameters) #NOT modifying directly
        {'demo_zip': ["('e1','e2')_detuning", 'probe_detuning']}

        """
        #give a dummy label if not provided
        if zip_label is None:
            zip_label = "zip_" + str(len(self._zipped_parameters))

        #ensure zip label does not already exist
        if zip_label in self._zipped_parameters.keys():
            raise ValueError(f"Parameters already zipped with label {zip_label}."
                             "Please use a unique label.")

        # check for at least 2 labels
        if len(parameters) < 2:
            raise ValueError(("Please provide at least 2 parameter labels "
                              f"to zip (only provided {len(parameters)})"))
            
        #check that all labels are unique
        if len(parameters) != len(set(parameters)): #set will be shorter if there are duplicates
            raise ValueError("parameters cannot be zipped to themselves")
        
        #check if any labels are already zipped
        current_zips = [self._zipped_parameters.values()]
        for l1, l2 in itertools.product(current_zips, parameters):
            if l1 == l2:
                raise ValueError(f"Parameter {l1} already zipped!")

        #ensure provided labels are valid for zipping
        previous_len = 0
        for l in parameters:
            
            # divide coupling from parameter
            # check if parameter exists done later
            split = l.split('_')
            if len(split) < 2:
                raise ValueError(f"Invalid parameter label {l}")
            try:
                # incoherent parameter
                gi = split.index('gamma')
                coupling = '_'.join(split[:gi])
                param = '_'.join(split[gi:])
            except ValueError:
                # coherent parameter
                if split[-1] in BASE_SCANNABLE_KEYS:
                    coupling = '_'.join(split[:-1])
                    param = split[-1]
                else:
                    coupling = '_'.join(split[:-2])
                    param = '_'.join(split[-2:])
            
            #make sure the label exists
            try:
                states = self._coupling_with_label(coupling)
            except ValueError:
                raise ValueError(f"{coupling} is not a label of any coupling in this sensor")
            
            #make sure parameter exists and is an array
            try:
                _ = self.couplings.edges[states][param]
            except KeyError:
                raise ValueError(f"Coupling {coupling} has no parameter {param}")
            if not hasattr(param, "__len__"):
                raise ValueError(f"Parameter {l} is not list-like and cannot be zipped")
            
            #make sure array-defined parameters are the same length
            current_len = len(self.couplings.edges[states][param])
            if previous_len > 0 and current_len != previous_len:
                raise ValueError(
                    f"Got length {current_len} for parameter \"{l}\", "
                    f"but \"{parameters[0]}\" is length {previous_len}")
            
            previous_len = current_len
                        
        parameters = list(parameters)
        parameters.sort()
        self._zipped_parameters[zip_label] = parameters


    def unzip_parameters(self, zip_label, verbose=True):
        
        """
        Remove a set of zipped parameters from the internal zipped_parameters list.

        If an element of the internal `_zipped_parameters` array matches ALL labels provided,
        removes it from the internal `zipped_parameters` method. If no such element is
        in `_zipped_parameters`, does nothing.

        Parameters
        ----------
        zip_label : str 
            The string label corresponding the key to be deleted in the `_zipped_parameters`
            attribute. 
        
        Notes
        -----
        .. note::
            This function should always be used rather than modifying the `_zipped_parameters`
            attribute directly.
            
        Examples
        --------
        >>> s = rq.Sensor(3)
        >>> det = np.linspace(-1,1,11)
        >>> s.add_coupling(states=(0,1), detuning=det, rabi_frequency=1, label="probe")
        >>> s.add_coupling(states=(1,2), detuning=det, rabi_frequency=1)
        >>> s.zip_parameters("probe_detuning", "(1,2)_detuning", zip_label="demo1")
        >>> print(s._zipped_parameters) #NOT modifying directly
        >>> s.unzip_parameters("demo1")
        >>> print(s._zipped_parameters) #NOT modifying directly
        {'demo1': ['(1,2)_detuning', 'probe_detuning']}
        {}
        
        If the labels provided are not a match, a message is printed and nothing is altered. 
        
        >>> s = rq.Sensor(3)
        >>> det = np.linspace(-1,1,11)
        >>> s.add_coupling(states=(0,1), detuning=det, rabi_frequency=1, label="probe")
        >>> s.add_coupling(states=(1,2), detuning=det, rabi_frequency=1)
        >>> s.zip_parameters("probe_detuning", "(1,2)_detuning")
        >>> print(s._zipped_parameters) #NOT modifying directly
        >>> s.unzip_parameters('blip_0')
        >>> print(s._zipped_parameters) #NOT modifying directly
        {'zip_0': ['(1,2)_detuning', 'probe_detuning']}
        No label matching blip_0, no action taken
        {'zip_0': ['(1,2)_detuning', 'probe_detuning']}
        
        """
        try:
            del self._zipped_parameters[zip_label]
        except KeyError:
            if verbose:
                print(f"No label matching {zip_label}, no action taken")


    def add_decoherence(self, states: States, gamma: ScannableParameter,
                        label: Optional[str] = None):
        """
        Add decoherent coupling to the graph between two states.

        If `gamma` is list-like, the sensor will scan over the values,
        solving the system for each different gamma, identically to the
        scannable parameters in coherent couplings.

        Parameters
        ----------
        states : tuple of ints
            Length-2 tuple of integers corresponding to the two states. The first
            value is the number of state out of which population decays, and the 
            second is the number of the state into which population decays.
        gamma : float or sequence
            The decay rate, in Mrad/s.
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
        s = rq.Sensor(3)
        >>> s.add_coupling(states=(0,1), detuning=1, rabi_frequency=1)
        >>> s.add_coupling(states=(1,2), detuning=1, rabi_frequency=1)
        >>> s.add_decoherence((2,0), 0.1, label="misc")
        >>> print(s.decoherence_matrix())
        [[0.  0.  0. ]
        [0.  0.  0. ]
        [0.1 0.  0. ]]
        
        Decoherence values can also be scanned. Here decoherece from states 2->0 is scanned
        between 0 and 0.5 for 11 values. We can also see how the Hamiltonian shape accounts
        for this to allow for clean broadcasting, indicating that the hamiltonian is identical
        accross all decoherence values.
        
        >>> s = rq.Sensor(3)
        >>> gamma = np.linspace(0,0.5,11)
        >>> s.add_coupling(states=(0,1), detuning=1, rabi_frequency=1)
        >>> s.add_coupling(states=(1,2), detuning=1, rabi_frequency=1)
        >>> s.add_decoherence((2,0), gamma)
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
        # invalidate existing decoherence data if already exists
        self._remove_edge_data(states, kind=label_full)
        self.couplings.add_edge(*states, **{label_full:gamma})

        # if edge doesn't have a label (ie decoherence only), add a default label
        if self.couplings.edges[states].get("label") is None:
            self.couplings.edges[states]["label"] = str(states).replace(" ", "")


    def add_transit_broadening(self, gamma_transit: ScannableParameter,
                               repop: Union[None, Dict[State, float]]= None,
                               label: str = "transit") -> None:
        """
        Adds transit broadening by adding a decoherence from each node to ground.
        
        For each graph node n, adds a decoherent transition from n the specified state
        (0 by default) using the :meth:`~.Sensor.add_decoherence` method with the `"transit"` label.
        See :meth:`~.Sensor.add_decoherence` for more details on labeling.

        If an array of transit values are provided, they will be automatically zipped together
        into a single scanning element.

        Parameters
        ----------
        gamma_transit: float or sequence
            The transit broadening rate in Mrad/s.
        repop: dict, optional
            Dictionary of states for transit to repopulate in to.
            The keys represent the state labels. The values represent
            the fractional amount that goes to that state.
            If the sum of values does not equal 1, population will not be conserved.
            Default is to repopulate everything into the ground state (either state 0
            or the first state in the basis passed to the :meth:`~.Sensor.__init__` method).

        Warns
        -----
        If the values of the `repop` parameter do not sum to 1, thus meaning
        population will not be conserved.
        
        Examples
        --------
        >>> s = rq.Sensor(3)
        >>> s.add_transit_broadening(0.1)
        >>> print(s.couplings.edges(data=True))
        >>> print(s.decoherence_matrix())
        [(0, 0, {'gamma_transit': 0.1}), (1, 0, {'gamma_transit': 0.1}), (2, 0, {'gamma_transit': 0.1})]
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
        
        if repop is None:
            ground_state = self.states[0]
            repop = {ground_state: 1.0}

        if isinstance(gamma_transit, list):
            # needed for multiplying branching ratios
            gamma_transit = np.array(gamma_transit)

        if not np.isclose(sum(repop.values()), 1.0):
            warnings.warn(('Repopulation branching ratios do not sum to 1!'
                           ' Population will not be conserved.'))

        for t, br in repop.items():
            for i in self.states:
                self.add_decoherence((i, t), gamma=gamma_transit*br, label="transit")

        if isinstance(gamma_transit, (list,np.ndarray)):
            # need to zip together all the transit rates
            axes = self.axis_labels()
            transit_axes = [s for s in axes if s.endswith('gamma_transit')]
            self.zip_parameters(*transit_axes, zip_label=label)


    def add_self_broadening(self, node: int, gamma: ScannableParameter,
                            label: str = "self"):
        """
        Specify self-broadening (such as collisional broadening) of a level.

        Equivalent to calling :meth:`~.Sensor.add_decoherence` and specifying both
        states to be the same, with the "self" label. For more complicated systems,
        it may be useful to further specify the source of self-broadening as, for
        example, "collisional" for easier bookkeeping and to ensure no values
        are overwritten.

        Parameters
        ----------
        node: int
            The integer number of the state node to which the broadening
            will be added. The integer corresponds to the state's position in
            the graph.
        gamma: float or sequence
            The broadening width to be added in Mrad/s.
        label: str, optional
            Optional label for the state. If `None`, decay will be stored on
            the graph edge as `"gamma"`. Otherwise, will cast as a string
            and decay will be stored on the graph edge as `"gamma_"+label`

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
        >>> print(s.decoherence_matrix())
        [(1, 1, {'gamma_self': 0.1, 'label': '(1,1)'})]
        [[0.  0.  0. ]
        [0.  0.1 0. ]
        [0.  0.  0. ]]

        """
        self.add_decoherence((node, node), gamma, label=label)


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
        >>> print(s.decoherence_matrix())
        [(1, 0, {'gamma_foo': 0.2, 'label': '(1,0)', 'gamma_bar': 0.1}), (2, 0, {'gamma': 0.05, 'label': '(2,0)'}), (2, 1, {'gamma': 0.05, 'label': '(2,1)'})]
        [[0.   0.   0.  ]
        [0.3  0.   0.  ]
        [0.05 0.05 0.  ]]
        
        Decoherences can be stacked just like any parameters of the Hamiltonian:
        
        >>> s = rq.Sensor(3)
        >>> gamma = np.linspace(0,0.5, 11)
        >>> s.add_decoherence((1,0), gamma)
        >>> print(s.decoherence_matrix().shape)
        (11,3,3)

        Defining decoherences between states labelled with string values works just like coherent couplings:
       
        >>> s = rq.Sensor(['g', 'e1', 'e2'])
        >>> s.add_decoherence(('e1', 'g'), 0.1)
        >>> s.add_decoherence(('e2', 'g'),0.1)
        >>> s.decoherence_matrix()
        array([[0. , 0. , 0. ],
               [0.1, 0. , 0. ],
               [0.1, 0. , 0. ]])

        """
        base_couplings = self.couplings.copy()

        int_states = {state: i for (i, state) in enumerate(self.states)}

        stack_shape = self._stack_shape()
        for states, param, arr in self.variable_parameters(apply_mesh=True):
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

        self.couplings = base_couplings
        return gamma_matrix


    def axis_labels(self, collapse: bool=True, full_labels:bool=False) -> List[str]:
        """
        Get a list of axis labels for stacked hamiltonians. 
        
        The axes of a hamiltonian
        stack are defined as the axes preceding the usual hamiltonian, which are always
        the last 2. These axes only exist if one of the parametes used to define
        a Hamiltonian are lists.

        Be default, labels which have been zipped using :meth:`~.Sensor.zip_parameters`
        will be combined into a single label, as this is how :meth:`~.Sensor.get_hamiltonian`
        treats these axes.

        The ordering of axis labels is 

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
        >>> print(s.get_hamiltonian().shape())
        >>> print(s.axis_labels())
        (3,3)
        []

        Adding list-like parameters expands the hamiltonian

        >>> s = rq.Sensor(3)
        >>> det = np.linspace(-10, 10, 11)
        >>> blue = {"states":(0,1), "rabi_frequency":1, "detuning":det, "label":"blue"}
        >>> red = {"states":(1,2), "rabi_frequency":3, "detuning":det}
        >>> s.add_couplings(blue, red)
        >>> print(s.get_hamiltonian().shape)
        >>> print(s.axis_labels())
        (11, 11, 3, 3)
        ['blue_detuning', '(1, 2)_detuning']

        The ordering of labels may change if string state names are used. The ordering
        is determined by the output of the :meth:`~.Sensor.variable_parameters` method.
        See method documentation for more detail.

        >>> s = rq.Sensor(['g', 'e1', 'e2'])
        >>> det = np.linspace(-10, 10, 11)
        >>> blue = {"states":('g','e1'), "rabi_frequency":1, "detuning":det, "label":"blue"}
        >>> red = {"states":('e1','e2'), "rabi_frequency":3, "detuning":det}
        >>> s.add_couplings(blue, red)
        >>> print(s.get_hamiltonian().shape)
        >>> print(s.axis_labels())
        (11, 11, 3, 3)
        ["('e1','e2')_detuning", 'blue_detuning']

        Zipping parameters combines their corresponding labels, since their Hamiltonians now 
        lie on a single axis of the stack. Here the axis of length 7 (axis 0) corresponds to the
        rabi frequencies and the axis of shape 11 (axis 1) corresponds to the zipped detunings
        
        >>> s = rq.Sensor(3)
        >>> s.add_coupling(states=(0,1), detuning=np.arange(11), rabi_frequency=np.linspace(-3, 3, 7))
        >>> s.add_coupling(states=(1,2), detuning=0.5*np.arange(11), rabi_frequency=1)
        >>> s.zip_parameters("(0,1)_detuning", "(1,2)_detuning", zip_label="detunings")
        >>> print(s.get_hamiltonian().shape)
        >>> print(s.axis_labels())
        >>> print(s.axsi_labels(full_labels=True))
        (7, 11, 3, 3)
        ['(0,1)_rabi_frequency', 'detunings']
        ['(0,1)_rabi_frequency', '(0,1)_detuning|(1,2)_detuning']

        """
        key_labels = []
        item_labels = []

        # build the base list of axis labels based just on couplings
        for states, param, _ in self.variable_parameters():
            if 'label' in self.couplings.edges[states].keys():
                label = str(self.couplings.edges[states]['label']) + '_' + str(param)
            else:
                label = str(states) + "_" + str(param)
            key_labels.append(label)
            item_labels.append(label)

        # combine labels for parameters that have been zipped and move to the end of stack
        if collapse:
            for key, params in self._zipped_parameters.items():
                new_label = _combine_parameter_labels(*params)

                for p in params:
                    key_labels.remove(p)
                    item_labels.remove(p)

                item_labels.append(new_label)
                key_labels.append(key)

        if full_labels:
            return item_labels
        else:
            return key_labels


    def variable_parameters(self, apply_mesh:bool = False
                            ) -> List[Tuple[States, str, np.ndarray]]:
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
            (i.e. stored as an array). They are ordered accordning to states,
            then according to variable name.
            Tuple entries of the list take the form `(states, param_name, value)`

        Examples
        --------
        >>> s = rq.Sensor(3)
        >>> vals = np.linspace(-1,2,3)
        >>> s.add_coupling(states=(1,2), rabi_frequency=vals, detuning=1)
        >>> s.add_coupling(states=(0,1), rabi_frequency=vals, detuning=vals)
        >>> for states, key, value in s.variable_parameters():
        ...     print(f"{states}: {key}={value}")
        (0, 1): detuning=[-1.   0.5  2. ]
        (0, 1): rabi_frequency=[-1.   0.5  2. ]
        (1, 2): rabi_frequency=[-1.   0.5  2. ]

        The order is important; in the unzipped case, it will sort as though all state labels
        were cast to strings, meaning integers will always be treated as first.
        
        >>> s = rq.Sensor([None, 'e1', 'e2'])
        >>> det1 = np.linspace(-1, 1, 3)
        >>> det2 = np.linspace(-1, 1, 5)
        >>> blue = {"states":(0,'e1'), "rabi_frequency":1, "detuning":det1}
        >>> red = {"states":('e1','e2'), "rabi_frequency":3, "detuning":det2}
        >>> s.add_couplings(blue, red)
        >>> for states, key, value in s.variable_parameters():
        ...    print(f"{states}: {key}={value}")
        >>> print(f"Axis Labels: {s.axis_labels()}")
        ('g', 1): detuning=[-1.  0.  1.]
        (1, 2): detuning=[-1.  -0.5  0.   0.5  1. ]
        Axis Labels: ["('g',1)_detuning", '(1,2)_detuning']

        """
        l=[]
        #function to compare mixed tuples
        def state_order(values):
            return [self.states.index(x) for x in values]

        for states in sorted(self.couplings.edges, key=state_order):
            
            edge_data = self.couplings.edges.get(states)
            
            for key, value in sorted(edge_data.items()):
                if not key.startswith("gamma") and key not in BASE_SCANNABLE_KEYS:
                    continue

                if isinstance(value, (list,np.ndarray)):
                    l.append((states, key, np.asarray(value)))
        
        if apply_mesh:
            vals = [val for _,_,val in l]
            mesh_vals = np.meshgrid(*vals, indexing='ij', sparse=True)
            mesh_vals = self._collapse_mesh(mesh_vals)
            l = [(*l[i][:2], mesh_vals[i].copy()) for i in range(len(l))]

        return l


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
        parameter_mesh = [v for _,_,v in self.variable_parameters(apply_mesh=True)]

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
        >>> s.zip_parameters("(0,1)_detuning", "(1,2)_detuning")
        >>> H = s.get_hamiltonian()
        >>> print(H.shape)
        (11, 3, 3)

        If the basis is provided as a list of string labels, the ordering of Hamiltonian rows
        And columns will correspond to the order of states provided.

        >>> s = rq.Sensor(['g', 'e1', 'e2'])
        >>> s.add_coupling(('g', 'e1'), detuning=1, rabi_frequency=1)
        >>> s.add_coupling(('e1', 'e2'), detuning=1.5, rabi_frequency=1)
        >>> print(s.get_hamiltonian())
        [[ 0. +0.j  0.5+0.j  0. +0.j]
         [ 0.5-0.j -1. +0.j  0.5+0.j]
         [ 0. +0.j  0.5-0.j -2.5+0.j]]

        """
        base_couplings = self.couplings.copy()

        #dictionary of state ordering used to determine indeces in ham
        int_states = {state: i for (i, state) in enumerate(self.states)}

        stack_shape = self._stack_shape(time_dependence='steady')

        for states, param, arr in self.variable_parameters(apply_mesh=True):

            self.couplings.edges[states][param] = arr

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

            # factor of 1/2 accounts for implicit rotating wave approximation
            hamiltonian[idx] = (f['rabi_frequency']*np.cos(f['phase'])
                                + 1j*f['rabi_frequency']*np.sin(f['phase']))/2
            hamiltonian[conj_idx] = np.conj(hamiltonian[idx])

        #add the numerical diagonal shifts
        for state1, state2, shift in nx.selfloop_edges(self.couplings, data='e_shift', default=0):
            
            shift_state = int_states[state1]
            idx = (..., shift_state, shift_state)

            hamiltonian[idx] += shift
        
        self.couplings = base_couplings
        return hamiltonian
    

    def _collapse_mesh(self, mesh):
        """Collapses the given mesh using rydiqule logic for parameter zipping.

        Expected to be given a mesh which is generated by applying `numpy.meshgrid`
        function on the output of :meth:`~.Sensor.variable_parameters` for the system.
        Given such a mesh, ensures that output mesh matches the shape expected by
        rydiqule's stacking convention, meaning that parameters that are zipped together
        will share an axis in the hamiltonian stack.

        Parameters
        ----------
        mesh : tuple(numpy.ndarray)
            The uncollapsed meshgid of parameters for the system. Typically the output of
            `numpy.meshgrid` called on :meth:`~.Sensor.variable_parameters`.

        Returns
        -------
        tuple(np.ndarray)
            The collapsed meshgid with zipped parameters sharing and axis.
        """

        mesh_copy = mesh.copy() #to not mess up original mesh
        
        labels_full = self.axis_labels(collapse=False) #uncollapsed labels
        labels_col = self.axis_labels(collapse=True, full_labels=True) #collapsed labels

        labels_split = [l.split("|") for l in labels_col]
        axes = np.zeros(len(labels_full), dtype=int)

        #generate a list of where each axis is 
        for i, l_base in enumerate(labels_full):
            for j in range(len(labels_col)):

                if l_base in labels_split[j]:
                    axes[i] = j

        #shift the "scanning" dimension of each parameter array to the appropriate axis
        for i, m in enumerate(mesh_copy):
            shape = np.ones(len(labels_col), dtype=int)
            shape[axes[i]] = m.size
            mesh_copy[i] = mesh_copy[i].reshape(shape)

        return mesh_copy


    def get_time_hamiltonians(self) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Get the hamiltonians for the time solver.

        Get both the steady state hamiltonian (as returned by :meth:`~.Sensor.get_hamiltonian`)
        and the time_dependent hamiltonians (as returned by :meth:`~.Sensor.get_time_couplings`).
        The time dependent hamiltonians give 2 terms, the hamiltonian corresponding
        to the real part of the coupling and the hamiltonian corresponding to the imaginary part.

        In the case where the basis of the `Sensor` was explicitly defined with a list
        of states, the ordering of rows and coulumns in the hamiltonian corresponds to the
        ordering of states passed in the basis. 

        Returns
        -------
        hamiltonian_base : np.ndarray
            The `(*l,n,n)` shape base hamiltonian of the system containing
            all elements that do not depend on time, where `n` is the basis size
            of the sensor.

        dipole_matrix_real : np.ndarray
            The `(M,n,n)` shape array of matrices representing the real
            time-dependent portion of the hamiltonian. For `0 <= i <= M`,
            the ith value along the first axis is the portion of the matrix which
            will be multiplied by the output of the ith `time_dependence` function.

        dipole_matrix_imag: nd.ndarray
            The `(M,n,n)` shape array of matrices representing the imaginary
            time-dependent portion of the hamiltonian. For `0 <= i <= M`,
            the ith value along the first axis is the portion of the matrix which
            will be multiplied by the output of the ith `time_dependence` function.

        Examples
        --------
        >>> s = rq.Sensor(2)
        >>> step = lambda t: 0. if t<1 else 1.
        >>> s.add_coupling(states=(0,1), detuning=1, rabi_frequency=1, time_dependence=step)
        >>> H_base, H_time_real, H_time_imaginary = s.get_time_hamiltonians()
        >>> print(H_base)
        >>> print(H_time_real)
        >>> print(H_time_imaginary)
        [[0.+0.j 0.+0.j]
        [0.+0.j 1.+0.j]]
        [array([[0. +0.j, 0.5+0.j],
            [0.5+0.j, 0. +0.j]])]
        [array([[0.+0.j , 0.+0.5j],
            [0.-0.5j, 0.+0.j ]])]

        If the basis is passed as a list, rows and columns are in the order specified:

        >>> s = rq.Sensor(['g', 'e'])
        >>> step = lambda t: 0. if t<1 else 1.
        >>> s.add_coupling(states=('g','e'), detuning=1, rabi_frequency=1, time_dependence=step)
        >>> H_base, H_time_real, H_time_imaginary = s.get_time_hamiltonians()
        >>> print(H_base)
        >>> print(H_time_real)
        >>> print(H_time_imaginary)
        [[ 0.+0.j  0.+0.j]
         [ 0.+0.j -1.+0.j]]
         [array([[0. +0.j, 0.5+0.j],
                [0.5+0.j, 0. +0.j]])]
         [array([[0.+0.j , 0.+0.5j],
                [0.-0.5j, 0.+0.j ]])]

        """
        hamiltonian_base = self.get_hamiltonian()
        dipole_matrices = self.get_time_couplings()

        return hamiltonian_base, *dipole_matrices


    def get_time_couplings(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Returns the list of matrices of all couplings in the system defined with
        a `time_dependence` key.

        The ouput will be two lists of matricies representing which terms of the hamiltonian
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
        >>> time_hams, time_hams_i = s.get_time_couplings()
        >>> for H in time_hams:
        ...     print(H)
        [[0.+0.j 1.+0.j 0.+0.j]
        [1.+0.j 0.+0.j 0.+0.j]
        [0.+0.j 0.+0.j 0.+0.j]]
        [[0.+0.j 0.+0.j 0.+0.j]
        [0.+0.j 0.+0.j 2.+0.j]
        [0.+0.j 2.+0.j 0.+0.j]]
            
        To handle stacking across the steady-state and time hamiltonians, the dimensions are 
        matched in a way that broadcasting works in a numpy-friendly way
        
        >>> s = rq.Sensor(3)
        >>> rabi = np.linspace(-1,1,11)
        >>> step = lambda t: 0 if t<1 else 1
        >>> blue = {"states":(0,1), "rabi_frequency":rabi, "detuning":1}
        >>> red = {"states":(1,2), "rabi_frequency":rabi, "detuning":0, 'time_dependence': step}
        >>> s.add_couplings(red, blue)
        >>> time_hams, time_hams_i = s.get_time_couplings()
        >>> print(s.get_hamiltonian().shape)
        >>> print(time_hams[0].shape)
        >>> print(time_hams_i[0].shape)
        (1, 11, 3, 3)
        (11, 1, 3, 3)
        (11, 1, 3, 3)

        """
        #save to re-add later
        base_couplings = self.couplings.copy()

        stack_shape = self._stack_shape(time_dependence='time')

        #dictionary of state ordering used to determine indeces in ham
        int_states = {state: i for (i, state) in enumerate(self.states)}

        for states, param, arr in self.variable_parameters(apply_mesh=True):
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
            
            #ignores numpy casting to real warning that we already account for
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if 'transition_frequency' in f:
                    time_hamiltonian[idx] = f['rabi_frequency']
                    time_hamiltonian[conj_idx] = np.conj(f["rabi_frequency"])
                    time_hamiltonian_i[idx] = 1j*f["rabi_frequency"]
                    time_hamiltonian_i[conj_idx] = np.conj(f["rabi_frequency"]*1j)
                
                #factor of 1/2 for rwa
                else:
                    time_hamiltonian[idx] = f['rabi_frequency']/2
                    time_hamiltonian[conj_idx] = np.conj(f["rabi_frequency"])/2
                    time_hamiltonian_i[idx] = 1j*f["rabi_frequency"]/2
                    time_hamiltonian_i[conj_idx] = np.conj(f["rabi_frequency"]*1j)/2

            matrix_list.append(time_hamiltonian)
            matrix_list_i.append(time_hamiltonian_i)
            
        self.couplings = base_couplings

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
            

        Examples:
            >>> s = rq.Sensor(3)
            >>> step = lambda t: 0 if t<1 else 1
            >>> wave = lambda t: np.sin(2000*np.pi*t)
            >>> f1 = {"states": (0,1), "transition_frequency":10, "rabi_frequency": 1, "time_dependence":wave}
            >>> f2 = {"states": (1,2), "transition_frequency":10, "rabi_frequency": 2, "time_dependence":step}
            >>> s.add_couplings(f1, f2)
            >>> print(s.get_time_dependence())
            [<function <lambda> at 0x7fb310edd9d0>, <function <lambda> at 0x7fb37c0c81f0>]

        """
        time_dependence = []
        for (i,j),f in self.couplings_with("time_dependence").items():
            time_dependence.append(f['time_dependence'])
        return time_dependence


    def get_hamiltonian_diagonal(self, values: dict, no_stack: bool=False) -> np.ndarray:
        """
        Apply addition and subtraction logic corresponding to the direction of the couplings.

        For a given state `n`, the path from ground will be traced to `n`.
        For each edge along this path,
        values will be added where the path direction and coupling direction match,
        and subtracting values where they do not.
        The sum of all such values along the path is the `n` th term in the output array. 

        Primarily for internal functions which help generate hamiltonians. Most commonly used
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
        >>> f3 = {"states": (2,3), "rabi_frequency": 3, "transition_frequency": 3}
        >>> s.add_couplings(f1, f2, f3)
        >>> print(s.get_value_dictionary("detuning"))
        {(0,1): 2, (1,2): 3}

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
        TypeError
            If `gamma_matrix` is not a numpy array.
        ValueError
            If `gamma_matrix` is not a square matrix of the appropriate size
        ValueError
            If the shape of `gamma_matrix` is not compatible with `self.basis_size`.

        Examples
        --------
        >>> s = rq.Sensor(2)
        >>> f1 = {"states": (0,1), "transition_frequency":10, "rabi_frequency": 1}
        >>> s.add_couplings(f1)
        >>> gamma = np.array([[.1,0],[.1,0]])
        >>> s.set_gamma_matrix(gamma)
        >>> print(s.decoherence_matrix())
        [[0.1 0. ]
        [0.1 0. ]]

        """
        if not isinstance(gamma_matrix, np.ndarray):
            raise TypeError(f'gamma_matrix must be a numpy array, not type {type(gamma_matrix)}')

        if gamma_matrix.shape != (self.basis_size,self.basis_size):
            raise ValueError((f'gamma_matrix has shape {gamma_matrix.shape}, '
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
        Returns the Hamiltonian with only detunings set to the kvector values for
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
        s_kvecs = [{k:v[i] for k,v in kvecs.items() if ~np.isclose(v[i],0)}
                   for i in range(spatial_dim)]
        if not any(s_kvecs):
            raise ValueError(('You must specify at least one non-zero '
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
        {(0, 1): {'rabi_frequency': 2, 'detuning': 1, 'phase': 0, 'kvec': (0, 0, 0), 'no_rwa_warning': False, 'label': '(0,1)'}}
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
    

    def states_with_label(self, label: str) -> List[State]:
        """Return a dict of all states with a label matching a given regular expression (regex) 
        pattern. The dictionary will be consist of keys which are string labels applied to states
        with the :meth:`~.Sensor.label_states` function, and values which are the corresponding
        integer values of the node on the graph. For more information on using regex patterns see 
        `this guide <https://docs.python.org/3/howto/regex.html#regex-howto>`.

        Parameters
        ----------
        label : string
            Regular expression pattern to match labels to. All labels matching the string will
            be returned in the keys of the dictionary. 

        Returns
        -------
        list
            List of all labels of states in the sensor which match the provided regex pattern.

        Raises
        ------
        ValueError
            If label is not a regular expression string.

        Examples
        --------
        >>> s = rq.Sensor(3)
        >>> s.add_coupling((0,1), detuning=1, rabi_freqency=1, label="hi mom")
        >>> s.add_coupling((1,2), detuning=2, rabi_requency=2)
        >>> s.label_states({0:"g", 1:"e1", 2:"e2"})
        >>> print(s.states_with_label("e[12]"))
        ['e1', 'e2']

        """
        try:
            re_match = re.compile(label)
        except TypeError:
            raise ValueError("label must be a regex string.")
        
        return [l for l in self.states if isinstance(l, str) and re_match.match(l) is not None]


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
        >>> s.add_coupling((0,1), detuning = 1, rabi_freqency=1, kvec=(0,0,1))
        >>> print(s.spatial_dim())
        1
        
        Multiple spatial dimensions can exist in a single coupling or 
        across multiple couplings
        
        >>> s = rq.Sensor(2)
        >>> s.add_coupling((0,1), detuning = 1, rabi_freqency=1, kvec=(1,0,1))
        >>> print(s.spatial_dim())
        2
        
        >>> s = rq.Sensor(3)
        >>> s.add_coupling((0,1), detuning = 1, rabi_freqency=1, kvec=(1,0,1))
        >>> s.add_coupling((1,2), detuning = 2, rabi_freqency=2, kvec=(0,1,0))
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
        ValueError
            If `states` has more than two elements.
        TypeError
            If `states` cannot be converted to a tuple.
        ValueError
            If either state in `states` is outside the basis.
        """
        try:
            tpl = tuple(states)
        except TypeError:
            raise ValueError(
                f'states argument of type {type(states)} cannot be interpreted as a tuple')
        if len(tpl) != 2:
            raise ValueError(
                f'A field must couple exactly 2 states, but {len(tpl)} are specified in {states}')
        for i in tpl:
            if i not in self.states:
                raise ValueError((f'State specification {i} is not a state in the basis'))

        return cast(States, tpl)  # cast for mypy


    def _stack_shape(self, time_dependence: Literal['steady', 'time', 'all']="all"
                     ) -> Tuple[int, ...]:
        """
        Internal function to get the shape of the tuple preceding the two hamiltonian
        axes in :meth:`~.get_hamiltonian()`

        """
        if len(self.variable_parameters()) == 0:
            return ()

        #make sure time_dependence is valid
        if time_dependence not in ["steady", "time", "all"]:
            raise ValueError("time_dependence must be one of 'steady', 'time', or 'all'")
        
        shape_array = np.array([m.shape for m in self.get_parameter_mesh()])
        stack_shape_full = np.max(shape_array, axis=0)

        axis_labels = self.axis_labels(full_labels=True)
        axis_labels_split = [l.split("|") for l in axis_labels]
        
        steady_idx = [] #axes which have steady-state variable parameters
        time_idx = [] #axes which have time-dependent variable parameters
        
        time_couplings = self.couplings_with("time_dependence").keys()
        time_rabi_labels = [str(s).replace(" ","")+"_rabi_frequency" for s in time_couplings]

        #find which axes have time vs steady-state parameters
        for i, l_list in enumerate(axis_labels_split):
            for l in l_list:
                if l in time_rabi_labels:
                    time_idx.append(i)
                else:
                    steady_idx.append(i)
        
        #apply steady and time indeces
        final_shape = np.ones_like(stack_shape_full)
        if time_dependence in ["steady", "all"]:
            final_shape[steady_idx] = stack_shape_full[steady_idx]
        if time_dependence in ["time", "all"]:
            final_shape[time_idx] = stack_shape_full[time_idx]
        
        return final_shape    


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
        >>> print(s.basis())
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
        ValueError
            If `kind` is not `'coherent'` and doesn't begin with `'gamma'`
        """

        if states not in self.couplings.edges:
            return

        if kind != 'coherent' and not kind.startswith('gamma'):
            msg = ("If clearing incoherent data,"
                   " must provide key to clear that starts with `gamma`, not {}")
            raise ValueError(msg.format(kind))

        #get rid of zipped couplings containing
        label = self.couplings.edges[states]["label"]
        zipped_idxs = []
        for idx, zip_strs in enumerate(self._zipped_parameters):
            zip_labels = [l.split("_",1)[0] for l in zip_strs]
            if label in zip_labels:
                zipped_idxs.append(idx)

        for idx in zipped_idxs[::-1]:
            del self._zipped_parameters[idx]
        
        # delete undesired keys from the edge
        for key in list(self.couplings.edges[states]):  # prevent generator
            if key == 'label':
                pass
            elif kind == 'coherent' and not key.startswith('gamma'):
                del self.couplings.edges[states][key]
            elif kind == key:
                del self.couplings.edges[states][key]
                
        # delete edge outright if it only has a label
        if not sum(k != 'label' for k in self.couplings.edges[states].keys()):
            self.couplings.remove_edge(*states)


    def _coupling_with_label(self, label: str) -> States:
        """
        Helper function to return the pair of states corresponding to a particular label string.
        For internal use.
        """
        label_map = {key:(state1, state2)
                     for state1, state2, key in self.couplings.edges(data="label")}
        if label in label_map.keys():
            return label_map[label]
        else:
            raise ValueError(f"No coupling with label {label}")
        
      
    def get_coupling_rabi(self, coupling_tuple: States = (0, 1)
                          ) -> Union[float, np.ndarray]:
        """
        Helper function that returns the Rabi frequency of the coupling
        from a Sensor for use in functions that return experimental values.

        Parameters
        ----------
        coupling_tuple: tuple of int
            The tuple that defines the coupling to extract to rabi frequencies from

        Returns
        -------
        float of numpy.ndarray
            Rabi frequency defined in the Sensor

        Warns
        -----
        UserWarning
            If the coupling has time dependence.
            In this case, the returned Rabi frequency may not be well defined.
        
        """
        coupling = self.couplings.edges[coupling_tuple]

        if coupling.get('time_dependence', False):
            warnings.warn(('Probe is time dependent.  Output of get_coupling_rabi '
                           'is not guaranteed to be well defined.'))

        rabi = coupling.get('rabi_frequency', None)
        if isinstance(rabi, (list, np.ndarray)):

            # get Rabi from parameter mesh so broadcasting works
            coupling_label = coupling['label']

            # get index in mesh for the scanned Rabi frequency
            labels = self.axis_labels()
            mesh_index = labels.index(coupling_label+'_rabi_frequency')
            parameter_mesh = self.get_parameter_mesh()

            return parameter_mesh[mesh_index]

        else:
            # it's just a number
            return rabi
