"""
Object used to store aspects of a solution when calling rydiule.solve()
Adds essential keys with "None" entries
"""
from __future__ import annotations
from typing import Optional, Union, List, cast

import copy

import numpy as np
import networkx as nx
from scipy.constants import c
import warnings
import itertools

# have to import this way to prevent circular imports
from rydiqule import sensor_utils
from .exceptions import RydiquleError, RydiquleWarning

Result = Union[float, np.ndarray]
ComplexResult = Union[complex, np.ndarray]

class Solution():
    """
    Results object that contains information from a solve.

    Methods implement a number of standard analyses on the result
    based on the density matrix observable formalism and
    Maxwell's equations for a plane wave in an
    optically-thin polarizable medium.
    """

    # attributes set in sensor
    _eta: Optional[float]
    """float, optional : Eta constant from the Cell.
    Not generally defined when using a Sensor."""
    _kappa: Optional[float]
    """float, optional : Kappa constant from the Cell.
    Not generally defined when using a Sensor."""
    _probe_tuple: Optional[sensor_utils.StateSpecs]
    """Coupling edge corresponding to probing field.
    Defined as the first added coupling when using a Sensor."""
    _probe_freq: Optional[float]
    """Probing transition frequency, in rad/s.
    Not generally defined when using a Sensor."""
    _cell_length: Optional[float]
    """Optical path length of the medium, in meters.
    Not generally defined when using a Sensor."""
    _beam_area: Optional[float]
    """Cross-sectional area of the probing beam, in square meters.
    Not generally defined when using a Sensor."""

    #attributes copied/computed from sensor
    couplings: nx.DiGraph
    """dict : Copy of the `Sensor.couplings` graph. """
    axis_labels: list[str]
    """list of str : Labels for the axes of scanned parameters.
    If doppler averaging but not summing, doppler dimensions are prepended."""
    axis_values: list
    """list : Value  corresponding to each axis.
    If doppler averaging but not summing, doppler classes in internal units are added."""
    rq_version: str
    """str : Version of rydiqule that created the Solution."""
    dm_basis: np.ndarray
    """list of str: The list of density matrix elements in the order they appear in the solution.
    See :meth:`Sensor.basis` for details."""

    #Sensor attributes stored for logistical simplicity
    variable_parameters: list
    """Output of the Sensor.variable_parameters() method for the Sensor used in a solve."""

    # doppler specific
    doppler_classes: Optional[np.ndarray]
    """numpy.ndarray, optional : Doppler classes used to perform the doppler average.
    Will be None if doppler averaging was not used."""

    #attributes from the solver
    rho: np.ndarray
    """numpy.ndarray : Solutions returned by the solver."""

    # time_solver specific
    t: np.ndarray
    """numpy.ndarray : Times the solution is returned at, when using the time solver.
    Undefined otherwise."""
    init_cond: np.ndarray
    """numpy.ndarray : Initial conditions, when using the time solver.
    Undefined otherwise."""


    # property attributes
    @property
    def eta(self) -> float:
        """Eta constant from the Cell.
        Not generally defined when using a Sensor."""
        if self._eta is None:
            raise RydiquleError("eta not defined. "
                                "Please define in Sensor and redo calculation")
        return self._eta
        

    @property
    def kappa(self) -> float:
        """Kappa constant from the Cell.
        Not generally defined when using a Sensor."""
        if self._kappa is None:
            raise RydiquleError("kappa not defined. "
                                "Please define in Sensor and redo calculation")
        return self._kappa
        
    
    @property
    def probe_tuple(self) -> sensor_utils.StateSpecs:
        """Coupling edge corresponding to probing field.
        Defined as the first added coupling when using a Sensor."""
        if self._probe_tuple is None:
            raise RydiquleError("probe_tuple not defined. "
                                "Please define in Sensor and redo calculation")
        return self._probe_tuple
        
    
    @property
    def probe_freq(self) -> float:
        """Probing transition frequency, in rad/s.
        Not generally defined when using a Sensor."""
        if self._probe_freq is None:
            raise RydiquleError("probe_freq not defined. "
                                "Please define in Sensor and redo calculation")
        return self._probe_freq


    @property
    def cell_length(self) -> float:
        """Optical path length of the medium, in meters.
        Not generally defined when using a Sensor."""
        if self._cell_length is None:
            raise RydiquleError("cell_length not defined. "
                                "Please define in Sensor and redo calculation")
        return self._cell_length
        

    @property
    def beam_area(self) -> float:
        """Cross-sectional area of the probing beam, in square meters.
        Not generally defined when using a Sensor."""
        if self._beam_area is None:
            raise RydiquleError("beam_area not defined. "
                                "Please define in Sensor and redo calculation")
        return self._beam_area
    

    @property
    def probe_rabi(self) -> ComplexResult:
        """Base laser rabi frequency of the probing transition of the `Sensor` used in a solve.

        The return of this function will be the appropriate shape to be cast using rydiqule's
        stacking convention. An error will be thrown if any of the base rabi frequencies
        of the probe coupling group do not match. (All rabi frequencies should match if 
        added using `add_coupling_group` and not overwritten)

        Returns
        -------
        The base rabi frequency of the transition defined as the rabi frequency of any of the
        couplings in the group divided by the `coherent_cc` on the corresponding edge (default 1).
        """
        return self.coupling_rabi(self.probe_tuple)
    

    @property
    def states(self) -> List[sensor_utils.State]:
        """Return a list of all states in the :class:`~.Sensor` used to produce solution.

        Returns
        -------
        list of states
            List of all states in the sensor used to produce solution. Order will match order in
            `Sensor`.
        """
        return list(self.couplings.nodes())
        

    def coupling_rabi(self, coupling: sensor_utils.StateSpecs) -> ComplexResult:
        """Rabi frequency, in Mrad/s, of a particular coupling or coupling group.
        
        Serves as a more general way of fetching a rabi frequency from the graph that supports
        couplings between manifolds. Throws an error if any of the couplings in the group specified 
        by `coupling`  do not have matching `rabi_frequency`, either in values or shape. 
        
        Parameters
        ----------
        coupling : tuple of states
            Pair of states or state manifolds specifying a coupling or coupling group respectively.
            If a group, all couplings in the group must have matching values and shapes for the
            `rabi_frequency`. This is most easily accomplished by adding the coupling to the group
            using :meth:`~.Sensor.add_coupling` with manifolds, but :meth:`~.Sensor.zip_parameters`
            can also be used.

        Returns
        -------
        float or np.ndarray
            The `rabi_frequency` of the coupling or of all couplings in the group specified by 
            `coupling`.

        Raises
        ------
        ValueError
            If `coupling` is a group with mismatched rabi frequencies, either in shape or value.

        Examples
        --------
        The basic functionality for couplings between single states mimics the functionality of
        just accessing the `rabi_frequency` attribute from the graph.
        
        >>> my_sensor = rq.Sensor(2)
        >>> my_sensor.add_coupling((0,1), rabi_frequency=1, detuning=1)
        >>> my_sensor.add_decoherence((1,0), 0.1)
        >>> sol = rq.solve_steady_state(my_sensor)
        >>> print(sol.couplings.edges[0,1]["rabi_frequency"])
        1
        >>> print(sol.coupling_rabi((0,1)))
        1

        It can also get the `rabi_frequency` for an entire coupling group. The `rabi_frequency` is
        treated as a laser property. Coupling coefficients as individual coupling properties, and so
        are not accounted for. 

        >>> g = (0,0)
        >>> e = (1, [-1,1])
        >>> cc = {
        ...     (g, (1,-1)): 0.5,
        ...     (g, (1,1)): 0.5
        ... }
        >>> my_sensor = rq.Sensor([g,e])
        >>> my_sensor.add_coupling((g,e), rabi_frequency=5, detuning=1, label="probe")
        >>> my_sensor.add_decoherence((e,g), 0.1, label="foo")
        >>> sol = rq.solve_steady_state(my_sensor)
        >>> print(sol.couplings.edges.data("rabi_frequency"))
        [((0, 0), (1, -1), 5), ((0, 0), (1, 1), 5), ((1, -1), (0, 0), None), ((1, 1), (0, 0), None)]
        >>> print(sol.coupling_rabi((g,e)))
        5

        """
        
        #check all rabis are match, we do this to avoid accidental misuse of the function
        states1 = sensor_utils.match_states(coupling[0], list(self.couplings.nodes))
        states2 = sensor_utils.match_states(coupling[1], list(self.couplings.nodes))

        rabis = [self.couplings.edges[s1,s2].get("rabi_frequency")
                 for s1,s2 in itertools.product(states1, states2)]

        #check all are the same shape
        #cast to array if another data type
        rabi_as_np = [np.asarray(r) for r in rabis]

        #ensure all the base rabis are the same. 
        # if they are not, they were probably added separately in sensor
        shape_is_match = all([r.shape==rabi_as_np[0].shape for r in rabi_as_np])
        value_is_match = np.all([np.isclose(r, rabi_as_np[0]) for r in rabi_as_np])

        if not (shape_is_match and value_is_match):
            raise ValueError("rabi freqencies do not match for all couplings in group")
        
        coupling_rabi_final = cast(ComplexResult, rabis[0])
        
        return coupling_rabi_final


    def rho_ij(self, i: sensor_utils.State, j: sensor_utils.State) -> ComplexResult:
        """
        Gets the full complex density matrix element corresponding to a given pair of indeces.

        Returns the entire array of density matrix elements corresponding to every combination of 
        parameters defined by the mesh of parameters in the system. The shape of the output
        array will match the `stack_shape` of the solution.
        
        Indeces can be provided either as integers which number the states, or using state
        labels (integers, tuples, or strings). For the case of tuples, only individual states
        can be used, full state specifications are invalid

        In the case of a state, it will be mapped to the corresponding integer, then the
        element itself is fetched using the :func:`~.get_rho_ij` function.

        Parameters
        ----------
        i: int, tuple, or string
            density matrix element row, or state label corresponding to row
        j: int, tuple or string
            density matrix element column, or state label corresponding to column

        Returns
        -------
        complex or numpy.ndarray
            `[i,j]` complex elment(s) of the density matrix

        Examples
        --------
        State labels and integer indeces can be used interchangeably.

        >>> rabis = np.linspace(1, 6, 5)
        >>> my_sensor = rq.Sensor(["g","e"])
        >>> my_sensor.add_coupling(("g","e"), rabi_frequency=rabis, detuning=1)
        >>> my_sensor.add_decoherence(("e","g"), 0.1)
        >>> sol = rq.solve_steady_state(my_sensor)
        >>> print(sol.rho_ij("g","e")) # doctest: +SKIP
        [0.3328-0.0166j 0.3184-0.0159j 0.2455-0.0123j 0.1933-0.0097j
         0.1579-0.0079j]
        >>> print(sol.rho_ij(0,1)) # doctest: +SKIP
        [0.3328-0.0166j 0.3184-0.0159j 0.2455-0.0123j 0.1933-0.0097j
         0.1579-0.0079j]

        """
        try:
            node_list = list(self.couplings.nodes())
            i_int = node_list.index(i)
            j_int = node_list.index(j)

        except ValueError:
            assert isinstance(i, int) and isinstance(j, int)
            i_int = i
            j_int = j

        return sensor_utils.get_rho_ij(self.rho, i_int, j_int)
    

    def get_solution_element(self, idx: int) -> Result:
        """
        Return a slice of an n_dimensional matrix of solutions of shape (...,n^2-1),
        where n is the basis size of the quantum system.

        Parameters
        ----------
        idx: int
            Solution index to slice.

        Returns
        -------
        float or numpy.ndarray
            Slice of solutions corresponding to index idx. For example,
            if sols has shape (..., n^2-1), sol_slice will have shape (...).

        Raises
        ------
        RydiquleError
            If `idx` in not within the shape determined by basis size.

        Examples
        --------
        >>> [g, e] = rq.D2_states("Rb85", expand=True)
        >>> c = rq.Cell('Rb85', [g, e])
        >>> c.add_coupling(states=(g, e), rabi_frequency=1, detuning=1)
        >>> sols = rq.solve_steady_state(c)
        >>> print(sols.rho.shape)
        (3,)
        >>> rho_01_im = sols.get_solution_element(0)
        >>> print(rho_01_im)
        0.0013711656

        """
        basis_size = np.sqrt(self.rho.shape[-1] + 1)
        try:
            sol_slice = self.rho.take(indices=idx, axis=-1)
        except IndexError as err:
            raise RydiquleError(f"No element with given index for {basis_size}-level system") from err

        return sol_slice
    

    def coupling_coefficient_matrix(self, coupling: sensor_utils.StateSpecs) -> np.ndarray:
        """Matrix of coefficients representing coupling strength for a particular coupling.

        Returns
        -------
        np.ndarray
            Adjacency matrix of `couplings` attributes
            generated from the `"coherent_cc"` parameter on the `coupling` edge.

        Examples
        --------
        >>> g = rq.ground_state(5, splitting="fs")
        >>> e = rq.D1_excited(5, splitting="fs")
        >>> my_cell = rq.Cell('Rb85', [g, e])
        >>> my_cell.add_coupling(states=(g, e), rabi_frequency=1, detuning=1, label="probe")
        >>> sol = rq.solve_steady_state(my_cell)
        >>> for e in sol.couplings.edges(data="coherent_cc"):
        ...     print(*e)
        (5, 0, 0.5, m_j=-0.5) (5, 1, 0.5, m_j=-0.5) -0.816496580927726
        (5, 0, 0.5, m_j=0.5) (5, 1, 0.5, m_j=0.5) 0.816496580927726
        (5, 1, 0.5, m_j=-0.5) (5, 0, 0.5, m_j=-0.5) None
        (5, 1, 0.5, m_j=-0.5) (5, 0, 0.5, m_j=0.5) None
        (5, 1, 0.5, m_j=0.5) (5, 0, 0.5, m_j=-0.5) None
        (5, 1, 0.5, m_j=0.5) (5, 0, 0.5, m_j=0.5) None
        >>> print(sol.coupling_coefficient_matrix(sol.probe_tuple))
        [[ 0.        0.        0.        0.      ]
         [ 0.        0.        0.        0.      ]
         [-0.816497  0.        0.        0.      ]
         [ 0.        0.816497  0.        0.      ]]
        """
        #integer values of each state that is in the upper and lower manifolds
        states1 = sensor_utils.match_states(coupling[0], list(self.couplings.nodes))
        states2 = sensor_utils.match_states(coupling[1], list(self.couplings.nodes))
        # have to rebuild graph only with edges we want
        # because networkx defaults to 1, not zero
        # on adjacency creation
        ag: nx.DiGraph = nx.DiGraph()
        ag.add_nodes_from(self.couplings.nodes)
        sg = self.couplings.subgraph(states1+states2)
        se = tuple(edge
                   for edge in sg.edges(data='coherent_cc')
                   if edge[2] is not None)
        ag.add_weighted_edges_from(se)

        coupling_adjacency_matrix = nx.to_numpy_array(ag).T.conj()

        return coupling_adjacency_matrix


    def get_observable(self, matrix: np.ndarray):
        """Returns the trace of a matrix product of the density matrix with a provided matrix.
        This is the standard definition of an measuremt of observable :math:`A` taken on a
        density matrix :math:`\\rho` given by :math:`tr(\\rho A)`. Note that this function
        first converts the density matrix into the standard complex basis rather than rydiqule's
        real basis, leading to potential round-off errors for *very* small density matrix elements. 

        The proveded observable wil respect `rydiqule` stacking convention, and the labels for
        each axis can be recovered via the `axis_labels` attribute.

        Parameters
        ----------
        matrix : np.ndarray
            The matrix representing the observable to be computed.

        Returns
        -------
        np.ndarray
            Array of observables using rydiqule's stacking convention.

        """
        rho_complex = sensor_utils.convert_dm_to_complex(self.rho)
        prod = rho_complex @ matrix

        tr_full = np.trace(prod, axis1=-1, axis2=-2)
        return tr_full


    def coupling_coefficient_observable(self, coupling: Optional[sensor_utils.StateSpecs]=None):
        """Get the observable associated with the output of 
        :meth:`~.sensor_solution.coupling_coefficient_matrix`.

        Calls the :meth:`~.sensor_solution.get_observable` function with the output of
        :meth:`~.sensor_solution.coupling_coefficient_matrix`, with rows and columns
        limited those defined by the probe coupling. If `coupling` is `None`, uses the
        tuple defined by `probe_tuple`. 

        This function is designed to get observable values associated with the probe laser in an
        experiment, such as transmission and absorption coefficients.

        Parameters
        ----------
        coupling : tuple of int or string, optional
            The 2-length tuple of state specifications to use in the observable calculation. Each state can
            be either an `int` or a regex string corresponding to a group of states. If `None`, uses the `probe_tuple`
            attribute as defined in the `Sensor` used to produce the solution. Defaults to `None`.

        Returns
        -------
        np.ndarray
            The observable or array of observables correspondinging to the coupling coefficients. 

        """
        if coupling is None:
            coupling = self.probe_tuple

        probe_coefficient_matrix = self.coupling_coefficient_matrix(coupling)

        return self.get_observable(probe_coefficient_matrix)


    def get_susceptibility(self) -> ComplexResult:
        """
        Return the atomic susceptibility on the probe transition.

        Experimental parameters must be defined manually for a `Sensor`.

        Returns
        -------
        complex or numpy.ndarray
            Susceptibility of the density matrix solution.

        Examples
        --------
        >>> [g, e] = rq.D2_states(5, expand=True)
        >>> c = rq.Cell('Rb85', [g, e], cell_length = 0.0001)
        >>> c.add_coupling(states=(g,e), rabi_frequency=1, detuning=1)
        >>> sols = rq.solve_steady_state(c)
        >>> print(sols.rho.shape)
        (3,)
        >>> sus = sols.get_susceptibility()
        >>> print(sus)
        (1.9734254e-05+0.000376067j)

        """
        probe_rabi = self.probe_rabi*1e6 #rad/s
        kappa = self.kappa
        probe_freq = self.probe_freq

        # reverse probe tuple order to get correct sign of imag
        rho_probe = self.coupling_coefficient_observable()

        # See Steck for last factor of 2.  Comes from Steck QO Notes page
        return kappa * (rho_probe * 2*c) / (probe_freq * (probe_rabi/2))


    def get_OD(self) -> Result:
        """
        Calculates the optical depth from the solution.  This equation comes from
        Steck's Quantum Optics Notes Eq. 6.74.

        Assumes the optically-thin approximation is valid.
        If a calculated OD for a solution exceeds 1,
        this approximation is likely invalid.

        Returns
        -------
        OD: float or numpy.ndarray
            Optical depth of the sample

        Warns
        -----
        UserWarning
            If any OD exceeds 1, which indicates the optically-thin approximation
            is likely invalid.

        Examples
        --------
        >>> [g, e] = rq.D2_states('Rb85')
        >>> c = rq.Cell('Rb85', [g, e], cell_length =  0.0001)
        >>> c.add_coupling(states=(g, e), rabi_frequency=1, detuning=1)
        >>> sols = rq.solve_steady_state(c)
        >>> print(sols.rho.shape)
        (3,)
        >>> OD = sols.get_OD()
        >>> print(OD)
        0.3028422896

        """

        probe_wavelength = np.abs(c/(self.probe_freq/(2*np.pi))) #meters
        probe_wavevector = 2*np.pi/probe_wavelength #1/meters
        OD = self.get_susceptibility().imag*self.cell_length*probe_wavevector
        if np.any(OD > 1.0):
            # optically-thin approximation probably violated
            msg = 'At least one solution has optical depth '\
                    'greater than 1. Integrated results are likely invalid.'
            warnings.warn(msg, RydiquleWarning)

        return OD


    def get_transmission_coef(self) -> Result:
        """
        Extract the transmission term from a solution.

        Assumes the optically-thin approximation is valid.

        Returns
        -------
        float or numpy.ndarray
            Numerical value of the probe absorption in fractional units
            (P_out/P_in).

        Examples
        --------
        >>> [g, e] = rq.D2_states('Rb85')
        >>> c = rq.Cell('Rb85', [g, e], cell_length =  0.0001)
        >>> c.add_coupling(states=(g, e), rabi_frequency=1, detuning=1)
        >>> sols = rq.solve_steady_state(c)
        >>> print(sols.rho.shape)
        (3,)
        >>> t = sols.get_transmission_coef()
        >>> print(t)
        0.73871559029

        """
        OD = self.get_OD()
        transmission_coef = np.exp(-OD)
        return transmission_coef
    

    def get_phase_shift(self) -> Result:
        """
        Extract the phase shift from a solution.

        Assumes the optically-thin approximation is valid.

        Returns
        -------
        float or numpy.ndarray
            Probe phase in radians.

        Examples
        --------
        >>> [g, e] = rq.D2_states('Rb85')
        >>> c = rq.Cell('Rb85', [g, e], cell_length =  0.00001)
        >>> c.add_coupling(states=(g, e), rabi_frequency=1, detuning=1)
        >>> sols = rq.solve_steady_state(c)
        >>> print(sols.rho.shape)
        (3,)
        >>> print(sols.get_phase_shift())
        80.5295218956
        
        """
        # reverse probe tuple order to get correct sign of imag
        # not actually necessary for this function since only need real part
        susc = self.get_susceptibility()
        n_refrac = 1+susc.real/2 #steck Quantum Optics Notes eq 6.71
        wavelength = 2*np.pi*c/self.probe_freq
        wavevector = 2*np.pi/wavelength
        phaseshift = n_refrac*self.cell_length*wavevector

        return phaseshift


    def copy(self):
        return copy.copy(self)
    
    
    def deepcopy(self):
        return copy.deepcopy(self)                                 
