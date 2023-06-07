"""
Standard methods for converting results to physical values.
"""

from .sensor import Sensor
from .cell import Cell
from .sensor_solution import Solution
from .sensor_utils import get_rho_ij
from .solvers import solve_steady_state

import numpy as np
import warnings

from scipy.constants import c

from typing import Tuple, Optional, Union


def get_transmission_coef(sol: Solution, cell: Cell, cell_length: float,
                          probe_tuple: Tuple[int, ...] = (0, 1)) -> np.ndarray:
    """
    Extract the transmission term from a solution.

    Assumes the optically-thin approximation is valid.

    Parameters
    ----------
    sol: :class:`~.Solution`
        A Solution object containing at least the `rho`
        attribute. Typically created as the return of :meth:`~.solve_steady_state`
        or `~.solve_time`.
    cell: :class:`~.Cell`
        The cell used to generate the solution. Used to get
        physical information needed for the calculation.
    cell_length: float
        Optical path length of the cell, in meters.
    probe_tuple: tuple of int, optional
        Tuple of probing coupling.
        Defaults to `(0,1)`.

    Returns
    -------
    numpy.ndarray
        Numerical value of the probe absorption in fractional units
        (P_out/P_in).

    Examples
    --------
    >>> c = rq.Cell('Rb85', *rq.D2_states('Rb85'))
    >>> c.add_coupling(states=(0,1), rabi_frequency=1, detuning=1)
    >>> sols = rq.solve_steady_state(c)
    >>> print(sols.rho.shape)
    >>> t = rq.get_transmission_coef(sols, c, 1e-3)
    >>> print(t)
    (3,)
    0.47653638415943955

    """
    probe_rabi = _get_probe_rabi(cell, probe_tuple)
    # reverse probe tuple order to get correct sign of imag
    rho_probe = get_rho_ij(sol.rho, *probe_tuple[::-1])
    OD = get_OD(rho_probe, cell_length, probe_rabi, cell.kappa)

    return np.exp(-OD)


def get_OD(rho_probe: np.ndarray, cell_length: float,
           probe_rabi: Union[float, np.ndarray], kappa: float,) -> np.ndarray:
    """
    Calculates the optical depth from the solution.

    Assumes the optically-thin approximation is valid.
    If a calculated OD for a solution exceeds 1,
    this approximation is likely invalid.

    Parameters
    ----------
    rho_probe: numpy.ndarray
        Array of matrix elements for the probing transition.
    cell_length: float
        Optical path length of the cell, in meters.
    probe_rabi: float or numpy.ndarray
        Probe Rabi frequency, in Mrad/s
    kappa: float
        kappa constant, in units of MHz/m

    Returns
    -------
    OD: numpy.ndarray
        Optical depth of the sample

    Warns
    -----
    UserWarning
        If any OD exceeds 1, which indicates the optically-thin approximation
        is likely invalid.

    Examples
    --------
    >>> c = rq.Cell('Rb85', *rq.D2_states('Rb85'))
    >>> c.add_coupling(states=(0,1), rabi_frequency=1, detuning=1)
    >>> sols = rq.solve_steady_state(c)
    >>> print(sols.rho.shape)
    >>> OD = rq.experiments.get_OD(rq.get_rho_ij(sols, 1, 0), 1e-3, 1, c.kappa)
    >>> print(OD)
    >>> OD2 = rq.experiments.get_OD(rq.get_rho_ij(sols, 1, 0), 1e-2, 1, c.kappa)
    >>> print(OD2)
    (3,)
    0.7412112017002291
    7.412112017002292
    ~/src/Rydiqule/src/rydiqule/experiments.py:103: UserWarning:
    At least one solution has optical depth greater than 1.
    Integrated results are likely invalid.

    """
    # calculate optical-depth assuming optically-then approximation is valid
    OD = kappa*rho_probe.imag*cell_length/probe_rabi
    if np.any(OD > 1.0):
        # optically-thin approximation probably violated
        warnings.warn(('At least one solution has optical depth '
                       'greater than 1. Integrated results are '
                       'likely invalid.'))

    return OD


def get_phase_shift(sol: Solution, cell: Cell, cell_length: float,
                    probe_tuple: Tuple[int, ...] = (0, 1)) -> np.ndarray:
    """
    Extract the phase shift from a solution.

    Assumes the optically-thin approximation is valid.

    Parameters
    ----------
    sol: :class:`~.Solution`
        Solution object to extract phase shift from.
    cell: :class:`~.Cell`
        The cell used to generate the solution. Used to get
        physical information needed for the calculation.
    cell_length: float
        Optical path length of the cell, in meters.
    probe_tuple: tuple of int, optional
        Tuple of probing coupling.
        Defaults to `(0,1)`.

    Returns
    -------
    numpy.ndarray
        Probe phase in radians.

    Examples
    --------
    >>> c = rq.Cell('Rb85', *rq.D2_states('Rb85'))
    >>> c.add_coupling(states=(0,1), rabi_frequency=1, detuning=1)
    >>> sols = rq.solve_steady_state(c)
    >>> print(sols.rho.shape)
    >>> phase_shift = rq.get_phase_shift(sols, c, 1e-3)
    >>> print(phase_shift)
    (3,)
    -0.03807271078849609

    """
    probe_rabi = _get_probe_rabi(cell, probe_tuple)
    # reverse probe tuple order to get correct sign of imag
    # not actually necessary for this function since only need real part
    rho_probe = get_rho_ij(sol.rho, *probe_tuple[::-1])

    return (cell.kappa/probe_rabi)*rho_probe.real*cell_length


def get_susceptibility(sol: Solution, cell: Cell,
                       probe_tuple: Tuple[int, ...] = (0, 1)) -> np.ndarray:
    """
    For a given density matrix solution and cell,
    return the atomic susceptibility on the probe transition.

    Parameters
    ----------
    sol: :class:`~.Solution`
        Solution object to extract susceptibility from.
    cell: :class:`~.Cell`
        The cell used to generate the solution. Used to get
        physical information needed for the calculation.
    probe_tuple: tuple of int, optional
        Tuple of probing coupling.
        Defaults to `(0,1)`.

    Returns
    -------
    numpy.ndarray
        Susceptibility of the density matrix solution.

    Examples
    --------
    >>> c = rq.Cell('Rb85', *rq.D2_states('Rb85'))
    >>> c.add_coupling(states=(0,1), rabi_frequency=1, detuning=1)
    >>> sols = rq.solve_steady_state(c)
    >>> print(sols.rho.shape)
    >>> sus = rq.get_susceptibility(sols, c)
    >>> print(f"{sus:.2f}")
    (3,)
    -24.40+474.99j

    """
    probe_rabi = _get_probe_rabi(cell, probe_tuple)
    # reverse probe tuple order to get correct sign of imag
    rho_probe = get_rho_ij(sol.rho, *probe_tuple[::-1])

    return 1e-6*(cell.kappa/probe_rabi)*rho_probe*cell.probe_freq/(2*c)


def get_solution_element(sols: Solution, idx: int) -> np.ndarray:
    """
    Return a slice of an n_dimensional matrix of solutions of shape (...,n^2-1),
    where n is the basis size of the quantum system.

    Parameters
    ----------
    sols: numpy.ndarray
        An N-Dimensional numpy array representing the final
        density matrix, with ground state removed, and written in the totally
        real equations basis. Can have arbitrary axes preceding density matrix
        axis.
    idx: int
        Solution index to slice.

    Returns
    -------
    numpy.ndarray
        Slice of solutions corresponding to index idx. For example,
        if sols has shape (..., n^2-1), sol_slice will have shape (...).

    Raises
    ------
    IndexError
        If `idx` in not within the shape determined by basis size.

    Examples
    --------
    >>> c = rq.Cell('Rb85', *rq.D2_states('Rb85'))
    >>> c.add_coupling(states=(0,1), rabi_frequency=1, detuning=1)
    >>> sols = rq.solve_steady_state(c)
    >>> print(sols.rho.shape)
    >>> rho_01 = rq.get_solution_element(sols, 0)
    >>> print(rho_01)
    (3,)
    -0.0013139903428765695

    """
    basis_size = np.sqrt(sols.rho.shape[-1] + 1)
    try:
        sol_slice = sols.rho.take(indices=idx, axis=-1)
    except IndexError:
        raise IndexError(f"No element with given index for {basis_size}-level system")

    return sol_slice


def get_snr(sensor: Sensor, optical_path_length: float, param_label: str,
            probe_tuple: Tuple[int, int] = (0, 1),
            phase_quadrature: bool = False,
            kappa: Optional[float] = None, eta: Optional[float] = None
            ) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Calculate a Sensor's signal-to-noise ratio, in a 1Hz bandwidth,
    to a specified signal parameter.

    SNR is calculated with respect to the signal parameter,
    relative to the inital value of the signal parameter.
    The returned mesh is similarly transformed from the typical sensor mesh,
    by replacing the total value of the signal parameter
    with the deviation in the signal parameter (relative
    to it's initial array value).

    The conventions used follow that of [1]_.

    Parameters
    ----------
    sensor : :class:`Sensor`
        sensor for which SNR should be calculated. The definition
        of sensor.couplings should contain at least one coupling with a list-like
        parameter. For the list-like parameter, the first array element is the
        "base" against which SNR for each other value is calculated.
    optical_path_length : int
        Sensor optical path length in meters.
    param_label : str
        Label of the axis with respect to which SNR is calculated.
        See :meth:`Sensor.axis_labels` for more details on axis labeling. The
        value corresponding to this label should be the list-like parameter
        with respect to which SNR should be calculated.
        This parameter list must have at least two elements,
        and SNR is calculated relative to the first element in the list
        for all other elements in the list.
    probe_tuple : tuple
        Two-integer tuple specifying the probing transition.
    phase_quadrature : :obj:`bool`, optional
        Whether the sensor is measured in the
        phase quadrature of the probe laser. False denotes measurement in the
        amplitude quadrature.  Default is False.
    kappa: float, optional
        Differential prefactor, in units of (Mrad/s)/m.
        Must be specified when using :class:`Sensor`.
    eta: float, optional
        Noise density prefactor, in units of root(Hz)
        Must be specified when using :class:`Sensor`.

    Returns
    -------
        snrs : numpy.ndarray
            Array of SNRs for the sensor with respect to the change
            in the signal parameter.  Calculated in units of amplitude relative
            to noise standard deviation.
        mesh : tuple(numpy.ndarray)
            Numpy meshgrid of the coupling parameters that yield each
            snr.  The signal parameter axis now shows the signal change.

    Raises
    ------
    ValueError
        If the specified param_label is not in `Sensor.axis_labels()`

    Examples
    --------
    >>> c = rq.Sensor()
    >>> c.add_coupling(states=(0,1), rabi_frequency=np.linspace(0, 1, 5), detuning=1)
    >>> snr, mesh = rq.get_snr(c, 0.01, '(0, 1)_rabi_frequency')
    >>> print(snr)
    >>> print(mesh)
    [     0. 137024. 273980. 410796. 547405.]
    [array([0., 0., 0., 1., 1.])]

    References
    ----------
    .. [1] D. H. Meyer, C. O'Brien, D. P. Fahey, K. C. Cox, and P. D. Kunz,
       "Optimal atomic quantum sensing using
       electromagnetically-induced-transparency readout,"
       Phys. Rev. A, vol. 104, p. 043103, 2021.
    """
    if (sensor.eta is not None or sensor.kappa is not None) and (eta is not None
                                                                 or kappa is not None):
        raise ValueError('Cannot provide alternate eta/kappa values from Sensor class')
    elif eta is None or kappa is None:
        if sensor.eta is None or sensor.kappa is None:
            raise ValueError(('Must specify both eta/kappa via arguments '
                              'to get_snr OR attributes of sensor.'))
        eta = sensor.eta
        kappa = sensor.kappa

    labels = sensor.axis_labels()
    try:
        sensitivity_axis = labels.index(param_label)
    except ValueError:
        raise ValueError(f"{param_label} label is not in sensor.axis_labels()")

    full_sols = solve_steady_state(sensor)
    rhos_ij = get_rho_ij(full_sols.rho, *probe_tuple[::-1])

    # check that OD isn't high so optically-thin approximation is valid
    probe_rabi = _get_probe_rabi(sensor, probe_tuple)
    _ = get_OD(rhos_ij, optical_path_length, probe_rabi, kappa)

    rho_diffs = rhos_ij - np.take(rhos_ij, [0], sensitivity_axis)

    if phase_quadrature:
        rho_diffs_quadrature = np.abs(np.real(rho_diffs))
    else:
        rho_diffs_quadrature = np.abs(np.imag(rho_diffs))

    rho_ij_noise = eta/kappa  # follows directly from eqns 4 and 6 of 2105.10494
    snrs: np.ndarray = rho_diffs_quadrature/rho_ij_noise*optical_path_length

    mesh = sensor.get_parameter_mesh(sparse=False)
    mesh[sensitivity_axis] -= np.take(mesh[sensitivity_axis], [0], sensitivity_axis)

    return snrs, mesh


def _get_probe_rabi(sensor: Sensor,
                    probe_tuple: Tuple[int, ...] = (0, 1)) -> Union[float, np.ndarray]:
    """
    Helper function that returns the correct probe Rabi frequency
    from a Sensor for use in functions that return experimental values.

    Parameters
    ----------
    sensor: :class:`~.Sensor`
        Sensor object that has the probe coupling defined
    probe_tuple: tuple of int
        The tuple that defines the probing coupling to extract

    Returns
    -------
    probe_rabi: float of numpy.ndarray
        Probe Rabi frequency defined in the Sensor

    Warns
    -----
    UserWarning
        If the probe coupling has time dependence.
        In this case, the returned Rabi frequency may not be well defined.
    
    """
    probe_coupling = sensor.couplings.edges[probe_tuple]

    if probe_coupling.get('time_dependence', False):
        warnings.warn(('Probe is time dependent.  Output of _get_probe_rabi '
                       'is not guaranteed to be well defined.'))

    rabi = probe_coupling.get('rabi_frequency', None)
    if isinstance(rabi, (list, np.ndarray)):

        # get Rabi from parameter mesh so broadcasting works
        probe_label = probe_coupling['label']

        # get index in mesh for the scanned Rabi frequency
        labels = sensor.axis_labels()
        mesh_index = labels.index(probe_label+'_rabi_frequency')
        parameter_mesh = sensor.get_parameter_mesh()

        return parameter_mesh[mesh_index]

    else:
        # it's just a number
        return rabi
