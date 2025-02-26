"""
Standard methods for converting results to physical values.
"""

from .sensor import Sensor
from .solvers import solve_steady_state

import numpy as np
import warnings

from typing import Tuple, List

from .exceptions import RydiquleError, TimeDependenceWarning


def get_snr(sensor: Sensor,
            param_label: str,
            phase_quadrature: bool = False,
            diff_nearest: bool = False,
            **kwargs
            ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Calculate a Sensor's signal-to-noise ratio in standard deviation,
    in a 1Hz bandwidth,
    to a specified signal parameter,
    assuming a homodyne measurement of optical field.

    SNR is calculated with respect to the signal parameter,
    relative to the inital value of the signal parameter.
    The returned mesh is similarly transformed from the typical sensor mesh,
    by replacing the total value of the signal parameter
    with the deviation in the signal parameter.

    The conventions used follow that of [1]_.

    Note
    ----
    The default is to return the SNR of an amplitude quadrature measurement.
    To convert to a power measurement (i.e. :math:`\\Omega_p^2`),
    the amplitude quadrature SNR must be divided by 2.
    To get the SNR in variance, square the result.

    Parameters
    ----------
    sensor : :class:`Sensor`
        sensor for which SNR should be calculated. The definition
        of sensor.couplings should contain at least one coupling with a list-like
        parameter. For the list-like parameter, the first array element is the
        "base" against which SNR for each other value is calculated.
    param_label : str
        Label of the axis with respect to which SNR is calculated.
        See :meth:`Sensor.axis_labels` for more details on axis labeling. The
        value corresponding to this label should be the list-like parameter
        with respect to which SNR should be calculated.
        This parameter list must have at least two elements,
        and SNR is calculated relative to the first element in the list
        for all other elements in the list.
    phase_quadrature : :obj:`bool`, optional
        Whether the sensor is measured in the
        phase quadrature of the probe laser. False denotes measurement in the
        amplitude quadrature.  Default is False.
    diff_nearest: bool, optional
        Controls method by which the SNR is calculated.
        The default (False) calculates the SNR with respect to the 0 index value.
        Setting True calculates the SNR with respect to nearest
        neighbor differences.
    kwargs : dict, optional
            Additional keyword arguments to pass to `rq.solve_steady_state()`. 

    Returns
    -------
    snrs : numpy.ndarray
        Array of SNRs for the sensor with respect to the change
        in the signal parameter.  Calculated in units of amplitude relative
        to noise standard deviation.  SNR referenced to 1 second BW.
    mesh : tuple(numpy.ndarray)
        Numpy meshgrid of the coupling parameters that yield each
        snr.  The signal parameter axis now shows the signal change.

    Raises
    ------
    RydiquleError
        If the specified param_label is not in `Sensor.axis_labels()`

    Examples
    --------
    >>> atom = "Rb85"
    >>> [g,e] = rq.D2_states("Rb85")
    >>> c = rq.Cell('Rb85', [g,e], cell_length=0.0001)
    >>> c.add_coupling(states=(g,e), rabi_frequency=np.linspace(1e-6, 1, 5), detuning=1, label="probe")
    >>> snr, mesh = rq.get_snr(c, 'probe_rabi_frequency')
    >>> print(snr) 
    [       0.       13947396.7 27887614.4 41813486.6
     55717871.1]
    >>> print(mesh) # doctest: +SKIP
    [array([0.      , 0.25    , 0.499999, 0.749999, 0.999999])]

    References
    ----------
    .. [1] D. H. Meyer, C. O'Brien, D. P. Fahey, K. C. Cox, and P. D. Kunz,
       "Optimal atomic quantum sensing using
       electromagnetically-induced-transparency readout,"
       Phys. Rev. A, vol. 104, p. 043103, 2021.
    """


    labels = sensor.axis_labels()
    try:
        sensitivity_axis = labels.index(param_label)
    except ValueError as err:
        raise RydiquleError(f"{param_label} label is not in sensor.axis_labels()") from err
    
    if len(sensor.couplings_with('time_dependence')):
        warnings.warn(TimeDependenceWarning('At least one coupling has time dependence. '
                                            'get_snr() only solves in steady-state; '
                                            'results may not be as expected.'))

    full_sols = solve_steady_state(sensor, **kwargs)
    rhos_ij = full_sols.coupling_coefficient_observable()

    _ = full_sols.get_OD()
    
    if diff_nearest:
        rho_diffs = np.diff(rhos_ij, axis = sensitivity_axis)
    else:
        rho_diffs = rhos_ij - np.take(rhos_ij, [0], sensitivity_axis)

    if phase_quadrature:
        rho_diffs_quadrature = np.abs(np.real(rho_diffs))
    else:
        rho_diffs_quadrature = np.abs(np.imag(rho_diffs))

    rho_ij_noise = full_sols.eta/full_sols.kappa  # follows directly from eqns 4 and 6 of 2105.10494
    # factor of 1e3 converts from root(MHz) to root(Hz)
    snrs: np.ndarray = rho_diffs_quadrature/rho_ij_noise*full_sols.cell_length*1e3

    mesh = sensor.get_parameter_mesh()
    mesh = [np.array(a) for a in np.broadcast_arrays(*mesh)] # avoid writing to views by copying
    mesh[sensitivity_axis] -= np.take(mesh[sensitivity_axis], [0], sensitivity_axis)

    return snrs, mesh
