import pytest
import numpy as np

# defines common fixtures used across component tests


@pytest.fixture
def Rb85_sensor_kwargs():
    """Defines a common set of sensor parameters for steady-state tests.
    """
    sensor_init_kwargs = {'gamma_transit': None,
                          'cell_length': 0,
                          'beam_area': 1e-6,
                          'beam_diam': None,
                          'temp': 300.0,
                          }
    return sensor_init_kwargs


def _find_zero_crossings(sols,detunings,sign='neg'):
    # ensure arrays are 1-D of same size
    assert sols.shape == detunings.shape
    assert len(sols.shape) == 1

    # find zero crossings
    crossing_inds = np.where(np.diff(np.signbit(sols)))[0]
    # only return zero crossings with gradient of `sign`
    sols_grad = np.gradient(sols)
    if sign == 'pos':
        crossing_inds = crossing_inds[sols_grad[crossing_inds] > 0]
    elif sign == 'neg':
        crossing_inds = crossing_inds[sols_grad[crossing_inds] < 0]
    # precisely calculate crossing points vs detuning, since
    # provided calc may not be evaluated at the exact zero crossing
    det_crossings = []
    for cross in crossing_inds:
        (x1, y1) = (detunings[cross], sols[cross])
        (x2, y2) = (detunings[cross+1], sols[cross+1])
        det_crossings.append(x1-(x2-x1)/(y2-y1)*y1)

    return det_crossings


@pytest.fixture
def find_zero_crossings():
    """Provides a function that can find zero crossings."""

    return _find_zero_crossings
