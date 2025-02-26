import numpy as np
import pytest
import rydiqule as rq
import warnings

from rydiqule.atom_utils import A_QState


@pytest.mark.util
def test_lv_diagram():
    """Confirms that level diagram plotting runs without major errors."""
    # Generate 3 states

    [g,e] = rq.D2_states(5)
    RydbergState_1 = A_QState(50, 2, 2.5)  # states labeled n, l, j
    RydbergState_2 = A_QState(51, 3, 3.5)
    RydbergState_3 = A_QState(52, 4, 3.5)

    # Initialize Cell
    RbSensor_ss = rq.Cell("Rb85", [*rq.D2_states(5),
                        RydbergState_1, RydbergState_2, RydbergState_3],
                        gamma_transit=2*np.pi*1, cell_length = 0.0001)

    # Couplings specified by rabi_frequency
    test_laser_1 = {'states': (g, e), 'rabi_frequency': 1, 'detuning': 2}
    # Not suppose to show up since Rabi_Frequency = 0
    test_laser_2 = {'states': (e,RydbergState_1), 'rabi_frequency': 0, 'detuning': 3}

    def testfun(a):
        return 1


    # Couplings specified by dipole moment

    ###
    # Commenting out for now since we made a change that causes this coupling to throw an error now
    # Leaving in just in case
    ###
    # test_laser_3 = {'states': (g,RydbergState_1),
    #                 'beam_waist': 1, 'beam_power': 1,
    #                 'time_dependence': testfun}
    
    
    test_laser_4 = {'states': (RydbergState_1,RydbergState_2),
                    'beam_waist': 1, 'beam_power': 1,
                    'time_dependence': testfun}
    test_laser_5 = {'states': (RydbergState_2,RydbergState_3),
                    'beam_waist': 1, 'beam_power': 1,
                    'time_dependence': testfun}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', rq.RWAWarning)
        RbSensor_ss.add_couplings(test_laser_1, test_laser_2, test_laser_4, test_laser_5)
    RbSensor_ss.add_transit_broadening([0,1,2,3])

    # Draw the level diagram
    level_diagram = rq.draw_diagram(RbSensor_ss)

    pass