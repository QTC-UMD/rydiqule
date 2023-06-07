import numpy as np
import pytest
import rydiqule as rq


@pytest.mark.util
def test_lv_diagram():
    """Confirms that level diagram plotting runs without major errors."""
    # Generate 3 states
    RydbergState_1 = [50, 2, 2.5, 0.5]  # states labeled n, l, j, m_j
    RydbergState_2 = [51, 3, 3.5, 0.5]
    RydbergState_3 = [52, 4, 3.5, 0.5]

    # Initialize Cell
    RbSensor_ss = rq.Cell("Rb85", *rq.D2_states(5),
                          RydbergState_1, RydbergState_2, RydbergState_3,
                          gamma_transit=2*np.pi*1)

    # Couplings specified by rabi_frequency
    test_laser_1 = {'states': (0,2), 'rabi_frequency': 1, 'detuning': 2}
    # Not suppose to show up since Rabi_Frequency = 0
    test_laser_2 = {'states': (1,3), 'rabi_frequency': 0, 'detuning': 3}

    def testfun(a):
        return 0

    # Couplings specified by dipole moment
    test_laser_3 = {'states': (0,3), 'rabi_frequency': 7,
                    'beam_waist': 1, 'time_dependence': testfun}
    test_laser_4 = {'states': (2,3), 'rabi_frequency': 8,
                    'beam_waist': 1, 'time_dependence': testfun}
    test_laser_5 = {'states': (3,4), 'rabi_frequency': 8,
                    'beam_waist': 1, 'time_dependence': testfun}

    RbSensor_ss.add_couplings(test_laser_1, test_laser_2, test_laser_3, test_laser_4, test_laser_5,
                              suppress_dipole_warn=True, suppress_rwa_warn=True)
    RbSensor_ss.add_transit_broadening([0,1,2,3])

    # Draw the level diagram
    level_diagram = rq.draw_diagram(RbSensor_ss)

    # Remove the couplings defined by rabi_frequency of 0.
    coupling_list = [test_laser_1, test_laser_2, test_laser_3, test_laser_4, test_laser_5]
    coupling_list = [x for x in coupling_list
                     if "rabi_frequency" in x.keys() and x.get('rabi_frequency') != 0]\
        + [x for x in coupling_list if "beam_power" in x.keys()]
        
    gamma_list = [x for x in coupling_list
                     if "gamma_transit" in x.keys()]

    total_arrows = len([item for sublist in level_diagram.arrows for item in sublist])
    total_levels = len(level_diagram.energies)

    # Verify that the function gets the correct number of arrows
    # (representing the couplings) and horizontal lines (representing the levels)
    assert total_levels == 5, \
        "Number of horizontal lines in the plot is not equal to number of states defined"
    assert total_arrows == len(coupling_list) + len(gamma_list), \
        "Number of arrows in the plot is not equal to number of couplings and dephasings defined"
