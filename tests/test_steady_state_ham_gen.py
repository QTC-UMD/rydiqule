import numpy as np
import pytest
from rydiqule import Sensor
from rydiqule import Cell
from rydiqule import D2_states


@pytest.mark.steady_state
def test_ladder_5_level():

    f1 = {'states':(0,1), 'rabi_frequency':0, 'detuning':1}
    f2 = {'states':(1,2), 'rabi_frequency':0, 'detuning':2}
    f3 = {'states':(2,3), 'rabi_frequency':0, 'detuning':4}
    f4 = {'states':(3,4), 'rabi_frequency':0, 'detuning':8}

    sensor = Sensor(5)
    sensor.add_couplings(f1, f2, f3, f4)

    expected_ham = np.diag([0, -1, -1-2, -1-2-4, -1-2-4-8])
    ham = sensor.get_hamiltonian()

    np.testing.assert_allclose(ham, expected_ham,
                               err_msg="5 Level ladder failed hamilonian generation")


@pytest.mark.steady_state
def test_lambda_4_level():

    f1 = {'states':(0,1), 'rabi_frequency':0, 'detuning':1}
    f2 = {'states':(1,3), 'rabi_frequency':0, 'detuning':2}
    f3 = {'states':(2,3), 'rabi_frequency':0, 'detuning':4}

    sensor = Sensor(4)
    sensor.add_couplings(f1, f2, f3)

    expected_ham = np.diag([0, -1, -1-2+4, -1-2])
    ham = sensor.get_hamiltonian()

    np.testing.assert_allclose(ham, expected_ham,
                               err_msg="4 Level lambda failed hamilonian generation")


@pytest.mark.steady_state
def test_v_5_level():

    f1 = {'states':(0,1), 'rabi_frequency':0, 'detuning':1}
    f2 = {'states':(1,2), 'rabi_frequency':0, 'detuning':2}
    f3 = {'states':(2,3), 'rabi_frequency':0, 'detuning':4}
    f4 = {'states':(2,4), 'rabi_frequency':0, 'detuning':8}

    sensor = Sensor(5)
    sensor.add_couplings(f1, f2, f3, f4)

    expected_ham = np.diag([0, -1, -1-2, -1-2-4, -1-2-8])
    ham = sensor.get_hamiltonian()

    np.testing.assert_allclose(ham, expected_ham,
                               err_msg="5 Level V failed hamilonian generation")


@pytest.mark.steady_state
def test_v_5_cell():

    f1 = {'states':(0,1), 'rabi_frequency':0, 'detuning':1}
    f2 = {'states':(1,2), 'rabi_frequency':0, 'detuning':2}
    f3 = {'states':(2,3), 'rabi_frequency':0, 'detuning':4}
    f4 = {'states':(2,4), 'rabi_frequency':0, 'detuning':8}

    state1 = [10, 2, 2.5, 0.5]
    state2 = [11, 3, 3.5, 0.5]
    state3 = [9, 3, 3.5, 0.5]

    RbSensor_ss = Cell('Rb85', *D2_states(5), state1, state2, state3, cell_length = 0.00001)

    RbSensor_ss.add_couplings(f1, f2, f3, f4)

    expected_ham = np.diag([0, -1, -1-2, -1-2-4, -1-2-8])
    ham = RbSensor_ss.get_hamiltonian()

    np.testing.assert_allclose(ham, expected_ham,
                               err_msg="5 Level V cell failed hamilonian generation")
