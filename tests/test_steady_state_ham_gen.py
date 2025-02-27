import numpy as np
import pytest
from rydiqule import Sensor
from rydiqule import Cell
from rydiqule import D2_states

from rydiqule.atom_utils import A_QState


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
def test_mixed_state_ladder():

    f1 = {'states':('g',1), 'rabi_frequency':0, 'detuning':1}
    f2 = {'states':(1,2), 'rabi_frequency':0, 'detuning':2}
    
    s = Sensor(['g', 1, 2])
    s.add_couplings(f1, f2)

    expected_ham = np.diag([0, -1, -1-2])
    ham = s.get_hamiltonian()
    
    np.testing.assert_allclose(ham, expected_ham,
                               err_msg="5 Level ladder failed hamilonian generation")

@pytest.mark.steady_state
def test_lambda_4_level():

    f1 = {'states':(0,1), 'rabi_frequency':0, 'detuning':1}
    f2 = {'states':(1,3), 'rabi_frequency':0, 'detuning':2}
    f3 = {'states':(2,3), 'rabi_frequency':0, 'detuning':4}

    s_f1 = {'states':('a','b'), 'rabi_frequency':0, 'detuning':1}
    s_f2 = {'states':('b','d'), 'rabi_frequency':0, 'detuning':2}
    s_f3 = {'states':('c','d'), 'rabi_frequency':0, 'detuning':4}

    sensor = Sensor(4)
    sensor.add_couplings(f1, f2, f3)

    sensor_str = Sensor(['a', 'b', 'c', 'd'])
    sensor_str.add_couplings(s_f1, s_f2, s_f3)

    expected_ham = np.diag([0, -1, -1-2+4, -1-2])
    ham = sensor.get_hamiltonian()
    ham_str = sensor.get_hamiltonian()

    np.testing.assert_allclose(ham, expected_ham,
                               err_msg="4 Level lambda failed hamilonian generation")
    np.testing.assert_allclose(ham_str, expected_ham,
                               err_msg="4 Level lambda failed ham generation with str labels")


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

    [g, e] = D2_states(5)  

    state1 = A_QState(10, 2, 2.5)
    state2 = A_QState(11, 3, 3.5)
    state3 = A_QState(9, 3, 3.5)

    f1 = {'states':(g,e), 'rabi_frequency':0, 'detuning':1}
    f2 = {'states':(e,state1), 'rabi_frequency':0, 'detuning':2}
    f3 = {'states':(state1,state2), 'rabi_frequency':0, 'detuning':4}
    f4 = {'states':(state1,state3), 'rabi_frequency':0, 'detuning':8}

    RbSensor_ss = Cell('Rb85', [g, e, state1, state2, state3], cell_length = 0.00001)

    RbSensor_ss.add_couplings(f1, f2, f3, f4)

    expected_ham = np.diag([0, -1, -1-2, -1-2-4, -1-2-8])
    ham = RbSensor_ss.get_hamiltonian()

    np.testing.assert_allclose(ham, expected_ham,
                               err_msg="5 Level V cell failed hamilonian generation")
  
@pytest.mark.steady_state
def test_e_shift():
    f1 = {'states':('g','e1'), 'rabi_frequency':0, 'detuning':1}
    f2 = {'states':('e1','e2'), 'rabi_frequency':0, 'detuning':2}

    s = Sensor(['g', "e1", "e2"])
    s.add_couplings(f1, f2)
    s.add_energy_shift('g', .5)
    s.add_energy_shift(['e1', 'e2'], 0.5, prefactors={"e1":2, "e2":3})

    expected_ham = np.diag([0, -1, -1-2]) + np.diag([.5, 1, 1.5])
    ham = s.get_hamiltonian()

    np.testing.assert_allclose(ham, expected_ham,
                               err_msg='State shifts failed hamiltonian generation')