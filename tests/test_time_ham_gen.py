import numpy as np
import rydiqule as rq
import pytest

@pytest.mark.time
def test_time_ham_gen():
    s = rq.Sensor(3)
    step = lambda t: 0 if t<1 else 1
    s.add_coupling(states=(0,1),detuning=1, rabi_frequency=2+2j)
    s.add_coupling(states=(1,2), detuning=2, rabi_frequency=[-1,0,1], time_dependence=step)
    calc_ham = s.get_hamiltonian()
    calc_ham_t = s.get_time_couplings()[0][0]
    
    h_true = np.zeros((1,3,3), dtype=np.complex128)
    h_true += np.diag([0,-1,-3])
    h_true[:,0,1]=(2+2j)/2
    h_true[:,1,0]=(2-2j)/2
    
    h_t_true = np.zeros((3,3,3),dtype=np.complex128)
    h_t_true[:,1,2] += np.array([-1,0,1])/2
    h_t_true[:,2,1] = np.array([-1,0,1])/2
    
    np.testing.assert_array_equal(calc_ham, h_true, err_msg="Steady-hamiltonian does not match theory")
    np.testing.assert_array_equal(calc_ham_t, h_t_true, err_msg="Time Hamiltonian does not match theory")

