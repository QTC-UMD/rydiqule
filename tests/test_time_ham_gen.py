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
    calc_ham_t = s.get_time_hamiltonian_components()[0][0]
    
    h_true = np.zeros((1,3,3), dtype=np.complex128)
    h_true += np.diag([0,-1,-3])
    h_true[:,0,1]=(2+2j)/2
    h_true[:,1,0]=(2-2j)/2
    
    h_t_true = np.zeros((3,3,3),dtype=np.complex128)
    h_t_true[:,1,2] += np.array([-1,0,1])/2
    h_t_true[:,2,1] = np.array([-1,0,1])/2
    
    np.testing.assert_array_equal(calc_ham, h_true, err_msg="Steady-hamiltonian does not match theory")
    np.testing.assert_array_equal(calc_ham_t, h_t_true, err_msg="Time Hamiltonian does not match theory")


@pytest.mark.time
def test_time_steady_ham_phase():

    s = rq.Sensor(2)

    tf = lambda t: 1 # dummy time function

    def get_ham(rabi, phase=0, cc=None, tf=None):
        s.add_coupling(states=(0,1), detuning=0,
                       rabi_frequency=rabi,
                       phase=phase,
                       coherent_cc=cc,
                       time_dependence=tf)
        return s.get_time_hamiltonian(0)

    # mixed complex
    mc1 = get_ham((1+1j)/np.sqrt(2))
    mc2 = get_ham(1, phase=np.pi/4)
    mc3 = get_ham((1+1j)/np.sqrt(2), tf=tf)
    mc4 = get_ham(1, phase=np.pi/4, tf=tf)

    np.testing.assert_allclose(mc1, mc2,
                               err_msg='Steady-state phase offset incorrect')
    np.testing.assert_allclose(mc1, mc3,
                               err_msg='Time complex Rabi incorrect')
    np.testing.assert_allclose(mc1, mc4,
                               err_msg='Time phase offset incorrect')


@pytest.mark.time
def test_time_steady_ham_cc():

    s = rq.Sensor(2)

    tf = lambda t: 1 # dummy time function

    def get_ham(rabi, phase=None, cc=None, tf=None, rwa=True):
        if rwa:
            d = {'detuning':0}
        else:
            d = {'transition_frequency':0}
        s.add_coupling(states=(0,1), 
                       rabi_frequency=rabi,
                       phase=phase,
                       coherent_cc=cc,
                       time_dependence=tf,
                       **d)
        return s.get_time_hamiltonian(0)

    # cc
    cc1 = get_ham(-1/3)
    cc2 = get_ham(1, cc=-1/3)
    cc3 = get_ham(-1/3, tf=tf)
    cc4 = get_ham(1, cc=-1/3, tf=tf)
    cc5 = get_ham(1, cc=-1/3, tf=tf, rwa=False)

    cc11 = get_ham(-1j/3)
    cc12 = get_ham(1/3, cc=-1j)
    cc13 = get_ham(1/3, cc=-1j, tf=tf)
    cc14 = get_ham(1/3, cc=-1j, tf=tf, rwa=False)

    np.testing.assert_allclose(cc1, cc2,
                                err_msg='Steady-state CC incorrect')
    np.testing.assert_allclose(cc1, cc3,
                                err_msg='Time negative Rabi incorrect')
    np.testing.assert_allclose(cc1, cc4,
                                err_msg='Time CC incorrect')
    np.testing.assert_allclose(cc1, cc5/2, # should differ by 1/2
                                err_msg='Steady-state CC non-RWA incorrect')
    np.testing.assert_allclose(cc11, cc12,
                            err_msg='Complex Rabi Steady-state CC incorrect')
    np.testing.assert_allclose(cc11, cc13,
                            err_msg='Complex Rabi Time CC incorrect')
    np.testing.assert_allclose(cc11, cc14/2, # should differ by 1/2
                            err_msg='Complex Rabi Time CC non-RWA incorrect')