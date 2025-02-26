import numpy as np
import pytest

import rydiqule as rq

@pytest.mark.time
@pytest.mark.backend
def test_rabi_flop_cyrk():
    """Tests that the cyrk.cyrk_ode backend runs"""

    def const(t):
        return 1
    
    blue_laser = {'states':(1,2), 'rabi_frequency':0.0, 'detuning':0, 'time_dependence': const}
    red_rabi_freq = 2*np.pi*4.4444
    red_laser = {'states':(0,1), 'rabi_frequency':red_rabi_freq, 'detuning':0}

    RbSensor = rq.Sensor(3)
    RbSensor.add_couplings(red_laser,blue_laser)
    gamma_matrix = np.zeros((3,3),dtype='double')
    gamma_matrix[0,1] = 0.1
    gamma_matrix[0,2] = 0.1
    gamma_matrix[1,2] = 0.1
    RbSensor.set_gamma_matrix(gamma_matrix)

    # RF field does not matter, just including one to test functionality
    solSize = RbSensor.basis_size * RbSensor.basis_size-1
    initCond = np.zeros(solSize)

    sampleNum = 1000
    endTime = 10.0
    solution = rq.solve_time(RbSensor, endTime, sampleNum, init_cond=initCond,rtol=1e-6,
                             solver='cyrk')
    fft1 = np.fft.fft(solution.rho[:,3])  # Checking if the main frequency is the Rabi frequency
    fft1 = fft1[:round(len(solution.t)/2)]
    fft1 = np.abs(fft1)

    freq_loc = np.argmax(fft1[1:])

    freq = np.linspace(np.pi/endTime, np.pi*sampleNum/endTime, len(fft1))
    val = freq[freq_loc]  # The strongest frequency value

    np.testing.assert_allclose(red_rabi_freq, val, rtol=0.05,
                               err_msg='cyrk_doe two level time solution not working')
