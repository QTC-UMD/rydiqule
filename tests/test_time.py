import numpy as np
import pytest

import rydiqule as rq
from rydiqule.atom_utils import A_QState


@pytest.mark.time
@pytest.mark.slow
@pytest.mark.doppler
def test_rabi_flop_doppler():
    """Tests that time solver functions with Doppler broadening
    """
    def const(t):
        return 1
    kp = 4*np.array([1,0,0])

    red_rabi_freq = 4.4444
    red_laser = {'states':(0,1), 'rabi_frequency':red_rabi_freq,
                 'detuning':0, 'kvec':kp, 'time_dependence': const}

    sensor = rq.Sensor(3, vP=0.001)

    sensor.add_couplings(red_laser)
    gamma = np.zeros((3,3))
    gamma[0,1] = 0.1
    sensor.set_gamma_matrix(gamma)

    # RF field does not matter, just including one to test functionality
    solSize = sensor.basis_size * sensor.basis_size-1
    initCond = np.zeros(solSize)

    dop_meth = {"method":"split", "n_doppler":20, "n_coherent":40}

    sampleNum = 4000
    endTime = 20.0
    solution = rq.solve_time(sensor, endTime, sampleNum,
                             init_cond=initCond, doppler_mesh_method=dop_meth,
                             doppler=True)
    fft1 = np.fft.fft(solution.rho[:,3])  # Checking if the main frequency is the Rabi frequency
    fft1 = fft1[1:round(len(solution.t)/2)]
    fft1 = np.abs(fft1)

    freq = np.linspace(np.pi/endTime, np.pi*sampleNum/endTime, len(fft1))
    freq_loc = np.argmax(fft1)

    val = freq[freq_loc]  # The strongest frequency value

    np.testing.assert_allclose(red_rabi_freq, val, rtol=0.05,
                               err_msg='Two level time solution not working with Doppler')


@pytest.mark.time
def test_rabi_flop():
    """Tests that time solver functions without Doppler broadening
    """

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
    solution = rq.solve_time(RbSensor, endTime, sampleNum, init_cond=initCond,rtol=1e-6)
    fft1 = np.fft.fft(solution.rho[:,3])  # Checking if the main frequency is the Rabi frequency
    fft1 = fft1[:round(len(solution.t)/2)]
    fft1 = np.abs(fft1)

    freq_loc = np.argmax(fft1[1:])

    freq = np.linspace(np.pi/endTime, np.pi*sampleNum/endTime, len(fft1))
    val = freq[freq_loc]  # The strongest frequency value

    np.testing.assert_allclose(red_rabi_freq, val, rtol=0.05,
                               err_msg='Two level time solution not working without Doppler')


@pytest.mark.time
@pytest.mark.slow
def test_time_match_steady():
    """Uses a single tone RF field to check if the time solver starting at zero reached steady state
    """

    [g, e] = rq.D2_states(5)
    rydberg_target_state = A_QState(150, 2, 2.5)
    rydberg_excited_state = A_QState(149, 3, 3.5)

    red_laser = {'states':(g,e), 'rabi_frequency':2*np.pi*4.0, 'detuning':2.0}
    blue_laser = {'states':(e,rydberg_target_state), 'rabi_frequency':2*np.pi*6.0, 'detuning':-1.0}
    rf_coupling_ss = {'states': (rydberg_target_state,rydberg_excited_state), 'rabi_frequency':2*np.pi*5.0, 'detuning': 0}

    RbSensor_ss = rq.Cell("Rb85", [g, e, rydberg_target_state, rydberg_excited_state],
                          cell_length=0,
                          gamma_transit=0.1)
    RbSensor_t = rq.Cell("Rb85", [g, e, rydberg_target_state, rydberg_excited_state],
                         cell_length=0,
                         gamma_transit=0.1)

    RbSensor_ss.add_couplings(red_laser, blue_laser, rf_coupling_ss)

    solSize = RbSensor_ss.basis_size * RbSensor_ss.basis_size-1
    initCond = np.zeros(solSize)

    (n, l, j, _, _, _) = RbSensor_ss.states[2]
    (n2, l2, j2, _, _, _) = RbSensor_ss.states[3]

    rf_freq = RbSensor_ss.atom.arc_atom.getTransitionFrequency(n2,l2,j2,n,l,j)*1E-6

    def rf_carrier(t):
        return np.cos(2*np.pi*rf_freq*t)

    rf_coupling_t = {'states': (rydberg_target_state, rydberg_excited_state), 'rabi_frequency':2*np.pi*5.0, 'time_dependence': rf_carrier}
    RbSensor_t.add_couplings(red_laser, blue_laser, rf_coupling_t)

    sampleNum = 10000
    endTime = 5

    # currently passes with end time 25 and Rtol = atol = 1e-4
    time_sol = rq.solve_time(RbSensor_t, endTime, sampleNum, initCond, rtol=1e-6, atol=1e-6)
    steady_sol = rq.solve_steady_state(RbSensor_ss)

    # certain off-diagonal elements will not match.
    # For now, test 1 diagonal and one off-diagonal element.
    np.testing.assert_allclose(steady_sol.rho[:2], time_sol.rho[-1][:2], rtol=0.1,
                               err_msg='Time solution does not converge to steady state')


@pytest.mark.time
def test_time_complex_match_steady():
    """Tests the time solver vs steady state
    with aribtrary (complex) and time-dependent rabi frequencies
    """

    def fun(t):
        if t < 1:
            return 0
        else:
            return 1

    def fun2(t):
        if t < 1:
            return 0
        else:
            return 1j

    end_time = 100
    sample_num = 20

    probe_t = {'states':(0,1), 'rabi_frequency':4+9j,'detuning':0.3, 'time_dependence': fun}
    probe_s = {'states':(0,1), 'rabi_frequency':4+9j,'detuning':0.3}
    dress = {'states':(1,2), 'rabi_frequency':3+2j,'detuning': .7}
    couple_t = {'states':(2,3), 'rabi_frequency':-9j+1, 'detuning':0.3, 'time_dependence': fun2}
    couple_s = {'states':(2,3), 'rabi_frequency':9+1j, 'detuning':0.3}

    gam = np.zeros((4,4),dtype=np.float64)
    gam[1,0] = 0.5
    gam[2,1] = 0.4
    gam[3,2] = 0.6
    gamma_matrix = gam

    sensor_t = rq.Sensor(4)
    sensor_s = rq.Sensor(4)

    sensor_t.add_couplings(probe_t, dress, couple_t)
    sensor_t.set_gamma_matrix(gamma_matrix)
    sensor_s.add_couplings(probe_s, dress, couple_s)
    sensor_s.set_gamma_matrix(gamma_matrix)

    sols_s = rq.solve_steady_state(sensor_s)
    sol_t = rq.solve_time(sensor_t, end_time, sample_num, rtol=1E-6, atol=1e-6)
    # sol_t_numba = rq.solve_time(sensor_t, end_time, sample_num, rtol=1E-6, atol=1e-6,
    #                             use_nkode = True, complex_numba = True)
    np.testing.assert_allclose(sols_s.rho,sol_t.rho[-1,:],atol=0.005,
            err_msg='Complex time and steady state solutions do not match.  non-compiled.')
    # np.testing.assert_allclose(sols_s.rho,sol_t_numba.rho[-1,:],atol=0.005,
    #       err_msg='Complex time and steady state solutions do not match.  numbakit.')


@pytest.mark.time
@pytest.mark.slow
def test_time_rwa():

    n = 159
    [g, e] = rq.D2_states(5)
    RydbergTargetState = A_QState(n+1, 2, 2.5)  # states labeled n, l, j
    RydbergExcitedState = A_QState(n, 3, 3.5)

    RbSensor_rwa = rq.Cell("Rb85", [g, e, RydbergTargetState, RydbergExcitedState],
                           cell_length=0,
                           gamma_transit=2*np.pi*1)
    RbSensor_norwa = rq.Cell("Rb85", [g, e, RydbergTargetState, RydbergExcitedState],
                             cell_length=0,
                             gamma_transit=2*np.pi*1)

    rf_freq = RbSensor_norwa.atom.arc_atom.getTransitionFrequency(n+1, 2, 2.5, n, 3, 3.5)#*1E-6
    
    field = 0.0006  # V/m

    def rf_fun(omega_0, omega_mod,factor):
        def fun(t):
            return factor*field*np.cos(omega_0*t)*np.sin(omega_mod*t)
        return fun

    rf_norwa = rf_fun(2*np.pi*rf_freq, 2*np.pi*5,2)
    rf_rwa = rf_fun(0, 2*np.pi*5,1)

    red_laser = {'states':(g,e), 'rabi_frequency':2*np.pi*0.6, 'detuning':0}
    blue_laser = {'states':(e,RydbergTargetState), 'rabi_frequency':2*np.pi*1.0, 'detuning':0}
    RFrwa = {'states': (RydbergTargetState,RydbergExcitedState), 'rabi_frequency':2*np.pi*0.6, "detuning":0, 'time_dependence': rf_rwa}
    RFno = {'states': (RydbergTargetState,RydbergExcitedState), 'rabi_frequency':2*np.pi*0.6, 'time_dependence': rf_norwa}

    RbSensor_rwa.add_couplings(red_laser, blue_laser,RFrwa)
    RbSensor_norwa.add_couplings(red_laser, blue_laser,RFno)

    RbSensor_rwa.add_transit_broadening(RbSensor_rwa.gamma_transit)
    RbSensor_norwa.add_transit_broadening(RbSensor_norwa.gamma_transit)

    endTime = 5
    sampleNum = 1000000

    time_sol_rwa = rq.solve_time(RbSensor_rwa, endTime, sampleNum, rtol=1E-6)
    time_sol_norwa = rq.solve_time(RbSensor_norwa, endTime, sampleNum, rtol=1E-6)

    transNO = np.imag(time_sol_norwa.get_susceptibility())
    transRWA = np.imag(time_sol_rwa.get_susceptibility())
    np.testing.assert_allclose(transNO,transRWA,rtol=0.01,
            err_msg='Non-RWA and RWA give different results when they should match')
    
    
@pytest.mark.time
def test_time_scan_rabi():
    step = lambda t: 0 if t<1 else 1
    rabi = np.linspace(-5,5,11)
    f1 = {"states":(0,1), "detuning":1, "rabi_frequency":1}
    f2_s = {"states": (1,2), "detuning":2, "rabi_frequency":rabi}
    f2_t = {"states": (1,2), "detuning":2, "rabi_frequency":rabi, "time_dependence":step}
    
    s_steady = rq.Sensor(3)
    s_time = rq.Sensor(3)
    s_steady.add_couplings(f1, f2_s)
    s_time.add_couplings(f1, f2_t)
    s_steady.add_transit_broadening(0.5)
    s_time.add_transit_broadening(0.5)
    
    sol_steady = rq.solve_steady_state(s_steady).rho
    sol_time = rq.solve_time(s_time, 500, 10, rtol=1e-3).rho
    
    np.testing.assert_allclose(sol_steady, sol_time[...,-1,:], atol=0.005)
