import numpy as np
import pytest

import rydiqule as rq


@pytest.mark.time
@pytest.mark.slow
def test_nbkit():
    """Tests that time solver functions with Doppler broadening
    """
    red_laser = {'states':(0,1), 'rabi_frequency':2*np.pi*0.6, 'detuning':0}

    RydbergTargetState = [150, 2, 2.5, 0.5]  # states labeled n, l, j, m_j
    RydbergExcitedState = [149, 3, 3.5, 0.5]

    RbSensor_time = rq.Cell("Rb85", *rq.D2_states(5),
                            RydbergTargetState, RydbergExcitedState,
                            cell_length=0,
                            gamma_transit=2*np.pi*.1)

    def rf_carrier(t):
        return np.sin(2*np.pi*10.0*t)

    sampleNum = 20
    endTime = 2

    blue_laser = {'states':(1,2), 'rabi_frequency':6.0, 'detuning': 0}
    rf = {'states':(2,3), 'rabi_frequency':1.0, 'detuning': 0, 'time_dependence': rf_carrier}
    RbSensor_time.add_couplings(blue_laser, red_laser, rf)

    sols_solve_ivp = rq.solve_time(RbSensor_time, endTime, sampleNum, solver='scipy', rtol=1e-6, atol=1e-6)
    sols_nbkode = rq.solve_time(RbSensor_time, endTime, sampleNum,  solver='nbkode', rtol=1e-6, atol=1e-6)

    np.testing.assert_allclose(sols_solve_ivp.rho[-1,:], sols_nbkode.rho[-1,:], atol=0.005, err_msg='The difference between two solutions is too big')


@pytest.mark.time
@pytest.mark.slow
def test_time_complex_match_steady_numbakit():
    """Tests the time solver vs steady state
    with aribtrary (complex) and time-dependent rabi frequencies
    using the numbakitode backend
    """

    def fun(t):
        if t < 1:
            return 0
        else:
            return 1

    def fun2(t):
        if t < 1:
            return 0j
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
    sol_t_numba = rq.solve_time(sensor_t, end_time, sample_num, rtol=1E-6, atol=1e-6,
                                solver='nbkode')
    np.testing.assert_allclose(sols_s.rho,sol_t_numba.rho[-1,:],atol=0.005,
           err_msg='Complex time and steady state solutions do not match using numbakit.')

# @pytest.mark.time
# @pytest.mark.doppler
# def test_numba_with_doppler():
#     """Tests that time solver functions with Doppler broadening
#     """
#     rf_rabi = 25 #Mrad/s
#     red_laser = {'states':(0,1), 'rabi_frequency':2*np.pi*0.6}  #fields are stored as dictioniaries
#     blue_laser = {'states':(1,2), 'rabi_frequency':2*np.pi*2.0, 'detuning': 0}
#
#     RbSensor_ss = rq.Cell(gamma_transit=2*np.pi*1, gamma_exc=2*np.pi*6, gamma_Ryd = 2*np.pi*.1)
#     RbSensor_time = rq.Cell(gamma_transit=2*np.pi*1, gamma_exc=2*np.pi*6, gamma_Ryd = 2*np.pi*.1)
#
#     RydbergTargetState = [350, 2, 2.5, 0.5]  #states labeled n, l, j, m_j
#     RydbergExcitedState = [349, 3, 3.5, 0.5]
#
#     RbSensor_ss.add_states(RydbergTargetState,RydbergExcitedState)
#     RbSensor_time.add_states(RydbergTargetState,RydbergExcitedState)
#
#     rf_freq = RbSensor_time.atom.getTransitionFrequency(*RydbergTargetState[:3],*RydbergExcitedState[:3])*1E-6
#
#     RbSensor_ss.add_couplings({'states':(2,3), "dipole_moment":"auto", "transition_frequency":"auto"})
#     rf_dipole_matrix = RbSensor_ss.rf_dipole_matrix()
#
#     rf_dipole_moment = rf_dipole_matrix[2,3]
#     field = 2*rf_rabi/rf_dipole_moment
#
#     def sig_and_LO(omega_0, delta, beta):
#         def fun(t):
#             return field*np.sin(omega_0*t)+field*beta*np.sin((omega_0+delta)*t)
#         return fun
#
#     sampleNum = 20
#     endTime = 1 # microseconds
#     rf = sig_and_LO(2*np.pi*rf_freq, 5, .1)
#
#     kpmag = 2*np.pi/780.241e-3 # sets end units correctly to Mrad/s
#     kcmag = 2*np.pi/480e-3
#     kp = kpmag*np.array([1,0,0])
#     kc = kcmag*np.array([-1,0,0])
#     vP = 242.387
#     wPp = kp*vP
#     wPc = kc*vP
#
#     d_meth = {"method":"uniform", "n_uniform":50}
#
#     red_laser = {'states':(0,1), 'rabi_frequency':2*np.pi*1.0, 'detuning':0, 'kvec':wPp}
#     blue_laser = {'states':(1,2), 'rabi_frequency':2*np.pi*6.0, 'detuning': 0, 'kvec':wPc}
#
#     RbSensor_time.add_fields(blue_laser, red_laser)
#     RbSensor_time.add_fields({"states":(2,3), "dipole_moment":"auto", "transition_frequency":"auto"})
#
#     time, time_sol_ivp = rq.solve_time(RbSensor_time, endTime, sampleNum, rf_input=rf, doppler=True, doppler_mesh_method=d_meth, use_nkode=False)
#     time, time_sol_nbkode = rq.solve_time(RbSensor_time, endTime, sampleNum, rf_input=rf, doppler=True, doppler_mesh_method=d_meth, use_nkode=True)
#
#     np.testing.assert_allclose(time_sol_ivp, time_sol_nbkode, rtol=0.1, err_msg='The difference between two solutions is too big')
