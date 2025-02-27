import numpy as np
import rydiqule as rq
import pytest
import warnings
from scipy.constants import c, hbar, e, epsilon_0

import scipy.constants
a0 = scipy.constants.physical_constants["Bohr radius"][0]


@pytest.mark.experiments
def test_OD_with_Steck():

    cell_length = .000001 #keep small to stay in the thin limit.
    beam_waist = 1
    beam_area = np.pi*beam_waist**2
    [D1g, D1e] = rq.D1_states('Rb85')

    rb_cell =  rq.Cell('Rb85', [D1g, D1e],
                       cell_length = cell_length, beam_area = beam_area)

    laserfreq = c/(795e-9)
    Joule_per_photon = hbar*2*np.pi*laserfreq
    beam_power = 1e-5 #watts

    photons_per_sec = beam_power/(Joule_per_photon)
    PhotonFlux = 2*beam_power/(np.pi*beam_waist**2*Joule_per_photon)
    gamma = rb_cell.decoherence_matrix()[1,0]

    laser_01 = {"states": (D1g, D1e), "detuning": 0, "beam_power": beam_power, "beam_waist": beam_waist }
    rb_cell.add_couplings(laser_01)

    omega = rb_cell.get_couplings()[D1g, D1e]['rabi_frequency']
    s = 2*(omega/(gamma))**2
    steck_scattering_rate = 1e6*(cell_length*rb_cell.density*gamma/2)*(s/(1+s))
    steck_OD_thin = steck_scattering_rate/PhotonFlux


    sol = rq.solve_steady_state(rb_cell)
    rq_OD = sol.get_OD()
    
    np.testing.assert_allclose(steck_OD_thin, rq_OD, rtol=0.025)


@pytest.mark.experiments
def test_susceptibility_with_steck():
    cell_length = .00001 #keep small to stay in the thin limit.
    beam_waist = .001
    beam_area = np.pi*beam_waist**2

    atom = "Rb85"
    states = rq.D1_states(atom)
    rb_cell = rq.Cell(atom, states, cell_length=cell_length, beam_area=beam_area)
    e_field=1e-7

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', rq.RWAWarning)
        rb_cell.add_coupling((states[0], states[1]), e_field=e_field,detuning = 5)
    sol = rq.solve_steady_state(rb_cell)
    rho_eg = sol.rho_ij(1,0)

    d = 1*rb_cell.couplings.edges[(states[0],states[1])]["dipole_moment"]*e*a0
    N = rb_cell.density

    sus_rq = sol.get_susceptibility()
    sus_steck = N*d*rho_eg/(e_field/2)/epsilon_0    
    
    np.testing.assert_allclose(sus_steck, sus_rq)


@pytest.mark.experiments
def test_phase_shift_with_steck():
    #these values are copied from the susceptibility test (above)
    #this test uses a sensor instead of a cell to make sure things can be passed
    
    cell_length = .00001 #keep small to stay in the thin limit.
    beam_waist = .001
    beam_area = np.pi*beam_waist**2
    kappa = 14176543411.043068
    probe_rabi = 1.3853997106115247e-8
    probe_freq = 2369435838304474.5
    probe_wavelength = c*2*np.pi/probe_freq
    probe_wavevector = 2*np.pi/probe_wavelength
    e_field=1e-7


    rb_cell = rq.Sensor(2)
    rb_cell.set_experiment_values(cell_length=cell_length, probe_freq = probe_freq,
                                  beam_area=beam_area, kappa = kappa)
    rb_cell.set_gamma_matrix(np.array([[0.23229296, 0.00000000e+00],
                                       [36.15760044, 0.00000000e+00]]))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', rq.RWAWarning)
        rb_cell.add_coupling((0,1),detuning = 5, rabi_frequency=probe_rabi)
    sol = rq.solve_steady_state(rb_cell)
    rho_eg = sol.rho_ij(1,0)

    d = 1.461003490986055e-29
    N = 1.5692729232906132e+16


    phase = sol.get_phase_shift()
    sus_steck = N*d*rho_eg/(e_field/2)/epsilon_0
    phase_steck = (1+sus_steck.real/2)*probe_wavevector*cell_length

    np.testing.assert_allclose(phase, phase_steck)
    np.testing.assert_allclose(sus_steck, sol.get_susceptibility())

