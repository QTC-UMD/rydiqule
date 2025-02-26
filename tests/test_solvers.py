import numpy as np
import rydiqule as rq
from rydiqule.sensor_utils import get_rho_ij
import pytest


@pytest.mark.steady_state
def test_linewidths():
    """Confirms that doppler-free linewidths are correct
    and that dephasing broadcasting is working.
    """

    gammas = np.array([1.0, 4.28, 9.17, 12.0])
    s = rq.Sensor(2)

    detunings = np.linspace(-10,10,201)

    probe = {'states':(0,1), 'rabi_frequency': 0.01, 'detuning':2*np.pi*detunings}

    s.add_decoherence((1,0), 2*np.pi*gammas)
    s.add_couplings(probe)

    sols = rq.solve_steady_state(s).rho
    isols = rq.get_rho_ij(sols, 1, 0).imag

    sol_ind = s.axis_labels().index('(0,1)_detuning')
    meas_gammas = np.abs(detunings[np.abs(isols - np.expand_dims(isols.max(axis=sol_ind),sol_ind)/2).argmin(axis=sol_ind)])*2
    # use relatively loose abstol since width finding is so crude
    np.testing.assert_allclose(meas_gammas, gammas, rtol=0, atol=0.1,
                               err_msg='2-level absorption linewidths do not match')


@pytest.mark.steady_state
@pytest.mark.util
def test_ham_slicing():
    """Confirms that breaking up the system during solves norminally works"""

    s = rq.Sensor(3)
    s.add_decoherence((1,0), 2*np.pi*6.0666)
    s.add_decoherence((2,1), 2*np.pi*10e-3)
    s.add_transit_broadening(2*np.pi*100e-3)

    detunings = np.linspace(-10,10,41)
    probe = {'states':(0,1), 'rabi_frequency':2*np.pi*0.1, 'detuning':2*np.pi*detunings, 'label':'probe'}
    couple = {'states':(1,2), 'rabi_frequency':2*np.pi*1, 'detuning':0, 'label':'couple'}

    s.add_couplings(probe,couple)

    # if slicing is broken, error will occur here
    sols = rq.solve_steady_state(s,n_slices=4)
    disp = sols.rho_ij(1,0).real

    # do a couple simple checks to roughly confirm output is dispersive
    zero_cross_ind = np.where(disp == 0)[0].squeeze()
    zero_cross = detunings[zero_cross_ind]

    assert zero_cross == pytest.approx(0.0), 'Resonance not resonant!'
    assert disp.max() == pytest.approx(-disp.min()), 'Resonance not symmetric!'


@pytest.mark.steady_state
@pytest.mark.parametrize("rf_rabi", np.linspace(1.0, 20.0, 5))  # in MHz
def test_DopFree_AT(find_zero_crossings, rf_rabi):
    """Confirms that the expected Autler-Townes splitting is generated
    in steady state without Doppler averaging or optical depth.
    """

    basis_size = 4
    s = rq.Sensor(basis_size)
    transit = 100e-3
    gamma = 5.75
    ryd_lifetime = 1e-3
    gam = np.zeros((s.basis_size,s.basis_size),dtype=np.float64)
    gam[1,0] = gamma
    gam[2,1] = ryd_lifetime
    gam[3,1] = ryd_lifetime
    gam[:,0] += transit
    s.set_gamma_matrix(2*np.pi*gam)

    detunings = np.linspace(-15, 15, 301)
    probe = {'states':(0,1), 'rabi_frequency':2*np.pi*0.1, 'detuning':0}
    coupling = {'states':(1,2), 'rabi_frequency': 2*np.pi*0.6, 'detuning':2*np.pi*detunings}
    rf = {'states':(2,3), 'rabi_frequency': 2*np.pi*rf_rabi, 'detuning':0}

    s.add_couplings(probe,coupling,rf)
    sols = rq.solve_steady_state(s).rho
    rsols = get_rho_ij(sols, 1,0).real

    # detect expected zero crossings in phase shift vs detuning
    # this will give us the AT splitting
    det_crossings = find_zero_crossings(rsols,detunings)

    # ensure we got the right number of crossings
    assert len(det_crossings) == 2

    AT_splitting = det_crossings[-1] - det_crossings[0]

    # tolerances set conservatively at 1% relative and 0.1 MHz absolute
    assert AT_splitting == pytest.approx(rf_rabi, rel=1e-2, abs=1e-1), \
        f'Expected AT={rf_rabi:.2f}, got {AT_splitting:.2f}'


@pytest.mark.steady_state
@pytest.mark.doppler
@pytest.mark.parametrize("rf_rabi", np.linspace(8.0, 20.0, 5))  # in MHz
def test_Dop_AT_couple(find_zero_crossings, rf_rabi):
    """Confirms that the expected Autler-Townes splitting is generated
    in steady state with Doppler averaging. For a coupling sweep, AT = rf_rabi
    """

    basis_size = 4
    vP = 242.387  # Rb85
    s = rq.Sensor(basis_size, vP=vP)
    transit = 100e-3
    gamma = 6.0666
    ryd_lifetime = 1e-3
    gam = np.zeros((s.basis_size,s.basis_size),dtype=np.float64)
    gam[1,0] = gamma
    gam[2,1] = ryd_lifetime
    gam[3,1] = ryd_lifetime
    gam[:,0] += transit
    s.set_gamma_matrix(2*np.pi*gam)

    # values correspond to Rb D2 scheme
    kpmag = 2*np.pi/780.241e-3  # sets end units correctly to Mrad/s
    kcmag = 2*np.pi/480e-3
    kp = kpmag*np.array([1,0,0])
    kc = kcmag*np.array([-1,0,0])

    detunings = np.linspace(-15, 15, 301)
    probe = {'states':(0,1), 'rabi_frequency':2*np.pi*1.5, 'detuning':0, 'kvec':kp}
    coupling = {'states':(1,2), 'rabi_frequency': 2*np.pi*4.0, 'detuning':2*np.pi*detunings,
                'kvec':kc}
    rf = {'states':(2,3), 'rabi_frequency': 2*np.pi*rf_rabi, 'detuning':0}

    s.add_couplings(probe,coupling,rf)
    sols = rq.solve_steady_state(s,doppler=True).rho
    rsols = get_rho_ij(sols, 1,0).real

    # detect expected zero crossings in phase shift vs detuning
    # this will give us the AT splitting
    det_crossings = find_zero_crossings(rsols,detunings)

    # ensure splitting is actually symmetric
    assert det_crossings[-1] == pytest.approx(-det_crossings[0],rel=1e-2), \
        'Assymetric AT splitting detected'

    AT_splitting = det_crossings[-1] - det_crossings[0]

    # tolerances set conservatively at 1% relative and 0.1 MHz absolute
    assert AT_splitting == pytest.approx(rf_rabi, rel=1e-2, abs=1e-1), \
        f'Expected AT={rf_rabi:.2f}, got {AT_splitting:.2f}'


@pytest.mark.steady_state
@pytest.mark.doppler
@pytest.mark.parametrize("rf_rabi", np.linspace(8.0, 20.0, 5))  # in MHz
def test_Dop_AT_probe(find_zero_crossings, rf_rabi):
    """Confirms that the expected Autler-Townes splitting is generated
    in steady state with Doppler averaging. For a coupling sweep, AT = kp/kc*rf_rabi
    """

    basis_size = 4
    vP = 242.387
    s = rq.Sensor(basis_size, vP=vP)
    transit = 100e-3
    gamma = 6.0666
    ryd_lifetime = 1e-3
    gam = np.zeros((s.basis_size,s.basis_size),dtype=np.float64)
    gam[1,0] = gamma
    gam[2,1] = ryd_lifetime
    gam[3,1] = ryd_lifetime
    gam[:,0] += transit
    s.set_gamma_matrix(2*np.pi*gam)

    # values correspond to Rb D2 scheme
    kpmag = 2*np.pi/780.241e-3  # sets end units correctly to Mrad/s
    kcmag = 2*np.pi/480e-3
    kp = kpmag*np.array([1,0,0])
    kc = kcmag*np.array([-1,0,0])

    detunings = np.linspace(-15, 15, 301)
    probe = {'states':(0,1), 'rabi_frequency':2*np.pi*1.5,'detuning':2*np.pi*detunings, 'kvec':kp}
    coupling = {'states':(1,2), 'rabi_frequency': 2*np.pi*4.0, 'detuning':0, 'kvec':kc}
    rf = {'states':(2,3), 'rabi_frequency': 2*np.pi*rf_rabi, 'detuning':0}

    s.add_couplings(probe,coupling,rf)
    sols = rq.solve_steady_state(s, doppler=True).rho
    rsols = get_rho_ij(sols, 1,0).real

    # detect expected zero crossings in phase shift vs detuning
    # this will give us the AT splitting
    det_crossings = find_zero_crossings(rsols,detunings)

    # ensure splitting is actually symmetric
    assert det_crossings[-1] == pytest.approx(-det_crossings[0],rel=1e-2), \
        'Assymetric AT splitting detected'

    AT_splitting = det_crossings[-1] - det_crossings[0]

    # tolerances set conservatively at 1% relative and 0.25 MHz absolute
    expAT = kpmag/kcmag*rf_rabi
    assert AT_splitting == pytest.approx(expAT, rel=1e-2, abs=2.5e-1), \
        f'Expected AT={expAT:.2f}, got {AT_splitting:.2f}'


@pytest.mark.steady_state
@pytest.mark.doppler
def test_Dop_2level():
    """Checks that simple doppler broadening of the probing transition
    is correctly calculated by the doppler averaging.
    """

    from scipy.optimize import curve_fit
    from scipy.constants import k,c
    from arc import Rubidium85

    # get actual atom parameters to compare against
    atom = Rubidium85()
    gState = [5,0,0.5]
    iState = [5,1,1.5]
    f0 = atom.getTransitionFrequency(*gState,*iState)
    gamma = atom.getTransitionRate(*iState,*gState)/2/np.pi*1e-6
    mRb85 = atom.mass
    T = 300

    # define out 2-level system
    basis_size = 2
    vP = np.sqrt(2*k*T/mRb85)
    s = rq.Sensor(basis_size, vP=vP)
    gam = np.zeros((s.basis_size,s.basis_size),dtype=np.float64)
    gam[1,0] = gamma
    s.set_gamma_matrix(2*np.pi*gam)

    kpmag = 2*np.pi/780.241e-3  # sets end units correctly to Mrad/s
    kp = kpmag*np.array([1,0,0])

    # calculate the theoretical doppler width for this transition, in MHz
    actualWidth = vP*np.sqrt(4*np.log(2)/c**2)*f0*1e-6

    # calculate the doppler absorption profile
    detunings = np.linspace(-750, 750, 301)
    probe = {'states':(0,1), 'rabi_frequency':2*np.pi*1.5,'detuning':2*np.pi*detunings,'kvec':kp}

    s.add_couplings(probe)
    sols = rq.solve_steady_state(s,doppler=True,
                                 doppler_mesh_method={'method':'uniform','wDop':2.5}).rho

    def gaussFit(f,A,w):
        return A*np.exp(-f**2/(2*w**2))

    rho10imag = get_rho_ij(sols, 1,0).imag
    maxval_init = rho10imag.max()
    p0 = [maxval_init,200]
    popt,pcov = curve_fit(gaussFit,detunings,rho10imag,p0=p0)

    measuredWidth = popt[1]*2*np.sqrt(2*np.log(2))

    # confirm measured width is within 2% of theorectical
    assert measuredWidth == pytest.approx(actualWidth,rel=2e-2), \
        'Doppler width of Rb85 D2 transition not correct'


@pytest.mark.exception
def test_solve_memory_fits():
    '''Confirms that checks for EOMs fitting in memory fail when appropriate.'''

    with pytest.raises(rq.RydiquleError, match='sum_doppler=False'):
        # We do not support sum_doppler=False when full solve does not fit in memory
        n = 4
        stack = (101, 101)
        doppler_stack = (561, 561)
        rq.slicing.slicing.get_slice_num(n, stack, doppler_stack, sum_doppler=False, weight_doppler=True)

    with pytest.raises(rq.RydiquleError, match='System is too large'):
        # this system requires 335.4 GiB
        n = 4
        stack = (101, 101)
        doppler_stack = (561, 561, 561)
        rq.slicing.slicing.get_slice_num(n, stack, doppler_stack, sum_doppler=True, weight_doppler=True)

@pytest.mark.exception
def test_init_cond_valid():
    '''Confirms the timesolver throws and error if init_cond are not physical'''

    s = rq.Sensor(3)
    s.add_coupling((0,1), detuning=0, rabi_frequency=1)
    s.add_coupling((1,2), detuning=0, rabi_frequency=1, time_dependence=lambda t:1)

    with pytest.raises(rq.RydiquleError, match='not positive semi-definite'):
        rq.solve_time(s, 10, 100, init_cond=np.array([1,0,0,0,0,0,0,0]))
