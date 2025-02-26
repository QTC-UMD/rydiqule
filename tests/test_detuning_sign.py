import numpy as np
import pytest
import rydiqule as rq

from rydiqule.atom_utils import A_QState

@pytest.mark.structure
@pytest.mark.time
def test_sensor_signs_ladder():
    """
    Checks that detuning signs follow convention: positive detuning = blue

    Does this by checking sign of AC Stark shift
    with probe coupling in the rotating frame
    and the dressing coupling not in the rotating frame
    for a ladder system.
    """

    end_time = 10 # us
    sample_num = 10

    d_freq = 20  # MHz
    d_det = 8  # MHz
    d_rabi = 4 # MHz

    # actual calculated magnitude will be less
    # than RWA result due to low carrier frequency violating RWA
    expected_shift_mag = np.abs(0.5*(d_det-np.sqrt(d_det**2+d_rabi**2)))

    def d_field(t):
        return np.cos(2*np.pi*(d_freq + d_det)*t)
    
    dets = np.linspace(-2, 2, 41)

    s = rq.Sensor(3)
    p = {'states':(0,1), 'rabi_frequency':2*np.pi*0.01, 'detuning':2*np.pi*dets}
    d = {'states':(1,2), 'rabi_frequency':2*np.pi*d_rabi,
         'transition_frequency':2*np.pi*d_freq, 'time_dependence':d_field}
    
    s.add_couplings(p,d)
    s.add_decoherence((1,0), 2*np.pi*1)
    s.add_decoherence((2,1), 2*np.pi*1)

    ladder_sols = rq.solve_time(s, end_time, sample_num)
    ladder_rho10i = ladder_sols.rho_ij(1,0).imag[:,-1]
    ladder_peak_freq = dets[np.argmax(ladder_rho10i)]

    assert ladder_peak_freq == pytest.approx(expected_shift_mag, abs=0.1), (
        'Detuning signs or magnitude wrong in ladder Sensor AC Stark shift test'
    )

@pytest.mark.structure
@pytest.mark.time
def test_sensor_signs_lambda():
    """
    Checks that detuning signs follow convention: positive detuning = blue

    Does this by checking sign of AC Stark shift
    with probe coupling in the rotating frame
    and the dressing coupling not in the rotating frame
    for a lambda system.
    """

    end_time = 10 # us
    sample_num = 10

    d_freq = 20  # MHz
    d_det = 8  # MHz
    d_rabi = 4 # MHz

    # actual calculated magnitude will be less
    # than RWA result due to low carrier frequency violating RWA
    expected_shift_mag = np.abs(0.5*(d_det-np.sqrt(d_det**2+d_rabi**2)))

    def d_field(t):
        return np.cos(2*np.pi*(d_freq + d_det)*t)
    
    dets = np.linspace(-2, 2, 41)

    s = rq.Sensor(3)
    p = {'states':(0,1), 'rabi_frequency':2*np.pi*0.01, 'detuning':2*np.pi*dets}
    d = {'states':(2,1), 'rabi_frequency':2*np.pi*d_rabi,
         'transition_frequency':2*np.pi*d_freq, 'time_dependence':d_field}
    
    s.add_couplings(p,d)
    s.add_decoherence((1,0), 2*np.pi*0.5)
    s.add_decoherence((1,2), 2*np.pi*0.5)

    lambda_sols = rq.solve_time(s, end_time, sample_num)
    lambda_rho10i = lambda_sols.rho_ij(1,0).imag[:,-1]
    lambda_peak_freq = dets[np.argmax(lambda_rho10i)]

    assert lambda_peak_freq == pytest.approx(-expected_shift_mag, abs=0.1), (
        'Detuning signs or magnitude wrong in lambda Sensor AC Stark shift test'
    )


@pytest.mark.structure
@pytest.mark.time
def test_sensor_signs_vee():
    """
    Checks that detuning signs follow convention: positive detuning = blue

    Does this by checking sign of AC Stark shift
    with probe coupling in the rotating frame
    and the dressing coupling not in the rotating frame
    for a vee system.
    """

    end_time = 10 # us
    sample_num = 10

    d_freq = 20  # MHz
    d_det = 8  # MHz
    d_rabi = 4 # MHz

    # actual calculated magnitude will be less
    # than RWA result due to low carrier frequency violating RWA
    expected_shift_mag = np.abs(0.5*(d_det-np.sqrt(d_det**2+d_rabi**2)))

    def d_field(t):
        return np.cos(2*np.pi*(d_freq + d_det)*t)
    
    dets = np.linspace(-2, 2, 41)

    s = rq.Sensor(3)
    p = {'states':(0,1), 'rabi_frequency':2*np.pi*0.01, 'detuning':2*np.pi*dets}
    d = {'states':(0,2), 'rabi_frequency':2*np.pi*d_rabi,
         'transition_frequency':2*np.pi*d_freq, 'time_dependence':d_field}
    
    s.add_couplings(p,d)
    s.add_decoherence((1,0), 2*np.pi*1)
    s.add_decoherence((2,0), 2*np.pi*1)

    vee_sols = rq.solve_time(s, end_time, sample_num)
    vee_rho10i = vee_sols.rho_ij(1,0).imag[:,-1]
    vee_peak_freq = dets[np.argmax(vee_rho10i)]

    assert vee_peak_freq == pytest.approx(-expected_shift_mag, abs=0.1), (
        'Detuning signs or magnitude wrong in vee Sensor AC Stark shift test'
    )

@pytest.mark.structure
@pytest.mark.time
@pytest.mark.slow
def test_cell_signs_higher():
    """
    Checks that detuning signs follow convention: positive detuning = blue

    Does this by checking sign of AC Stark shift
    with optical couplings in the rotating frame
    and the rf coupling not in the rotating frame.

    Checks sign for rf coupling to higher energy state.
    """

    end_time = 10  # us
    sample_num = 10

    atom = "Rb85"
    [g, e1] = rq.D2_states(atom)
    r_target = A_QState(150, 2, 2.5)
    
    rf_detuning = 8  # MHz
    rf_rabi = 4
    expected_shift_mag = np.abs(0.5*(rf_detuning-np.sqrt(rf_detuning**2+rf_rabi**2)))
    
    dets = np.linspace(-1,1,21)

    # couple to higher energy state
    r_couple = A_QState(149, 3, 3.5)
    c = rq.Cell(atom, [*rq.D2_states(atom), r_target, r_couple],
                cell_length=0,
                gamma_transit=2*np.pi*0.2)
    
    rf_freq = np.abs(c.atom.get_transition_frequency(r_target,r_couple)*1e-6)
    def rf_field(t):
        return np.cos(2*np.pi*(rf_freq+rf_detuning)*t)

    p = {'states':(g,e1), 'rabi_frequency':2*np.pi*0.1, 'detuning':0}
    d = {'states':(e1,r_target), 'rabi_frequency':2*np.pi*1.0, 'detuning':2*np.pi*dets}
    rf = {'states':(r_target, r_couple), 'rabi_frequency':2*np.pi*rf_rabi, 'time_dependence': rf_field}
    c.add_couplings(p,d,rf)

    time_sols_up = rq.solve_time(c, end_time, sample_num)
    up_rho10i = time_sols_up.rho_ij(1,0).imag[:,-1]
    up_peak_freq = dets[np.argmin(up_rho10i)]

    assert up_peak_freq == pytest.approx(expected_shift_mag, abs=0.1), (
        'Detuning signs or magnitude wrong in Cell coupling to higher state test'
    )


@pytest.mark.structure
@pytest.mark.time
@pytest.mark.slow
def test_cell_signs_lower():
    """
    Checks that detuning signs follow convention: positive detuning = blue

    Does this by checking sign of AC Stark shift
    with optical couplings in the rotating frame
    and the rf coupling not in the rotating frame.

    Checks sign for rf coupling to lower energy state.
    """

    end_time = 10  # us
    sample_num = 10

    atom = "Rb85"

    [D1g, D1e] = rq.D2_states(atom)

    r_target = A_QState(150, 2, 2.5)
    # couple to lower energy state
    r_couple = A_QState(151, 1, 1.5)
    
    rf_detuning = 8  # MHz
    rf_rabi = 4
    expected_shift_mag = np.abs(0.5*(rf_detuning-np.sqrt(rf_detuning**2+rf_rabi**2)))
    
    dets = np.linspace(-1,1,21)


    c = rq.Cell(atom, [*rq.D2_states(atom), r_target, r_couple],
                cell_length=0,
                gamma_transit=2*np.pi*0.2)
    
    rf_freq = np.abs(c.atom.get_transition_frequency(r_target,r_couple)*1e-6)
    def rf_field(t):
        return np.cos(2*np.pi*(rf_freq+rf_detuning)*t)

    p = {'states':(D1g,D1e), 'rabi_frequency':2*np.pi*0.1, 'detuning':0}
    d = {'states':(D1e,r_target), 'rabi_frequency':2*np.pi*1.0, 'detuning':2*np.pi*dets}
    rf = {'states':(r_couple,r_target), 'rabi_frequency':2*np.pi*rf_rabi, 'time_dependence': rf_field}
    c.add_couplings(p,d,rf)

    time_sols_down = rq.solve_time(c, end_time, sample_num)
    down_rho10i = time_sols_down.rho_ij(1,0).imag[:,-1]
    down_peak_freq = dets[np.argmin(down_rho10i)]

    assert down_peak_freq == pytest.approx(-expected_shift_mag, abs=0.1), (
        'Detuning signs or magnitude wrong in Cell coupling to lower state test'
    )