import numpy as np
from arc import Rubidium85
import rydiqule as rq
import pytest


@pytest.mark.exception
def test_cell_coupling_exceptions(Rb85_sensor_kwargs):
    '''Test that Cell-specific checks for couplings fail appropriately.'''

    c = rq.Cell('Rb85', *rq.D2_states(50), **Rb85_sensor_kwargs)

    with pytest.raises(ValueError, match='only define one'):
        f1 = {'states':(0,1), 'rabi_frequency':10, 'e_field':5}
        c.add_coupling(**f1)

    with pytest.raises(ValueError, match='only define one'):
        f1 = {'states':(0,1), 'rabi_frequency':10, 'beam_power':5}
        c.add_coupling(**f1)

    with pytest.raises(ValueError, match='only define one'):
        f1 = {'states':(0,1), 'e_field':5, 'beam_waist':1.5}
        c.add_coupling(**f1)

    with pytest.raises(ValueError, match='only define one'):
        f1 = {'states':(0,1), 'beam_power':5}
        c.add_coupling(**f1)

    with pytest.raises(ValueError, match='scanned simultaneously'):
        f1 = {'states':(0,1), 'beam_power':[1e-6, 2e-6], 'beam_waist':[100e-6,200e-6]}
        c.add_coupling(**f1)

    with pytest.raises(ValueError, match='Cell does not support explicit'):
        f2 = {'states':(0,1), 'rabi_frequency':2, 'transition_frequency':5}
        c.add_coupling(**f2)
        
    with pytest.raises(ValueError, match="States must either be of length 4 or 5."):
        c2 = rq.Cell('Rb85', *rq.D2_states('Rb85'), [1,2,3,4,5,6], cell_length = .0001)


@pytest.mark.structure
def test_dipole_scaling(Rb85_sensor_kwargs):
    """Tests that fields specified by dipole moment are scaled correctly."""

    rt = [50,2,2.5,0.5]
    re = [49,3,3.5,0.5]
    c = rq.Cell('Rb85', *rq.D2_states(50), rt, re, **Rb85_sensor_kwargs)

    def func(t):
        return 1
    c.add_couplings({'states':(2,3), 'e_field':1, 'detuning':0,'time_dependence':func})

    # diff between c.rf_dipole_matrix is scale_factor
    base, dipole_mat, dipole_mat_i = c.get_time_hamiltonians()
    rf_dipole_moment = dipole_mat[0][2,3]
    desired_rabi = 5  # Mrad/s
    # rf_dipole_moment is defined to have units of (Mrad/s)/(V/m)
    field = desired_rabi/rf_dipole_moment

    atom = Rubidium85()
    arc_rabi = 0.5*atom.getRabiFrequency2(*rt, *re[:3], 0, field)*1e-6  # Mrad/s

    assert desired_rabi == pytest.approx(arc_rabi), 'Internal dipole moments not scaled correctly'


@pytest.mark.structure
def test_decoherences(Rb85_sensor_kwargs):
    """Confirms that the decoherence matrix is building correctly.
    """
    gState = [5, 0, 0.5, 0.5]
    iState = [5, 1, 1.5, 0.5]
    rState = [50, 2, 2.5, 0.5]
    rcState = [51, 1, 1.5, 0.5]

    atom = Rubidium85()

    # all rates are in MHz
    gamma = 1/atom.getStateLifetime(*iState[0:3])*1e-6
    rState_lifetime = 1/atom.getStateLifetime(*rState[0:3])*1e-6
    rcState_lifetime = 1/atom.getStateLifetime(*rcState[0:3])*1e-6

    riDecay = atom.getTransitionRate(*rState[:3], *iState[:3])*1e-6
    rrcDecay = atom.getTransitionRate(*rState[:3], *rcState[:3])*1e-6
    rcgDecay = atom.getTransitionRate(*rcState[:3], *gState[:3])*1e-6
    transit = 2*np.pi*65.5286e-3*0

    # calculates all dipole allowed dephasings
    # any dephasing not accounted for from the natural lifetime
    # is assumed to decay to the ground state
    gamma_expected = np.zeros((4, 4))
    gamma_expected[1, 0] = gamma
    gamma_expected[2, 1] = riDecay
    gamma_expected[2, 3] = rrcDecay
    gamma_expected[3, 0] = rcgDecay

    gamma_expected[2, 0] += rState_lifetime - riDecay - rrcDecay
    gamma_expected[3, 0] += rcState_lifetime - rcgDecay

    gamma_expected[:, 0] += transit

    cell = rq.Cell('Rb85', gState, iState,
                   rState, rcState, **Rb85_sensor_kwargs)
    cell.add_transit_broadening(transit)

    np.testing.assert_allclose(cell.decoherence_matrix(), gamma_expected,
                               atol=2*np.pi*1e-4, rtol=0,
                               err_msg='gamma matrix for 2-photon;1 RF not equal')

@pytest.mark.exception
def test_warns_dipole(Rb85_sensor_kwargs):
    """Test that cell raises an error when a non dipole-allowed transition is added"""
    c = rq.Cell('Rb85', *rq.D2_states(5), [50,2,2.5,0.5], **Rb85_sensor_kwargs)
    
    with pytest.warns(UserWarning, match='not electric-dipole allowed'):
        f1 = {'states':(0,2), 'rabi_frequency':1, 'detuning':0}
        c.add_coupling(**f1)


@pytest.mark.structure
def test_hyperfine_dipole():
    extra_states = [[6, 2, 2.5, 4, 0], [7, 1, 1.5, 4, 1]]
    c_hyp = rq.Cell('Rb85',*rq.D2_states('Rb85'), *extra_states, cell_length = 0.0001)
    c_hyp.add_coupling((0,1), rabi_frequency=1, detuning=1)
    c_hyp.add_coupling((1,2), rabi_frequency=1, detuning=1)
    c_hyp.add_coupling((3,2), rabi_frequency=1, detuning=1, q=-1)
    
    atom = Rubidium85()
    states = c_hyp.states_list()
    
    dipole_01 = atom.getDipoleMatrixElement(*states[0], *states[1], 0)
    assert dipole_01 == c_hyp.couplings.edges[0,1]["dipole_moment"]
    
    dipole_12 = -1 * atom.getDipoleMatrixElementHFStoFS(*states[2], *states[1], 0)
    assert dipole_12 == c_hyp.couplings.edges[1,2]["dipole_moment"]
    
    dipole_32 = atom.getDipoleMatrixElementHFS(*states[3], *states[2], -1)
    assert dipole_32 == c_hyp.couplings.edges[3,2]["dipole_moment"]
