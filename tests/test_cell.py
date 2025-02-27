import itertools
import numpy as np
from arc import Rubidium85
import rydiqule as rq
import pytest

from rydiqule.atom_utils import A_QState


@pytest.mark.exception
def test_cell_coupling_exceptions(Rb85_sensor_kwargs):
    '''Test that Cell-specific checks for couplings fail appropriately.'''

    c = rq.Cell('Rb85', rq.D2_states(50), **Rb85_sensor_kwargs)

    [g,e] = rq.D2_states(50)

    with pytest.raises(rq.RydiquleError, match='only define one'):
        f1 = {'states':(g,e), 'rabi_frequency':10, 'e_field':5}
        c.add_coupling(**f1)

    with pytest.raises(rq.RydiquleError, match='only define one'):
        f1 = {'states':(g,e), 'rabi_frequency':10, 'beam_power':5}
        c.add_coupling(**f1)

    with pytest.raises(rq.RydiquleError, match='only define one'):
        f1 = {'states':(g,e), 'e_field':5, 'beam_waist':1.5}
        c.add_coupling(**f1)

    with pytest.raises(rq.RydiquleError, match='only define one'):
        f1 = {'states':(g,e), 'beam_power':5}
        c.add_coupling(**f1)

    with pytest.raises(rq.RydiquleError, match='scanned simultaneously'):
        f1 = {'states':(g,e), 'beam_power':[1e-6, 2e-6], 'beam_waist':[100e-6,200e-6]}
        c.add_coupling(**f1)

    with pytest.raises(rq.RydiquleError, match='Cell does not support explicit'):
        f2 = {'states':(g,e), 'rabi_frequency':2, 'transition_frequency':5}
        c.add_coupling(**f2)

    with pytest.raises(rq.RydiquleError, match='unit propagation axis'):
        f3 = {'states':(g,e), 'rabi_frequency':2, 'detuning':5, 'kvec':(1,0,0)}
        c.add_coupling(**f3)

    with pytest.raises(rq.RydiquleError, match='|kunit|'):
        f4 = {'states':(g,e), 'rabi_frequency':2, 'detuning':5, 'kunit':(10,0,0)}
        c.add_coupling(**f4)


@pytest.mark.structure
def test_dipole_scaling(Rb85_sensor_kwargs):
    """Tests that fields specified by dipole moment are scaled correctly."""

    [g,e1] = rq.D2_states(5)
    rt = A_QState(50,2,2.5)
    re = A_QState(49,3,3.5)


    c = rq.Cell('Rb85', [g, e1, rt, re], **Rb85_sensor_kwargs)

    def func(t):
        return 1
    c.add_couplings({'states':(rt, re), 'e_field':1, 'detuning':0,'time_dependence':func})

    # diff between c.rf_dipole_matrix is scale_factor
    dipole_mat, dipole_mat_i = c.get_time_hamiltonian_components()
    rf_dipole_moment = dipole_mat[0][2,3]
    desired_rabi = 5  # Mrad/s
    # rf_dipole_moment is defined to have units of (Mrad/s)/(V/m)
    field = desired_rabi/rf_dipole_moment

    atom = Rubidium85()
    arc_rabi = 0.0
    # average over allowed dipole transitions between degenerate sublevels
    # note that the dipole moment magnitude is the same for mj and -mj here
    for mj in [0.5, 1.5, 2.5]:
        arc_rabi += 0.5*atom.getRabiFrequency2(*rt[:3], mj, *re[:3], 0, field)*1e-6  # Mrad/s
    arc_rabi /= 3

    assert desired_rabi == pytest.approx(arc_rabi), 'Internal dipole moments not scaled correctly'


@pytest.mark.structure
def test_decoherences(Rb85_sensor_kwargs):
    """Confirms that the decoherence matrix is building correctly.
    """
    gState = A_QState(5, 0, 0.5)
    iState = A_QState(5, 1, 1.5)
    rState = A_QState(50, 2, 2.5)
    rcState = A_QState(51, 1, 1.5)

    atom = Rubidium85()

    # all rates are in MHz
    gamma = 1/atom.getStateLifetime(*iState[:3])*1e-6
    rState_lifetime = 1/atom.getStateLifetime(*rState[:3])*1e-6
    rcState_lifetime = 1/atom.getStateLifetime(*rcState[:3])*1e-6

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

    cell = rq.Cell('Rb85', [gState, iState, rState, rcState], 
                   **Rb85_sensor_kwargs)
    cell.add_transit_broadening(transit)

    np.testing.assert_allclose(cell.decoherence_matrix(), gamma_expected,
                               atol=2*np.pi*1e-4, rtol=0,
                               err_msg='gamma matrix for 2-photon;1 RF not equal')


@pytest.mark.structure
def test_hyperfine_dipole():
    [g, e1] = [A_QState(5, 0, 0.5, m_j=0.5), A_QState(5, 1, 1.5, m_j=0.5)]
    [e2, e3] = [A_QState(6, 2, 2.5, f=4, m_f=0), A_QState(7, 1, 1.5, f=4, m_f=1)]
    c_hyp = rq.Cell('Rb85',[g, e1, e2, e3], cell_length = 0.0001)
    c_hyp.add_coupling((g,e1), rabi_frequency=1, detuning=1)
    c_hyp.add_coupling((e1,e2), rabi_frequency=1, detuning=1)
    c_hyp.add_coupling((e3,e2), rabi_frequency=1, detuning=1, q=-1)
    
    atom = Rubidium85()
    states = c_hyp.states
    
    dipole_01 = atom.getDipoleMatrixElement(*g[:4], *e1[:4], 0)
    assert dipole_01 == c_hyp.couplings.edges[g,e1]["dipole_moment"]
    
    dipole_12 = atom.getDipoleMatrixElementHFStoFS(*e2[:3], *e2[4:], *e1[:4], 0)
    assert dipole_12 == c_hyp.couplings.edges[e1,e2]["dipole_moment"]
    
    dipole_32 = atom.getDipoleMatrixElementHFS(*e3[:3], *e3[4:], *e2[:3], *e2[4:], -1)
    assert dipole_32 == c_hyp.couplings.edges[e3,e2]["dipole_moment"]


@pytest.mark.structure
def test_rabi_scaling():
    "Confirms generated hamiltonian scales from spherical moment correctly"
    atom = rq.RQ_AlkaliAtom(Rubidium85())
    r1 = A_QState(50, 2, 1.5, m_j='all')
    r2 = A_QState(51, 1, 0.5, m_j='all')
    r1_exp = rq.expand_qnums([r1], atom.arc_atom.I)
    r2_exp = rq.expand_qnums([r2], atom.arc_atom.I)

    FS = {}
    for (s1, s2) in itertools.product(r1_exp, r2_exp):
        for q in [-1, 0, 1]:
            elem = atom.get_spherical_dipole_matrix_element(s1, s2, q)
            if not np.isclose(elem, 0.0):
                FS[(s1,s2)] = elem

    assert (max(FS.values()) / min(FS.values()))**2 == pytest.approx(3.0), 'Spherical ME mis-match'

    c = rq.Cell('Rb85', [r1, r2])
    rabi = 2.3
    rf = {'states':(r2, r1), 'rabi_frequency':rabi, 'detuning':0, 'label':'rf', 'q':1}
    c.add_couplings(rf)
    ham = c.get_hamiltonian()
    # note that Cell uses sph*2 for the coherent_cc, but divides by 2 for the RWA, so factors cancel
    assert len(r1_exp) == np.count_nonzero(ham), 'Incorrect number of couplings generated'
    assert ham.max() == pytest.approx(rabi*max(FS.values())), 'Max Rabi scaling incorrect'
    assert ham[ham>0].min() == pytest.approx(rabi*min(FS.values())), 'Min Rabi scaling incorrect'
    
@pytest.mark.structure
def test_rabi_consistency():
    "Confirms alternative specification of Coupling strength are consistent"

    g = rq.A_QState(5, 0, 0.5, f=3, m_f='all')
    i = rq.A_QState(5, 1, 1.5, f=4, m_f='all')
    r = rq.A_QState(50, 2, 2.5, m_j='all')
    r2 = rq.A_QState(51, 1, 1.5, m_j='all')

    c = rq.Cell('Rb85', [g,i,r,r2])

    # generate alternative coupling strength definitions
    atom = rq.RQ_AlkaliAtom(Rubidium85())
    probe_power = 1e-6 # W
    probe_waist = 100e-6 # m
    probe_q = 1
    couple_power = 50e-3 # W
    couple_waist = 110e-6 # m
    couple_q = -1
    rf_field = 0.15 # V/m
    rf_waist = 10e-2 # m
    rf_q = 0

    probe_field = atom.gaussian_center_field(probe_power, probe_waist)
    couple_field = atom.gaussian_center_field(couple_power, couple_waist)

    rf_field_conv = atom.gaussian_center_field(1e-6, rf_waist)/np.sqrt(1e-6)
    rf_power = (rf_field/rf_field_conv)**2

    probe_red_rabi = atom.get_reduced_rabi_frequency(g, i, probe_power, probe_waist)*1e-6/2/np.pi
    couple_red_rabi = atom.get_reduced_rabi_frequency(i, r, couple_power, couple_waist)*1e-6/2/np.pi
    rf_red_rabi = atom.get_reduced_rabi_frequency2(r2, r, rf_field)*1e-6/2/np.pi

    # set couplings with Rabi frequencies
    p_rabi = {'states':(g,i), 'rabi_frequency':2*np.pi*probe_red_rabi, 'detuning':0, 'q':probe_q, 'label':'probe'}
    c_rabi = {'states':(i,r), 'rabi_frequency':2*np.pi*couple_red_rabi, 'detuning':0, 'q':couple_q, 'label':'couple'}
    r_rabi = {'states':(r2,r), 'rabi_frequency':2*np.pi*rf_red_rabi, 'detuning':0, 'q':rf_q, 'label':'rf'}

    c.add_couplings(p_rabi, c_rabi, r_rabi)
    ham_rabi = c.get_hamiltonian()

    # set couplings with fields
    p_field = {'states':(g,i), 'e_field':probe_field, 'detuning':0, 'q':probe_q, 'label':'probe'}
    c_field = {'states':(i,r), 'e_field':couple_field, 'detuning':0, 'q':couple_q, 'label':'couple'}
    r_field = {'states':(r2,r), 'e_field':rf_field, 'detuning':0, 'q':rf_q, 'label':'rf'}

    c.add_couplings(p_field, c_field, r_field)
    ham_field = c.get_hamiltonian()

    # set couplings with beam powers and waists
    p_power = {'states':(g,i), 'beam_power':probe_power, 'beam_waist':probe_waist, 'detuning':0, 'q':probe_q, 'label':'probe'}
    c_power = {'states':(i,r), 'beam_power':couple_power, 'beam_waist':couple_waist, 'detuning':0, 'q':couple_q, 'label':'couple'}
    r_power = {'states':(r2,r), 'beam_power':rf_power, 'beam_waist':rf_waist, 'detuning':0, 'q':rf_q, 'label':'rf'}

    c.add_couplings(p_power, c_power, r_power)
    ham_power = c.get_hamiltonian()

    # confirm all generated hamiltonians match
    np.testing.assert_allclose(ham_field, ham_power,
                           err_msg='Field and Power specifications do not match')
    np.testing.assert_allclose(ham_rabi, ham_field,
                          err_msg='Rabi and Field specifications do not match')
