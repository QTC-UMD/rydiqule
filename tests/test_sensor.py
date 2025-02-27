import numpy as np
import rydiqule as rq
import pytest
import warnings


@pytest.mark.structure
def test_dephasings():
    """Tests that decoherence matrix is made correctly, with higher dimensions."""

    b = 4
    g10 = 1
    g21 = np.array([0.1, 0.2, 0.5, 1.0])
    g22 = 5
    g32 = 0.01
    g13 = 0.7
    gt = [4, 6, 8]

    gam_exp = np.zeros((len(gt), len(g21), b,b),dtype=float)
    gam_exp[...,1,0] = g10
    gam_exp[...,2,2] = g22
    gam_exp[...,3,2] = g32
    gam_exp[...,1,3] = g13
    gam_exp[...,2,1] = g21[np.newaxis,:]
    gam_exp[...,0] += np.array(gt)[:,np.newaxis,np.newaxis]*0.6
    gam_exp[...,1] += np.array(gt)[:,np.newaxis,np.newaxis]*0.4

    s = rq.Sensor(b)
    s.add_decoherence((1,0), g10)
    s.add_decoherence((2,1), g21)
    s.add_self_broadening(2, g22)
    s.add_decoherence((3,2), g32)
    s.add_decoherence((1,3), g13)
    s.add_transit_broadening(gt, {0:0.6, 1:0.4})
    s_gam = s.decoherence_matrix()

    np.testing.assert_allclose(s_gam, gam_exp, err_msg='Sensor decoherence matrix incorrect')


@pytest.mark.exception
def test_coupling_exceptions():
    '''Confirms the bad input fields to add_coupling raise correct exceptions.
    '''

    s = rq.Sensor(2)
    td = lambda t:1

    with pytest.raises(rq.RydiquleError):
        # specify both detuning and transition frequency
        f1 = {'states':(0,1), 'detuning':0, 'transition_frequency':1}
        s.add_coupling(**f1)

    with pytest.warns(rq.RWAWarning):
        # checks for warning about likely accidental RWA violation
        f2 = {'states':(0,1), 'transition_frequency': 5001, "time_dependence":td}
        s.add_coupling(**f2)

    # RWA violation should not be shown
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('ignore', rq.RWAWarning)
        f3 = {'states':(0,1), 'transition_frequency': 5001, "time_dependence":td}
        s.add_coupling(**f3)

        assert len(record) == 0, f'RWA warning not supressed, got {record[0]}'

    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        f4 = {'detuning':0, 'rabi_frequency':20}
        s.add_couplings(f4)

    with pytest.raises(rq.RydiquleError, match='non-zero kvector'):
        f5 = {'states':(0,1), 'detuning':0}
        s.add_couplings(f5)
        s.get_doppler_shifts()

    with pytest.raises(rq.RydiquleError, match='Coherent coupling must'):
        f6 = {'states': (0,0), 'detuning':0, 'rabi_frequency':1}
        s.add_couplings(f6)

    with pytest.raises(rq.RydiquleError, match="frame phase offset"):
        f7 = {'states':(0,1), 'rabi_frequency':1, 'phase':0, 'transition_frequency':10}
        s.add_couplings(f7)

    with pytest.warns(rq.RydiquleWarning):
        # checks that improper kvec definitions raise a helpful warning
        f8 = {'states':(0,1), 'rabi_frequency':1, 'detuning':0, 'kvec':(240*2*np.pi/780e-3, 0, 0)}
        s.add_couplings(f8)

@pytest.mark.exception
def test_states_validation():
    '''Assures that bad state specifications a caught.'''

    s = rq.Sensor(2)

    with pytest.raises(rq.RydiquleError, match='cannot be interpreted as a tuple'):
        # bad input type
        s._states_valid(1.0)

    with pytest.raises(rq.RydiquleError, match='must couple exactly 2 states'):
        # too many labels passed
        s._states_valid((0,1,2))

    with pytest.raises(rq.RydiquleError, match='not a state in the basis'):
        # outside basis
        s._states_valid((0,2))


@pytest.mark.exception
def test_basis_exception():
    '''Assures that invalid bases are correctly caught'''
    #bad state type 
    with pytest.raises(rq.RydiquleError, match="not a valid state"):
        s = rq.Sensor(['g', 'e', None])
    
    #double state labels
    with pytest.raises(rq.RydiquleError, match="All state labels must be unique"):
        s = rq.Sensor(['g', 'e', 'e'])

    #bad basis type
    with pytest.raises(rq.RydiquleError, match="a list of states or an integer defining their range"):
        l = lambda x: x
        s = rq.Sensor(l)


@pytest.mark.exception
def test_zip_param_validation():
    '''Assures that incorrect usage of Sensor.zip_parameters is caught.'''

    s = rq.Sensor(3)

    f1 = {'states':(0,1), 'detuning': [1,2,3], 'rabi_frequency':10, 'label':'f1'}
    f2 = {'states':(1,2), 'detuning': [1,2,3], 'rabi_frequency':[10,11,12, 13], 'label':'f2'}
    s.add_couplings(f1,f2)

    with pytest.raises(rq.RydiquleError, match='provide at least 2'):
        s.zip_parameters({'f1':'detuning'})

    with pytest.raises(rq.RydiquleError, match='not a label of any coupling'):
        s.zip_parameters({'f1':'detuning', 'f3':'detuning'})

    with pytest.raises(rq.RydiquleError, match='Got length'):
        s.zip_parameters({'f1':'detuning', 'f2':'rabi_frequency'})

@pytest.mark.exception
def test_gamma_matrix_validation():
    '''Confirms that validation checks trigger for bad gamma matrices.'''

    s = rq.Sensor(2)

    with pytest.raises(rq.RydiquleError, match='must be a numpy array'):
        gam = [[0,1], [1,0]]
        s.set_gamma_matrix(gam)

    with pytest.raises(rq.RydiquleError, match='gamma_matrix has shape'):
        gam = np.array([[0,1,2],[1,0,2],[2,1,0]])
        s.set_gamma_matrix(gam)

@pytest.mark.structure
def test_coupling_group_2_pair():
    ''' Tests that adding couplings individually and as a group in a simple case work identically'''
    #base sensor with couplings added manually
    s_base = rq.Sensor(['a1', 'a2', 'b1', 'b2'])
    s_base.add_coupling(('a1', 'b1'), detuning=1, rabi_frequency=0.75)
    s_base.add_coupling(('a1', 'b2'), detuning=1, rabi_frequency=0.2)
    s_base.add_coupling(('a2', 'b1'), detuning=1, rabi_frequency=0.6)
    s_base.add_coupling(('a2', 'b2'), detuning=1, rabi_frequency=0.4)

    #test sensor with couplings added with coupling_group function
    s_group = rq.Sensor(['a1', 'a2', 'b1', 'b2'])
    cg = {('a1','b1'):0.75, ('a1','b2'):0.2, ('a2', 'b1'):0.6, ('a2', 'b2'):0.4}
    s_group.add_coupling((['a1','a2'], ['b1','b2']), rabi_frequency=1, detuning=1, coupling_coefficients=cg, label='test')

    #test sensor with couplings added with add_couplings
    s_group2 = rq.Sensor(['a1', 'a2', 'b1', 'b2'])
    cg = {('a1','b1'):0.75, ('a1','b2'):0.2, ('a2', 'b1'):0.6, ('a2', 'b2'):0.4}
    c = dict(states=(['a1','a2'], ['b1','b2']), rabi_frequency=1, detuning=1, coupling_coefficients=cg, label='test')
    s_group2.add_couplings(c)

    ham_base = s_base.get_hamiltonian()
    ham_group = s_group.get_hamiltonian()
    ham_group2 = s_group2.get_hamiltonian()

    np.testing.assert_allclose(ham_base, ham_group)
    np.testing.assert_allclose(ham_base, ham_group2)

@pytest.mark.structure
def test_coupling_group_scan():
    '''tests that coupling group hamiltonians are shaped appropriately'''
    #base sensor with couplings added manually
    s_base = rq.Sensor(['g', 'b1', 'b2'])
    det = np.linspace(-1,1,5)
    rabi = np.linspace(-2,2,7)
    s_base.add_coupling(("g", 'b1'), detuning=det, rabi_frequency=0.75*rabi)
    s_base.add_coupling(('g', 'b2'), detuning=det, rabi_frequency=0.2*rabi)

    s_base.zip_parameters({('g','b1'):"rabi_frequency", ('g','b2'):"rabi_frequency"}, zip_label="test_rabi_frequency")
    s_base.zip_parameters({('g','b1'):"detuning", ('g','b2'):"detuning"}, zip_label="test_detuning")

    #test sensor with couplings added with coupling_group function
    s_group = rq.Sensor(['g', 'b1', 'b2'])
    cg = {('g','b1'):0.75, ('g','b2'):0.2}
    s_group.add_coupling(('g', ['b1','b2']), rabi_frequency=rabi, detuning=det, coupling_coefficients=cg, label='test')

    ham_base = s_base.get_hamiltonian()
    ham_group = s_group.get_hamiltonian()

    np.testing.assert_allclose(ham_base, ham_group)

@pytest.mark.structure
def test_coupling_group_overwrite():
    '''Tests that overwritting a coupling group succeeds as expected'''

    s = rq.Sensor(['g', 'b1', 'b2'])
    det = np.linspace(-1,1,3)
    cg = {('g','b1'):1, ('g','b2'):0.5}
    s.add_coupling(('g',['b1','b2']), rabi_frequency=1, detuning=det, coupling_coefficients=cg, label='test')
    hams1 = s.get_hamiltonian()
    s.add_coupling(('g',['b1','b2']), rabi_frequency=-1, detuning=det, coupling_coefficients=cg, label='test')
    hams2 = s.get_hamiltonian()
    hams_diff = hams1-hams2

    # Ensures that rabi frequencies of hamiltonians update correctly
    np.testing.assert_equal(hams_diff[0,0], [0, 1.0, 0.5])

@pytest.mark.structure
def test_dephasing_group():
    #create expected gamma
    gamma_exp = np.zeros((3,3,3))
    gamma_exp[:,1,0]=0.025
    gamma_exp[:,2,0]=0.075
    gamma_exp*=np.array([1,2,3]).reshape((3,1,1))

    s = rq.Sensor(['g','e1','e2'])
    gamma = np.linspace(0.1, 0.3, 3)
    cg = {('e1','g'):0.25, ('e2','g'):0.75}
    s.add_decoherence((['e1','e2'],'g'), gamma, label="test", coupling_coefficients=cg)

    assert np.allclose(gamma_exp, s.decoherence_matrix())
