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

    gam_exp = np.zeros((len(g21), len(gt), b,b),dtype=float)
    gam_exp[...,1,0] = g10
    gam_exp[...,2,2] = g22
    gam_exp[...,3,2] = g32
    gam_exp[...,1,3] = g13
    gam_exp[...,2,1] = g21[:,np.newaxis]
    gam_exp[...,0] += np.array(gt)[np.newaxis,:,np.newaxis]*0.6
    gam_exp[...,1] += np.array(gt)[np.newaxis,:,np.newaxis]*0.4

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

    with pytest.raises(ValueError):
        # specify both detuning and transition frequency
        f1 = {'states':(0,1), 'detuning':0, 'transition_frequency':1}
        s.add_coupling(**f1)

    with pytest.warns(UserWarning):
        # checks for warning about likely accidental RWA violation
        f2 = {'states':(0,1), 'transition_frequency': 5001}
        s.add_coupling(**f2)

    # RWA violation should not be shown
    with warnings.catch_warnings(record=True) as record:
        f3 = {'states':(0,1), 'transition_frequency': 5001, 'suppress_rwa_warn':True}
        s.add_coupling(**f3)

        assert len(record) == 0, 'RWA warning not supressed'

    with pytest.raises(ValueError, match="\'states\' parameter must be specified for any field"):
        f4 = {'detuning':0, 'rabi_frequency':20}
        s.add_couplings(f4)

    with pytest.raises(ValueError, match='non-zero kvector'):
        f5 = {'states':(0,1), 'detuning':0}
        s.add_couplings(f5)
        s.get_doppler_shifts()

    with pytest.raises(ValueError):
        f6 = {'states': (0,0), 'detuning':0, 'rabi_frequency':1}
        s.add_couplings(f6)


@pytest.mark.exception
def test_states_validation():
    '''Assures that bad state specifications a caught.'''

    s = rq.Sensor(2)

    with pytest.raises(ValueError, match='cannot be interpreted as a tuple'):
        # bad input type
        s._states_valid(1.0)

    with pytest.raises(ValueError, match='must couple exactly 2 states'):
        # too many labels passed
        s._states_valid((0,1,2))

    with pytest.raises(ValueError, match='not a state in the basis'):
        # outside basis
        s._states_valid((0,2))


@pytest.mark.exception
def test_basis_exception():
    '''Assures that invalid bases are correctly caught'''
    #bad state type 
    with pytest.raises(ValueError, match="must be integers or strings"):
        s = rq.Sensor(['g', 'e', None])
    
    #double state labels
    with pytest.raises(ValueError, match="All state labels must be unique"):
        s = rq.Sensor(['g', 'e', 'e'])

    #bad basis type
    with pytest.raises(TypeError, match="must be an integer or iterable"):
        l = lambda x: x
        s = rq.Sensor(l)


@pytest.mark.exception
def test_zip_param_validation():
    '''Assures that incorrect usage of Sensor.zip_parameters is caught.'''

    s = rq.Sensor(3)

    f1 = {'states':(0,1), 'detuning': [1,2,3], 'rabi_frequency':10, 'label':'f1'}
    f2 = {'states':(1,2), 'detuning': [1,2,3], 'rabi_frequency':[10,11,12, 13], 'label':'f2'}
    s.add_couplings(f1,f2)

    with pytest.raises(ValueError, match='provide at least 2'):
        s.zip_parameters('f1_detuning')

    with pytest.raises(ValueError, match='not a label of any coupling'):
        s.zip_parameters('f1_detuning', 'f3_detuning')

    with pytest.raises(ValueError, match='Got length'):
        s.zip_parameters('f1_detuning', 'f2_rabi_frequency')

@pytest.mark.exception
def test_gamma_matrix_validation():
    '''Confirms that validation checks trigger for bad gamma matrices.'''

    s = rq.Sensor(2)

    with pytest.raises(TypeError, match='must be a numpy array'):
        gam = [[0,1], [1,0]]
        s.set_gamma_matrix(gam)

    with pytest.raises(ValueError, match='gamma_matrix has shape'):
        gam = np.array([[0,1,2],[1,0,2],[2,1,0]])
        s.set_gamma_matrix(gam)
