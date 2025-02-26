import pytest

import rydiqule as rq
from rydiqule.atom_utils import A_QState, validate_qnums, expand_qnums


@pytest.mark.exception
def test_state_validation():
    """Confirms that invalid combinations of quantum numbers are not allowed"""

    with pytest.raises(ValueError, match="not a valid combination"):
        validate_qnums(A_QState(5, 1, 1.5, f=2), I=1.5)
    with pytest.raises(ValueError, match="not a valid combination"):
        validate_qnums(A_QState(5, 1, 1.5, f=2), I=1.5)
    with pytest.raises(ValueError, match="not a valid combination"):
        validate_qnums(A_QState(5, 1, 1.5, m_j=0.5, f=2), I=1.5)

@pytest.mark.exception
def test_qnum_validation():
    """Confirms that Invalid quantum numbers are not allowed"""

    with pytest.raises(AssertionError, match="invalid n quantum number"):
        validate_qnums(A_QState(5.5, 1.5, 1.5), I=1.5)
    
    with pytest.raises(AssertionError, match="invalid l quantum number"):
        validate_qnums(A_QState(5, 1.5, 1.5), I=1.5)

    with pytest.raises(AssertionError, match="invalid j quantum number"):
        validate_qnums(A_QState(5, 1, 1), I=1.5)

    with pytest.raises(AssertionError, match="m_j must"):
        validate_qnums(A_QState(5, 1, 1.5, m_j=2.5), I=1.5)
    
    with pytest.raises(AssertionError, match="f must"):
        validate_qnums(A_QState(5, 1, 1.5, f=5, m_f=0), I=1.5)

    with pytest.raises(AssertionError, match="m_f must"):
        validate_qnums(A_QState(5, 1, 1.5, f=2, m_f=1.5), I=1.5)

@pytest.mark.structure
def test_match_AQState():
    """Tests that the correct sublist is returned by match_A_QState"""
    
    state_full = A_QState(5, 1, 1.5, f="all", m_f="all")
    state_f1 = A_QState(5, 1, 1.5, f=1, m_f="all")

    full_list = expand_qnums([state_full], I=3/2)
    f1_list = expand_qnums([state_f1], I=3/2)

    assert(rq.sensor_utils.match_states(state_f1, compare_list=full_list) == f1_list)