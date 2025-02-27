import pytest

import rydiqule as rq
from rydiqule.atom_utils import A_QState
from arc import Rubidium85


import numpy as np

@pytest.mark.exception
def test_RQ_AlkaliAtom_Exceptions_Dipole():
    """Confirms that input validations work for RQ_AlkaliAtom functions"""

    atom = rq.RQ_AlkaliAtom(Rubidium85())

    NLJ_state = A_QState(5, 0, 0.5)
    FS_state = A_QState(6, 1, 1.5, m_j=0.5)
    HFS_state = A_QState(7, 1, 1.5, f=2, m_f=1) 

    #NLJ -> FS
    with pytest.raises(rq.AtomError, match='Invalid transition type'):
        atom.get_dipole_matrix_element(NLJ_state, FS_state, 0)
    with pytest.raises(rq.AtomError, match='Transition between NLJ and FS'):
        atom.get_transition_rate(NLJ_state, FS_state, 0)

    #NLJ -> HFS
    with pytest.raises(rq.AtomError, match='Invalid transition type'):
        atom.get_dipole_matrix_element(NLJ_state, HFS_state, 0)
    with pytest.raises(rq.AtomError, match='Transition between NLJ and HFS'):
        atom.get_transition_rate(NLJ_state, HFS_state, 0)

@pytest.mark.structure
def test_average_dipole_moment():
    """Confirms averaged dipole moment between NLJ states is consistent"""

    atom = rq.RQ_AlkaliAtom(Rubidium85())

    expected_ame = 0.011090367580708254

    s1 = A_QState(5, 1, 1.5)
    s2 = A_QState(50, 2, 2.5)
    ame = atom.get_dipole_matrix_element(s1, s2, 0)

    assert ame == pytest.approx(expected_ame), 'Average Matrix element between NLJ states changed'


@pytest.mark.structure
def test_dephasing_normalization():
    """Confirms dephasing transition rates are normalized"""

    arc_atom = Rubidium85()
    atom = rq.RQ_AlkaliAtom(arc_atom)
    I_qnum = arc_atom.I


    fs = rq.A_QState(50, 2, 2.5, m_j="all")
    fs_nlj = rq.A_QState(50, 2, 2.5)
    fs2 = rq.A_QState(52, 1, 1.5, m_j="all")
    fs2_nlj = rq.A_QState(52, 1, 1.5)
    hfs = rq.A_QState(5, 1, 1.5, f="all", m_f="all")
    hfs_nlj = rq.A_QState(5, 1, 1.5)


    fs_full = rq.expand_qnums([fs], I=I_qnum)
    fs2_full = rq.expand_qnums([fs2], I=I_qnum)
    hfs_full = rq.expand_qnums([hfs], I=I_qnum)

    actual = arc_atom.getTransitionRate(*fs[:3], *hfs[:3])

    nlj = atom.get_transition_rate(fs_nlj, hfs_nlj)

    assert nlj == pytest.approx(actual), 'NLJ transition rate not correct'

    # confirm that each mJ sublevel of fs sums to the nlj total rate

    FStoHFS = {}
    for fs_state in fs_full:
        mj=fs_state[3]
        FStoHFS[mj] = 0.0
        for hfs_state in hfs_full:
            FStoHFS[mj] += atom.get_transition_rate(fs_state, hfs_state)

    for mj, rate in FStoHFS.items():
        assert rate == pytest.approx(nlj), f'FStoHFS total transition rate for {mj:.1f} is not correct'

    #equivalent test for an pure fs transition
    nlj2 = arc_atom.getTransitionRate(*fs2[:3], *fs[:3])
    FS = {}
    for fs2_state in fs2_full:
        mj2=fs2_state[3]
        FS[mj2] = 0.0
        for fs_state in fs_full:
            FS[mj2] += atom.get_transition_rate(fs2_state, fs_state)

    for mj, rate in FS.items():
        assert rate == pytest.approx(nlj2), f'FStoHFS total transition rate for {mj:.1f} is not correct'