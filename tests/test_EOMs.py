import numpy as np
import pytest
from rydiqule.sensor_utils import generate_eom
from rydiqule.solvers import steady_state_solve_stack


@pytest.mark.util
def test_2level_EOM():
    """Tests that EOMs are generated correctly using a simple two-level
    system with a far detuned coupling field.
    """

    omega1 = 1
    natLifetime = 0.01
    delta1 = 10

    # expected EOM arrays
    # basis is rho01_real, rho01_imag, rho11_real
    rhs = -np.array([0,-omega1/2,0])
    lhs = -np.array([[natLifetime/2,delta1,0],
                    [-delta1,natLifetime/2,omega1],
                    [0,-omega1,natLifetime]])

    ham = np.array([[0,omega1/2],[omega1/2,delta1]])
    decoh = np.zeros((2,2),dtype=np.double)
    decoh[1,0] = natLifetime

    eom, constant_term = generate_eom(ham, decoh)

    np.testing.assert_allclose(rhs, constant_term,
                               err_msg='Constant terms do not match for 2-level EOMs')
    np.testing.assert_allclose(lhs, eom, err_msg='EOM terms do not match for 2-level EOMs')


@pytest.mark.util
def test_simple_Ham():
    """Tests that EOMs are generated correctly using a simple three-level
    system with far detuned fields.
    """

    omega1 = 0.55555+1.3333j
    decayRate = .01
    delta1 = 10
    delta2 = 10
    omega2 = .5

    rho_01_pred = -omega1.real/delta1
    rho_10_pred = -omega1.imag/delta1
    # rho_21_pred = -omega2/delta1
    # rho_12_pred = -omega2/delta2
    # these predictions come from analytically solving the master eqn.
    # see simple mathematica notebook that can be supplied by KCC

    ham = np.array([[0,np.conj(omega1),0],[omega1,delta1,omega2],[0,omega2,delta2]])
    decoh = np.array([[decayRate,0,0],[decayRate,0,0],[decayRate,0,0]])
    # redrabi = omega1
    # redphase = 0
    eom, constant_term = generate_eom(ham, decoh, real_eom=True)
    rho = steady_state_solve_stack(eom, constant_term)
    np.testing.assert_allclose([rho_01_pred,rho_10_pred], [rho[0],rho[2]],atol=0.1,
                               err_msg='did not correctly solve large-detuning test Hamiltonian')
    # note: I do not currently know what the basis is after make_real.
    # But the code asserts that Re(rho_01) is rho[0], so start with that one.
