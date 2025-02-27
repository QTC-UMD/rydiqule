import numpy as np
import rydiqule as rq
import pytest

@pytest.mark.steady_state
@pytest.mark.doppler
@pytest.mark.dev
def test_analytic_1D_doppler():
    """Test that 1D analytic doppler matches direct sampling"""

    atom = 'Rb87'

    states = [
        rq.ground_state(atom),
        rq.D2_excited(atom),
        rq.A_QState(41, 2, 5/2),
        rq.A_QState(40, 3, 7/2)
    ]

    cell = rq.Cell(atom, states)

    detunings = 2*np.pi*np.linspace(-30, 30, 201)
    Omega_r = 2*np.pi*2
    Omega_b = 2*np.pi*5
    Omega_rf = 2*np.pi*np.array([0, 5, 40])

    kunit_r = np.array([1, 0, 0])
    kunit_b = np.array([-1, 0, 0])

    red = {'states': (states[0],states[1]), 'detuning': detunings, 'rabi_frequency': Omega_r, 'kunit': kunit_r}
    blue = {'states': (states[1],states[2]), 'detuning': 0, 'rabi_frequency': Omega_b, 'kunit': kunit_b}
    rf = {'states': (states[2],states[3]), 'detuning': 0, 'rabi_frequency': Omega_rf}

    cell.add_couplings(red, blue, rf)

    sol_riemann = rq.solve_steady_state(cell, doppler=True,
                                        doppler_mesh_method={'method': 'split',
                                                             'width_coherent':0.28,
                                                             'n_coherent': 1001})

    sol_exact = rq.doppler_1d_exact(cell)

    np.testing.assert_allclose(sol_exact.rho, sol_riemann.rho,
                               rtol=1e-5, atol=7e-5,
                               err_msg='Sampled and analytic 1D doppler do not match')
