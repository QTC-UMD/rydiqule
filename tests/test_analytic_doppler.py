import numpy as np
import rydiqule as rq
import pytest


@pytest.mark.exception
def test_analytic_exceptions():
    """Test input validation of solve_doppler_analytic"""

    atom = 'Rb87'

    states = [
        rq.ground_state(atom),
        rq.D2_excited(atom),
        rq.A_QState(41, 2, 5/2),
    ]

    cell = rq.Cell(atom, states)

    with pytest.raises(rq.RydiquleError, match='at least 1'):

        red = {'states': (states[0],states[1]),
               'detuning': 1, 'rabi_frequency': 2}
        blue = {'states': (states[1],states[2]),
                'detuning': 0, 'rabi_frequency': 4}

        cell.add_couplings(red, blue)

        rq.solve_doppler_analytic(cell)

    with pytest.raises(rq.RydiquleError, match='no doppler shifts'):
        kunit_r = np.array([1, 0, 0])
        kunit_b = np.array([-1, 0, 0])

        red = {'states': (states[0],states[1]),
               'detuning': 1, 'rabi_frequency': 2, 'kunit': kunit_r}
        blue = {'states': (states[1],states[2]),
                'detuning': 0, 'rabi_frequency': 4, 'kunit': kunit_b}

        cell.add_couplings(red, blue)

        rq.solve_doppler_analytic(cell, analytic_axis=1)

    with pytest.raises(rq.RydiquleError, match='no doppler shifts'):
        kunit_r = np.array([1, 0, 0])
        kunit_b = np.array([0, 0, -1])

        red = {'states': (states[0],states[1]),
               'detuning': 1, 'rabi_frequency': 2, 'kunit': kunit_r}
        blue = {'states': (states[1],states[2]),
                'detuning': 0, 'rabi_frequency': 4, 'kunit': kunit_b}

        cell.add_couplings(red, blue)

        rq.solve_doppler_analytic(cell, analytic_axis=1)

@pytest.mark.steady_state
@pytest.mark.doppler
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

    sol_exact = rq.solve_doppler_analytic(cell)

    np.testing.assert_allclose(sol_exact.rho, sol_riemann.rho,
                               rtol=1e-5, atol=7e-5,
                               err_msg='Sampled and analytic 1D doppler do not match')
    

@pytest.mark.steady_state
@pytest.mark.doppler
def test_analytic_2D_doppler():
    """Test that 2D analytic doppler matches direct sampling regardless of analytic axis"""

    atom = 'Rb85'

    states = [
        rq.ground_state(atom),
        rq.A_QState(5, 1, 1/2),
        rq.A_QState(6, 0, 1/2),
        rq.A_QState(31, 1, 1/2)
    ]

    cell = rq.Cell(atom,states)

    detunings = 2*np.pi*np.linspace(-10,10,201)
    Omega_p = 2*np.pi*2
    Omega_d = 2*np.pi*10
    Omega_R = 2*np.pi*1

    theta = 4.526
    phi = 2.556

    kunit1 = np.array([-1,0,0])
    kunit2 = np.array([-1*np.cos(theta),-1*np.sin(theta),0])
    kunit3 = np.array([-1*np.cos(phi),-1*np.sin(phi),0])

    probe = {'states': (states[0],states[1]), 'detuning': 0, 'rabi_frequency': Omega_p, 'kunit': kunit1}
    dressing = {'states': (states[1],states[2]), 'detuning': 0, 'rabi_frequency': Omega_d, 'kunit': kunit2}
    Rydberg = {'states': (states[2],states[3]), 'detuning': detunings, 'rabi_frequency': Omega_R, 'kunit': kunit3}

    cell.add_couplings(probe, dressing, Rydberg)

    # Lower cell temperature to narrow the velocity distribution, allowing coarser mesh.
    # This reduces memory requirements and computation time for unit testing.
    cell.vP = 20

    m = {"method":"split", "n_coherent":151, "n_doppler":101, "width_doppler":2.5, "width_coherent":0.2}
    sol_hyb_0 = rq.solve_doppler_analytic(cell, analytic_axis=0, doppler_mesh_method=m)
    sol_hyb_1 = rq.solve_doppler_analytic(cell, analytic_axis=1, doppler_mesh_method=m)
    sol_riemann = rq.solve_steady_state(cell, doppler=True,doppler_mesh_method=m)

    np.testing.assert_allclose(sol_hyb_0.rho, sol_riemann.rho,
                               rtol=1e-3, atol=7e-3,
                               err_msg='Sampled and hybrid with analytic axis 0 do not match')
    
    np.testing.assert_allclose(sol_hyb_1.rho, sol_riemann.rho,
                               rtol=1e-3, atol=7e-3,
                               err_msg='Sampled and hybrid with analytic axis 1 do not match')
    
    np.testing.assert_allclose(sol_hyb_0.rho, sol_hyb_1.rho,
                               rtol=1e-3, atol=7e-3,
                               err_msg='Hybrid wtih analytic axis 0 and hybrid with analytic axis 1 do not match')