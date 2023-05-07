import numpy as np
import rydiqule as rq
import pytest


@pytest.mark.structure
def test_zip_detunings():
    """
    Tests that zipping parameters works properly for a simple
    variable detuning case where red and blue detuning are scanned
    in parallel for a simple 3-level system.
    """

    # what the correct hamiltonian will be for this system
    expected_ham = np.array([[[0. + 0.j,  0.5+0.j,  0. + 0.j],
                              [0.5-0.j, -2. + 0.j,  1. + 0.j],
                              [0. + 0.j,  1. - 0.j, -2. + 0.j]],

                             [[0. + 0.j,  0.5+0.j,  0. + 0.j],
                              [0.5-0.j, -1. + 0.j,  1. + 0.j],
                              [0. + 0.j,  1. - 0.j,  0. + 0.j]],

                             [[0. + 0.j,  0.5+0.j,  0. + 0.j],
                              [0.5-0.j,  0. + 0.j,  1. + 0.j],
                              [0. + 0.j,  1. - 0.j,  2. + 0.j]],

                             [[0. + 0.j,  0.5+0.j,  0. + 0.j],
                              [0.5-0.j,  1. + 0.j,  1. + 0.j],
                              [0. + 0.j,  1. - 0.j,  4. + 0.j]],

                             [[0. + 0.j,  0.5+0.j,  0. + 0.j],
                              [0.5-0.j,  2. + 0.j,  1. + 0.j],
                              [0. + 0.j,  1. - 0.j,  6. + 0.j]]])

    s = rq.Sensor(3)
    detunings_red = np.linspace(-2, 2, 5)
    detunings_blue = 2 + detunings_red

    red = {"states": (0,1), "detuning": detunings_red, "rabi_frequency": 1, "label": "red"}
    blue = {"states": (1,2), "detuning": detunings_blue, "rabi_frequency": 2, "label": "blue"}

    s.add_couplings(red, blue)
    s.zip_parameters("red_detuning", "blue_detuning")

    test_ham = s.get_hamiltonian()

    np.testing.assert_equal(test_ham, expected_ham)


@pytest.mark.structure
def test_zip_rabi():
    """
    Tests that zipping parameters works properly for a simple
    variable rabi case where red and blue rabi_frequency are scanned
    in parallel for a simple 3-level system.
    """

    expected_ham = np.array([[[0. + 0.j, 0.5+0.j, 0. + 0.j],
                              [0.5-0.j, 0. + 0.j, 1. + 0.j],
                              [0. + 0.j, 1. - 0.j, 0. + 0.j]],

                             [[0. + 0.j, 1. + 0.j, 0. + 0.j],
                              [1. - 0.j, 0. + 0.j, 2. + 0.j],
                              [0. + 0.j, 2. - 0.j, 0. + 0.j]],

                             [[0. + 0.j, 1.5+0.j, 0. + 0.j],
                              [1.5-0.j, 0. + 0.j, 3. + 0.j],
                              [0. + 0.j, 3. - 0.j, 0. + 0.j]],

                             [[0. + 0.j, 2. + 0.j, 0. + 0.j],
                              [2. - 0.j, 0. + 0.j, 4. + 0.j],
                              [0. + 0.j, 4. - 0.j, 0. + 0.j]],

                             [[0. + 0.j, 2.5+0.j, 0. + 0.j],
                              [2.5-0.j, 0. + 0.j, 5. + 0.j],
                              [0. + 0.j, 5. - 0.j, 0. + 0.j]]])

    s = rq.Sensor(3)
    rabi_red = np.linspace(1, 5, 5)
    rabi_blue = 2 * rabi_red

    red = {"states": (0,1), "detuning": 0, "rabi_frequency": rabi_red}
    blue = {"states": (1,2), "detuning": 0, "rabi_frequency": rabi_blue}

    s.add_couplings(red, blue)
    s.zip_parameters("(0,1)_rabi_frequency", "(1,2)_rabi_frequency")

    test_ham = s.get_hamiltonian()
    np.testing.assert_equal(test_ham, expected_ham)


@pytest.mark.structure
def test_zip_phase():
    """
    Tests that zipping parameters works properly for a simple
    variable phase case where red and blue phase are scanned
    in parallel for a simple 3-level system.
    """
    expected_ham = np.array([[[0. + 0.j, 0. - 0.5j, 0. + 0.j],
                              [0. + 0.5j, 0. + 0.j, 0. - 1.j],
                              [0. + 0.j, 0. + 1.j, 0. + 0.j]],

                             [[0. + 0.j, 0.35355-0.35355j, 0. + 0.j],
                              [0.35355+0.35355j, 0. + 0.j, 0.70711-0.70711j],
                              [0. + 0.j, 0.70711+0.70711j, 0. + 0.j]],

                             [[0. + 0.j, 0.5 + 0.j, 0. + 0.j],
                              [0.5 - 0.j, 0. + 0.j, 1. + 0.j],
                              [0. + 0.j, 1. - 0.j, 0. + 0.j]],

                             [[0. + 0.j, 0.35355+0.35355j, 0. + 0.j],
                              [0.35355-0.35355j, 0. + 0.j, 0.70711+0.70711j],
                              [0. + 0.j, 0.70711-0.70711j, 0. + 0.j]],

                             [[0. + 0.j, 0. + 0.5j, 0. + 0.j],
                              [0. - 0.5j, 0. + 0.j, 0. + 1.j],
                              [0. + 0.j, 0. - 1.j, 0. + 0.j]]])

    s = rq.Sensor(3)
    phases = np.pi/2 * np.linspace(-1, 1, 5)

    red = {"states": (0,1), "detuning": 0, "rabi_frequency": 1, "phase": phases}
    blue = {"states": (1,2), "detuning": 0, "rabi_frequency": 2, "phase": phases}

    s.add_couplings(red, blue)
    s.zip_parameters("(0,1)_phase", "(1,2)_phase")

    test_ham = s.get_hamiltonian()
    np.testing.assert_allclose(expected_ham, test_ham, atol=1e-4)
