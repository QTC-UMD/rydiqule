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
                              [0.5-0.j, 2. + 0.j,  1. + 0.j],
                              [0. + 0.j,  1. - 0.j, 2. + 0.j]],

                             [[0. + 0.j,  0.5+0.j,  0. + 0.j],
                              [0.5-0.j, 1. + 0.j,  1. + 0.j],
                              [0. + 0.j,  1. - 0.j,  0. + 0.j]],

                             [[0. + 0.j,  0.5+0.j,  0. + 0.j],
                              [0.5-0.j,  0. + 0.j,  1. + 0.j],
                              [0. + 0.j,  1. - 0.j,  -2. + 0.j]],

                             [[0. + 0.j,  0.5+0.j,  0. + 0.j],
                              [0.5-0.j,  -1. + 0.j,  1. + 0.j],
                              [0. + 0.j,  1. - 0.j,  -4. + 0.j]],

                             [[0. + 0.j,  0.5+0.j,  0. + 0.j],
                              [0.5-0.j,  -2. + 0.j,  1. + 0.j],
                              [0. + 0.j,  1. - 0.j,  -6. + 0.j]]])

    s = rq.Sensor(3)
    detunings_red = np.linspace(-2, 2, 5)
    detunings_blue = 2 + detunings_red

    red = {"states": (0,1), "detuning": detunings_red, "rabi_frequency": 1, "label": "red"}
    blue = {"states": (1,2), "detuning": detunings_blue, "rabi_frequency": 2, "label": "blue"}

    s.add_couplings(red, blue)
    s.zip_parameters({"red":"detuning", "blue":"detuning"})

    test_ham = s.get_hamiltonian()

    np.testing.assert_equal(test_ham, expected_ham,
                            err_msg='Detuning zip failed')


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
    s.zip_parameters({(0,1):"rabi_frequency", (1,2):"rabi_frequency"})

    test_ham = s.get_hamiltonian()
    np.testing.assert_equal(test_ham, expected_ham,
                            err_msg='Rabi zip failed')


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
    s.zip_parameters({(0,1):"phase", (1,2):"phase"})

    test_ham = s.get_hamiltonian()
    np.testing.assert_allclose(expected_ham, test_ham, atol=1e-4,
                               err_msg='Phase zip failed')


@pytest.mark.structure
def test_unzip():
    s = rq.Sensor(3)
    det = np.linspace(-1, 1, 11)
    rabi = np.linspace(-1, 1, 13)

    s.add_coupling((0,1), detuning=det, rabi_frequency=rabi, label="blue")
    s.add_coupling((1,2), detuning=det, rabi_frequency=rabi, label="red")

    shape1 = s._stack_shape()
    s.zip_parameters({"red":"detuning", "blue":"detuning"})
    shape2 = s._stack_shape()
    s.unzip_parameters("zip_0")
    shape3 = s._stack_shape()

    np.testing.assert_equal(shape1, shape3,
                            err_msg='Unzipping unsuccessful')
    np.testing.assert_raises(AssertionError, np.testing.assert_equal,
                             shape2, shape3,
                             err_msg='Zipping unsuccessful')
    
@pytest.mark.structure
def test_zip():

    s = rq.Sensor(3)
    det = np.linspace(-1, 1, 11)
    rabi = np.linspace(-1, 1, 13)
    gam = np.linspace(-0.1, 0.1, 11)

    s.add_coupling((0,1), detuning=det, rabi_frequency=rabi, label="blue")
    s.add_coupling((1,2), detuning=det, rabi_frequency=rabi, label="red_transition")

    s.add_decoherence((1,0), gam)
    s.add_decoherence((2,1), gam, label='coupling')

    s.zip_parameters({'red_transition':'detuning', 'blue':'detuning'})
    s.unzip_parameters('zip_0')
    s.zip_parameters({'red_transition':'rabi_frequency', 'blue':'rabi_frequency'})
    s.unzip_parameters('zip_0')
    s.zip_parameters({'red_transition':'detuning', '(1,0)':'gamma'})
    s.unzip_parameters('zip_0')
    s.zip_parameters({'(2,1)':'gamma_coupling', 'blue':'detuning'})
    s.unzip_parameters('zip_0')


@pytest.mark.structure
def test_zip_zip():
    """Tests whether the zip zip function produces the same hamiltonian
    as extracting the diagonal manually.
    """
    s = rq.Sensor(5)
    det = np.linspace(-1, 1, 11)

    s.add_coupling((0, [1,2]), rabi_frequency=1, detuning=det, label="foo")
    s.add_coupling((0, [3,4]), rabi_frequency=3, detuning=2*det, label="bar")

    ham_big = s.get_hamiltonian()
    ham_big_diag = np.einsum("ii...->i...", ham_big)

    s.zip_zips("foo_detuning", "bar_detuning", new_label="foobar_detuning")
    ham_small = s.get_hamiltonian()

    np.testing.assert_equal(ham_small, ham_big_diag)
