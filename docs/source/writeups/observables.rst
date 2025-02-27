Observables
===========

Rydiqule can be used to calculate physical outcomes of experiments.
These functions get placed into two categories, **Experiments** and **Observables**.
Observables are quantities that can be computed directly from a Solution object, with no additional information.
These functions can be found as methods of :class:`~.Solution`.
Examples of Observables are the susceptibility, optical depth, transmission coefficient, and phase shift 
of a probing field.
Experiments are quantities that require more information.
At the time of writing, the only example is the :func:`~.get_snr` function.

Observable Derivation
---------------------

Most of the Observables are derived from the optical/electrical susceptibility.
Susceptibility is given by the optical polarizability :math:`P^+`,

.. math::
    
    P^+ = n \bra{g}\hat{d}\ket{e}\rho_{eg} = \epsilon_0 \chi E_{tot}/2

This equation may be found in Steck Eq. 6.69 [1]. 
:math:`\bra{g}\hat{d}\ket{e}` is the dipole matrix element,
:math:`\rho_{eg}` is the density matrix element,
:math:`n` is the atomic density,
:math:`\chi` is susceptibility,
and :math:`E_{tot}` is to amplitude of the electric field.
The factor of two arises from the rotating wave approximation,
such that :math:`P^+` represents the atomic polarizability in the rotating frame.

Observable Validation
---------------------

Rydiqule calculates the susceptibility using equivalent equations,
but written in terms of atomic constants :math:`\kappa` and :math:`\eta` (see [2] for definitions).
We validate that rydiqule calculates these observable quantities in a manner consistent with
a canonical reference, namely Steck's Quantum and Atom Optics notes [1].
The unit tests in 
`test_experiments.py <https://github.com/naqslab/Rydiqule/blob/master/tests/test_experiments.py>`_
test that rydiqule and Steck align,
but assume that the density matrix element :math:`\rho_{eg}` is correct.
Validity of density matrix elements is checked in numerous other tests.

Another more stringent test of Rydiqule is comparing the optical depth using two different methods that are both appropriate for a 2-level atom.
All losses from an ideal two-level atom arise from scattering from the excited state.
This allows one to write the scattering rate in terms of :math:`\rho_{ee}`,
as is done in Steck's Quantum and Atom Optics notes, Eq. 5.273 [1].
Another way to calculate the scattering rate is using the imaginary term of the susceptibility
corresponding to the probing field coupling.
This is the method Rydiqule uses, and can be found in Eq 6.73 of [1]. 
This test is implemented in `test_experiments.py <https://github.com/naqslab/Rydiqule/blob/master/tests/test_experiments.py>`_
as the `test_OD_with_steck` unit test.

Running just these observable and experiment unit tests can be done by installing the `pytest` dependencies (see :doc:`Unit Tests <../dev/tests>`),
and running the following command from the package parent directory.

.. code-block:: shell

    pytest .\tests -m experiments

.. rubric:: References

.. container:: reference csl-bib-body
    :name: refs

    .. container:: csl-entry
        :name: ref-steck_quantum_2022

        [1] D. A. Steck, *Quantum and Atom Optics*, 0.13.15 ed. (2022) 

    .. container:: csl-entry
        :name: ref-meyer_optimal_2021

        [2] D. H. Meyer *et. al.*, *Optimal atomic quantum sensing using electromagnetically-induced-transparency readout*, Phys. Rev. A **104** 043103 (2021)
