Equations of Motion Generation
==============================

This document details a few notes about the theoretical operations
taking place under the hood in the Rydiqule Modelling package. In
particular, we discuss the methods that Rydiqule uses to numerically
solve differential equations for density matrices.

Hamiltonian and Rotating Wave Approximation
-------------------------------------------

For a two level atom interacting with an electric field
:math:`\textbf{E}`, the dipole interaction Hamiltonian is,

.. math:: H = \omega \ket{e}\!\bra{e} -\textbf{d}\cdot \textbf{E}

where the Rabi frequency is defined as :math:`\Omega = \textbf{d}\cdot\textbf{E}/\hbar`.
The electric field is,

.. math::

   \begin{aligned}
       \textbf{E}&=\textbf{E}_0 \cos(\omega t + \phi)\\
       &=\frac{\textbf{E}_0}{2}(e^{i \omega t} + e^{-i \omega t})\end{aligned}

The dipole operator can be written  [1]

.. math:: \textbf{d} = \bra{g}\textbf{d} \ket{e}(\ket{g}\!\bra{e}+ \ket{e}\!\bra{g})

The operator :math:`\ket{g}\!\bra{e}` evolves at frequency
:math:`e^{i \omega t}` under the bare Hamiltonian, so we expand and take
the slowing evolving terms (RWA, see  [1]).

.. math::

   \begin{aligned}
       H_\text{RWA} &= \omega \ket{e}\!\bra{e} \\
       &-\bra{g}\textbf{d}\ket{e}\cdot \frac{\textbf{E}_0}{2}(\sigma^+e^{-i\omega t}+\sigma^-e^{i\omega t})\end{aligned}

where :math:`\sigma^+ = \ket{g}\!\bra{e}` and
:math:`\sigma^- = \ket{e}\!\bra{g}`

Equations of Motion
-------------------

The Master equation that governs system dynamics used by Rydiqule is the `Linbladian <https://en.wikipedia.org/wiki/Lindbladian>`_.
It is a semi-classical formulation of the Schroedinger equation for use in open quantum systems.

.. math:: \dot{\rho} = -\frac{i}{\hbar}[H,\rho]-\mathcal{L}

This can be written in summation notation (using Kronnecker deltas),

.. math::

   \dot{\rho}_{ij}=-\frac{i}{\hbar}\left(H_{ik}\rho_{kj}
   - \rho_{ik}H_{kj}\right)
   + \sum_{m,n}\frac{\Gamma_{mn}}{2}\left(2\delta_{ij}\rho_{mm}
   - \delta_{mi}\rho_{ij}-\delta_{mj}\rho_{ij}\right)

More generally, we can re-write the system of equations as a super-operator,

.. math:: \dot{\rho}_{ij} = R_{ik}\rho_{kj}

By re-shaping these equations, using :external+numpy:func:`numpy.reshape`, we can convert
this into a linear set of differential equations in matrix form (see :func:`~.generate_eom()`).
With the re-shaped density vector :math:`p`, the equations of motion become

.. math::

   \label{eq:master}
       \dot{p}_l = M_{li}p_{i}

This is a linear set of equations we can easily solve with
:external+numpy:func:`numpy.linalg.solve`. 

Our reshaping procedure defines a new computational basis
that is, for basis size b,

.. math:: l = b\times j+i

For example,

.. container:: center

   = ==
   l ij
   = ==
   0 00
   1 10
   2 20
   3 01
   4 11
   5 21
   = ==

For the programmatic code, we need knowledge of this relationship.

Removing the Ground State
-------------------------

The density vector (matrix) is physically constrained, so that the total
population is one. This constraint is not included in the equations of
motion. This leads to numerical instabilities. The best way to fix this
instability is to algebraically remove one of the equations of motion
(ie the ground state). To remove the ground state, we apply the
constraint

.. math:: \rho_{00} = 1-\sum_i\rho_{ii}.

Writing this in terms of :math:`p` gives,

.. math:: p_0 = 1-\sum_{x} p_{[(b+1)\times x]}

We use this to re-write Eq. \\ref{eq:master},

.. math::

   \label{eq:groundRemoved}
       \dot{p}_l =  M_{li}p_{i} - M_{l0}p_{0} + M_{l0}\left(1-\sum_{x} p_{[(b+1)\times x]}\right)

This is the equation we must implement to remove the ground state.

In the code, we can apply Eq. \\ref{eq:groundRemoved}
and then we can simply remove the first column of :math:`M_{li}`. In the
code, we implement this transformation by replacing the set of equations
:math:`M_{li}`,

.. math:: M_{li} \rho_i \rightarrow (M_{li} + M'_{li})\rho_i + c_l

The constant term :math:`c` is equivalent to the first column of
:math:`M_{li}`.

.. math:: c_l = M_{l0}

The term we need to add, :math:`M'` is

.. math:: M'_{li} = -M_{l0}\sum_x p_{[i=(b+1)\times x]}

This can be implemented as the tensor product of two vectors

.. math:: M'_{li} = -M_{l0} \otimes p^*

where :math:`M_{i0}` is just :math:`M[:,0]` and
:math:`p^*=p_{[j=(b+1)\times x]}` is a vector of ones and zeros that is
generated with list comprehension.

The end result is an equation where each ground state term of the
density matrix :math:`\rho_{00}` is replaced by the sum of all excited
states.

Making the Equations Real
-------------------------

Numerically, converting to a real set of equations is important, because
it prohibits the buildup of “imaginary populations” in quantum states.
In other words, some equations in the equations of motion are physically
required to be real, and some are complex. Machine rounding errors
causes leakage into the imaginary parts of the populations equation,
which is unphysical. Under certain solving conditions the equations are
not stable to this buildup. Converting all the equations to real solves
the issue.

The equation we want to solve (for the density vector :math:`p`) is,

.. math:: \dot{p_c} = M_c\cdot p_c + c_c

where the :math:`_c` notation represents that each term is complex.

The change in basis that we implement is shown below in equation and
table format,

.. math::

   \begin{aligned}
       \rho_{ii} &\rightarrow \rho_{ii}\\
       \rho_{ij} &\rightarrow Re(\rho_{ij}),\,\, i>j\\
       \rho_{ji} &\rightarrow Im(\rho_{ij}),\,\, i<j\end{aligned}

.. container:: center

   ========= ================= =====================
   :math:`l` real :math:`ij`   complex :math:`ij`
   ========= ================= =====================
   0         :math:`\rho_{00}` :math:`\rho_{00}`
   1         :math:`\rho_{10}` Re(:math:`\rho_{10}`)
   2         :math:`\rho_{20}` Re(:math:`\rho_{20}`)
   3         :math:`\rho_{30}` Re(:math:`\rho_{30}`)
   4         :math:`\rho_{01}` Im(:math:`\rho_{10}`)
   5         :math:`\rho_{11}` :math:`\rho_{11}`
   ========= ================= =====================

We implement this with a transformation matrix :math:`U` that is unitary
up to a scale factor,

.. math::

   \begin{aligned}
       M_r &= U\cdot M_c \cdot U^{-1}\\
       c_r &= U\cdot c_c \end{aligned}

This matrix is calculated in the :func:`~.get_basis_transform` helper
function and is subsequently used to transform between the complex and
real bases.

Converting Solutions Back to the Complex Basis
----------------------------------------------

Rydiqule's solutions are kept in its computational basis (i.e. real, with first state removed).
Standard observable calculations using this basis are provided by :class:`~.Solution`.
If you would like to convert the solutions back to the complex basis directly,
the utility function :func:`.sensor_utils.convert_dm_to_complex` can be used.
The :attr:`Solution.complex_rho` convenience attribute provides this conversion for
solutions.

.. rubric:: References

.. container:: references csl-bib-body
   :name: refs

   .. container:: csl-entry
      :name: ref-steck_quantum_2022

      [1]D. A. Steck, *Quantum and Atom Optics*, 0.13.15 ed. (2022).
