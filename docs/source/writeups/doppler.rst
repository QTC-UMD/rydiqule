Doppler Averaging
=================

This document discusses the methods rydiqule uses to implement doppler-averaging of modelling results. 
As of version 2.1.0, rydiqule now provides two functions for implementing doppler-averaging: :func:`~.solve_steady_state` that averages numerically by sampling 
and :func:`~.solve_doppler_analytic` that averages one spatial dimension analytically and the remaining spatial dimensions numerically.
The theoretical background is provided for completeness,
but is fairly standard.
Rydiqule's implementation of this is less obvious in order to optimize computational efficiency by fully leveraging numpy's vectorized operations.
The primary goal of this document is to clearly outline, in a single place, the underlying conventions used by rydiqule when doppler averaging.

Assumptions
-----------

Rydiqule currently makes implicit assumptions when modelling atomic systems that influence the treatment of doppler averaging.
First, rydiqule's equations of motion are single atom, meaning that atom-atom interactions are ignored.
Second, rydiqule only solves these equations under the optically-thin assumption, meaning that input parameters do not change.
In particular, absorption of the optical fields through an extended ensemble is not considered.

Both assumptions greatly simplify doppler averaging.
Specifically, these assumptions allow us to further assume that atoms in different velocity classes do not interact,
meaning that doppler averaging entails a simple weighted average of the atomic response at different velocities by the velocity distribution.

Note that rydiqule assumes a three dimensional distribution of velocities, as is the case for a vapor.

Choosing Velocity Classes to Calculate
--------------------------------------

The primary influence of doppler velocities on the atomic physics is via Doppler shifts of the applied optical fields:
defined as :math:`\Delta = \vec{k}\cdot\vec{v}`, where :math:`\vec{k}=2\pi/\lambda\cdot\hat{k}` is the k-vector of the the optical field
and :math:`\vec{v}` is the velocity vector of the atomic velocity class in question.
The Doppler shift experienced by an atom is due to the sum of Doppler shifts from each component projection of the atomic velocity relative to the k-vector of the field.
In practice, the optical fields are often configured such that there is a basis where some of the k-vector components are zero,
reducing the number of dimensions that need to be considered.
In the simplest case, all optical fields are colinear,
meaning all Doppler shifts are due to velocities projected along a single axis.

Choosing which velocity classes to calculate when performing doppler averaging is a fairly complex meshing problem.
Our ultimate goal is to numerically approximate a continuous integral in one to three dimensions using a discrete sum approximation.
For thermal vapors, where the spread of doppler velocities is large,
resulting in Doppler shifts much greater than other detunings, linewidths, or Rabi frequenices in the problem,
most velocity classes only contribute minor incoherent absorption to the final result.
This allows for much coarser meshes.
The difficulty lies in determining which velocity classes, a-priori, can participate in generally narrow coherent processes.
This difficulty scales with the number of optical fields in the problem, as each new field increases the number of possible coherent resonances.

In rydiqule, we have striven to keep the specification of Doppler classes fairly flexible,
with the constraint that velocity classes along each averaged dimension are the same (ie a rectangular grid).
These classes are calculated in the :func:`~.doppler_classes` function,
which contains options for complete user specification of all classes,
as well as some convenience distributions that can be specified via a few parameters.

Maxwell-Boltzmann Distribution
------------------------------

Given that rydiqule gives single-atom solutions,
the total atomic response for a given set of parameters is directly proportional to the total number of atoms represented by those parameters.
The Maxwell-Boltzmann distribution is used to determine the fraction of the total population that is in each velocity class.

Following the conventions used by the `Wikipedia page for the Maxwell-Boltzmann Distribution <https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution>`_,
the distribution of velocities for an ensemble of three dimensions is

.. math:: f_\vec{v}(v_x, v_y, v_z) = \left(\frac{m}{2\pi k T}\right) e^{-\frac{m}{2kT}\left(v_x^2+v_y^2+v_z^2\right)}

Here, :math:`v_i` represents the atomic velocity component along the cartesian axes,
:math:`k` is Boltzmann's constant, :math:`T` is the ensemble temperature in Kelvin,
and :math:`m` is the atomic mass in kilograms.

This distribution has a number of general properties.
To begin, it is normalized such that integrating over all velocities in 3D space will give unity.
There are also a few characteristic speeds associated with this 3D distribution:
the most probable speed :math:`v_p`, the mean speed :math:`\langle v\rangle`, and the rms speed :math:`v_{rms}`.

.. math::

    \begin{align}
    v_p &= \sqrt{\frac{2kT}{m}}\\
    \langle v\rangle &= \sqrt{\frac{8kT}{\pi m}}\\
    v_{rms} &= \sqrt{\frac{3kT}{m}}
    \end{align}

Note that speed is defined as the magnitude of the velocity vector.
Also note that all of these quantities are related to each other by simple numerical prefactors.
Finally, observe that the distribution above can be easily separated into each cartesian component.

.. math:: f_\vec{v} = \prod_i \frac{1}{v_p \sqrt{\pi}} e^{-\frac{-v_i^2}{v_p^2}}

Note that the velocity distribution of each spatial component of the velocity is independently normalized.
This allows us to readily produce the appropriate weighting distribution for 1, 2 and 3 dimensional averages as needed
without having to perform redundant calculations of distributions where the density matrices to be averaged do not depend on a particular :math:`v_i`.
This function is implemented in :func:`gaussian3d`.

Numerically Averaging Velocity Classes
--------------------------------------

Given the above assumptions, the doppler average of the density matrix solutions is given by the integral

.. math::

    \bar{\rho_{ij}} = \int \rho_{ij}(\vec{v}) f_\vec{v} d^3v

This integral is numerically approximated via a finite sum.

.. math::

    \bar{\rho_{ij}} \approx \sum_{klm} \rho_{ij}(v_k, v_l, v_m) f_\vec{v}(v_k, v_l, v_m) \Delta v_k \Delta v_l \Delta v_m

In rydiqule, the weighting function :math:`f_\vec{v}` is implemented in :func:`gaussian3d`,
the volume element :math:`\Delta v_k \Delta v_l \Delta v_m` is calculated as the product of the gradients
along each axis as calculated by :func:`numpy:numpy.gradient` on the specified velocity classes.

We again note that when all k-vectors along a particular axis are zero,
:math:`\rho_{ij}(v_k, v_l, v_m)` is constant along that axis and that axis of the sum can be separated
and assumed to sum to unity due to normalization of the weighting distribution along each dimension.

Rydiqule's Implementation (Numeric Method)
++++++++++++++++++++++++++++++++++++++++++

Rydiqule's implementation of Doppler averaging is optimized to minimize duplicate calculations and fully leverage numpy's vectorized and broadcasting operations.
The general steps of :func:`~.solve_steady_state` are as follows:

#. Choose the doppler velocities to use for the mesh in the average.
#. Generate the Equations of Motion (EOMs) for the base zero velocity class using the machinery described in :doc:`Equations of Motion Generation <eom_notes>`.
#. Generate the part of the EOMs that are proportional to the atomic velocity components :math:`v_i`.
   This is done by generating EOMs for the system with all parameters set to zero except for the optical detunings with associated non-zero k-vector components :math:`k_i`,
   multiplied by :math:`v_P` to give the most probable Doppler shifts.
#. Generate the complete set of EOMs for all velocity classes via a broadcasting sum of the base EOMs with the Doppler EOMs multiplied by the normalized velocity classes along each axis.
   Each non-zero spatial axis that is to be summed over is pre-pended as an axis to the EOM tensor, as described in :doc:`Stacking Conventions <stacking_conventions>`.
#. Solve the entire stack of EOMs.
#. Weight the EOMs according to their velocity classes via the Maxwell-Boltzmann distribution and the discrete velocity volume element, as described above.
#. Sum the solutions along the velocity axes.

Internally, rydiqule defines the necessary components for Doppler averaging via three quantities:

- the normalized velocity classes :math:`d`, provided by :func:`~.doppler_classes`
- the most probable speed :math:`v_P` (in m/s), provided by the user as a class attribute
- the optical k-vector :math:`\vec{k} = 2\pi /\lambda\cdot\hat{k}` in (Mrad/m), provided for each coupling that has Doppler shifts to be averaged over

This construction has the benefit of allowing for meshes (ie velocity classes) to be defined in a general way relative to the distribution width :math:`v_P`,
making them easily re-usable for any velocity distribution that obeys the Maxwell-Boltzmann distribution.

Analytically Averaging Velocity Classes
---------------------------------------

Rydiqule's implementation of analytic doppler-averaging follows the propagator method derived in `Exact steady state of perturbed open quantum systems <https://arxiv.org/abs/2501.06134>`_
by Omar Nagib and Thad Walker. 
In one spatial dimension, the time evolution of a system is governed by the master equation in the superoperator form

.. math::

    \dot{\rho} = \mathcal{L} \rho 

At steady state, this equation becomes

.. math::

    \mathcal{L} \rho = 0

Considering a velocity class, :math:`v`, :math:`\mathcal{L}` can be divided into two parts:

.. math::

    \mathcal{L} \rho_v = (\mathcal{L_0} + v \mathcal{L_1}) \rho_v = 0

where :math:`\mathcal{L_0}` and :math:`\mathcal{L_1}` do not depend on :math:`v`.
A propagator, :math:`G_v`, is then constructed such that

.. math::

    G_v \rho_0 = \rho_v

where :math:`\rho_0` is the unique steady state solution when :math:`v=0`. As shown by Nagib and Walker, this propagator :math:`G_v` is constructed as

.. math::

    G_v = \frac{\mathbb{1}}{\mathbb{1} + v \mathcal{L}_0^- \mathcal{L}_1}

where :math:`\mathcal{L}_0^-` is the Drazin inverse of :math:`\mathcal{L}_0`.
Suppose that :math:`\mathcal{L}_0^- \mathcal{L}_1` can be eigendecomposed as

.. math::

    \mathcal{L}_0^- \mathcal{L}_1 = \sum_{\lambda = \lambda_1}^{\lambda_N} \lambda r_{\lambda} l_{\lambda}^T 

where :math:`r_{\lambda}` and :math:`l_{\lambda}` are the right and left eigenvectors of :math:`\mathcal{L}_0^- \mathcal{L}_1`, respectively.
Then the propagator is decomposed as

.. math::

    G_v = \sum_{\lambda = \lambda_1}^{\lambda_N} \frac{1}{1 + v \lambda} r_{\lambda} l_{\lambda}^T 

Thus,

.. math::

    \rho_v = G_v \rho_0 = \sum_{\lambda = \lambda_1}^{\lambda_N} \frac{1}{1 + v \lambda} r_{\lambda} l_{\lambda}^T \rho_0

Note that this approach extracts all :math:`v`-dependence into the algebraic prefactor :math:`1/1+v\lambda`.
As a result, we can simply integrate analytically over :math:`v` to compute the ensemble average:

.. math::

    \bar{\rho} = \int \rho_v f_{v} dv = \sum_{\lambda = 0} r_{\lambda} l_{\lambda}^T \rho_0 
    + \sum_{\lambda \not= 0} \frac{\sqrt{\pi/2}}{\sqrt{-\lambda^2} \sigma_v} \exp{\frac{-1}{2 \lambda^2 \sigma_v^2}} 
    \left(1 + \text{erf}\left[ \frac{\sqrt{-\lambda^2}}{\sqrt{2} \lambda^2 \sigma_v} \right] \right) r_{\lambda} l_{\lambda}^T \rho_0

Note that by rydiqule convention, :math:`\mathcal{L}_v` contains the prefactor :math:`\sqrt{2}\sigma_v`. Additionally, for numeric stability,
rydiqule utilizes `scipy.special.erfcx <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erfcx.html>`_.
Thus, the equation implemented in :func:`~.solve_doppler_analytic` is

.. math::

    \bar{\rho} = \sum_{\lambda = 0} r_{\lambda} l_{\lambda}^T \rho_0 
    + \sum_{\lambda \not= 0} \frac{\sqrt{\pi}}{\sqrt{-\lambda^2}} 
    \text{erfcx}\left( \frac{-\sqrt{-\lambda^2}}{\lambda^2} \right) r_{\lambda} l_{\lambda}^T \rho_0

Rydiqule's Implementation (Analytic Method)
+++++++++++++++++++++++++++++++++++++++++++

Rydiqule's implementation of Doppler averaging is optimized to minimize duplicate calculations and fully leverage numpy's vectorized and broadcasting operations.
In the case of one spatial dimension, :func:`~.solve_doppler_analytic` computes the doppler-averaged solution as outlined above.
In the case of two or three spatial dimensions, :func:`~.solve_doppler_analytic` computes the doppler-averaged solution as follows:

#. Choose the doppler velocities to use for the numeric axes in the average.
#. Generate the Equations of Motion (EOMs) for the base zero velocity class using the machinery described in :doc:`Equations of Motion Generation <eom_notes>`.
#. Generate the part of the EOMs that are proportional to the atomic velocity components in the numeric axes :math:`v_i`.
   This is done by generating EOMs for the system with all parameters set to zero except for the optical detunings with associated non-zero k-vector components :math:`k_i`,
   multiplied by :math:`v_P` to give the most probable Doppler shifts.
#. Generate the complete set of EOMs for all numeric velocity classes via a broadcasting sum of the base EOMs with the Doppler EOMs multiplied by the normalized velocity classes along each axis.
   Each numeric axis is pre-pended as an axis to the EOM tensor, as described in :doc:`Stacking Conventions <stacking_conventions>`.
#. Average over the analytic axis at each point on the numeric velocity mesh using the method above.
#. Weight the analytic averages according to their velocity classes via the Maxwell-Boltzmann distribution and the discrete velocity volume element, as described in the numeric method.
#. Sum the solutions along the numeric axes.

Internally, rydiqule defines the necessary components for Doppler averaging via three quantities:

- the normalized velocity classes :math:`d`, provided by :func:`~.doppler_classes`
- the most probable speed :math:`v_P` (in m/s), provided by the user as a class attribute
- the optical k-vector :math:`\vec{k} = 2\pi /\lambda\cdot\hat{k}` in (Mrad/m), provided for each coupling that has Doppler shifts to be averaged over

This construction has the benefit of allowing for meshes (ie velocity classes) to be defined in a general way relative to the distribution width :math:`v_P`,
making them easily re-usable for any velocity distribution that obeys the Maxwell-Boltzmann distribution.


.. _kvec update:

Migrating Doppler averaging from v1 to v2
-----------------------------------------

With the release of v2 of rydiqule, how the user provides the above quantities has changed for both :class:`~.Sensor` and :class:`~.Cell`.

In v1, the `'kvec'` parameter of the coupling was defined as the most probable Doppler shift vector (ie :math:`\vec{k}*v_P`).
This has been changed in v2 such that `'kvec'` is now defined as the optical k-vector only (in units of Mrad/m),
and :math:`v_P` is provided separately at :class:`~.Sensor` instantiation or by manually updating the :attr:`~.Sensor.vP` class attribute.
Put simply, moving :class:`~.Sensor` simulations from v1 to v2 means no longer multiplying the k-vector by :math:`v_P`,
and providing the :attr:`~.Sensor.vP` attribute.

For :class:`~.Cell`, v1 code followed the same old convention.
Now that :class:`~.Cell` has improved :doc:`ARC integration <nlj>`,
couplings in :class:`~.Cell` take the `'kunit'` argument which defines the unit propagation axis only.
The :math:`v_P` and :math:`2\pi/\lambda` factors are calculated automatically and applied to any coupling with `'kunit'` defined.
