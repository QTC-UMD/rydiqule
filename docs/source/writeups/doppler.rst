Introduction
============

This document discusses the methods rydiqule uses to implement doppler-averaging of modelling results.
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

The basic process under these assumptions is as follows:

#. Calculate the density matrix solutions to the equations of motion for a wide range of velocity classes.
#. Weight the density matrix solutions with the Maxwell-Distribution, assigning the relative atomic density per velocity class to each solution.
#. Sum the weighted solutions.

Note that rydiqule assumes a three dimensional distribution of velocities, as is the case for a vapor.

Choosing Velocity Classes to Calculate
======================================

The primary influence of doppler velocities on the atomic physics is via Doppler shifts of the applied optical fields:
defined as :math:`\Delta = \vec{k}\cdot\vec{v}`, where :math:`\vec{k}=2\pi/\lambda\cdot\hat{k}` is the k-vector of the the optical field
and :math:`\vec{v}` is the velocity vector of the atomic velocity class in question.
The Doppler shift experienced by an atom is due to the sum of Doppler shifts from each component projection of the atomic velocity relative to the k-vector of the field.
In practice, the optical fields are often configured such that there is a basis where some of the k-vector components are zero,
reducing the number of dimensions that need to be considered.
In the simplest case, all optical fields are colinear,
meaning all Doppler shifts are due to velocities projected along a single axis.

Choosing which velocity classes to calculate when performing doppler averaging is a fairly complex meshing problem.
Our ultimate goal is to numerically approximate a continuous integratal in one to three dimensions using a discrete sum approximation.
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
==============================

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

Doppler Averaging
=================

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

Rydiqule's Implementation
=========================

Rydiqule's implementation of Doppler averaging is optimized to minimize duplicate calculations and fully leverage numpy's vectorized and broadcasting operations.
The general steps are are follows:

#. Choose the doppler velocities to use for the mesh in the average.
#. Generate the Equations of Motion (EOMs) for the base zero velocity class using the machinery described in :doc:`Equations of Motion Generation <eom_notes>`.
#. Generate the part of the EOMs that are proportional to the atomic velocity components :math:`v_i`.
   This is done by generating EOMs for the system with all parameters set to zero except for the optical detunings with associated non-zero k-vector components :math:`k_i`.
#. Generate the complete set of EOMs for all velocity classes via a broadcasting sum of the base EOMs with the Doppler EOMs multiplied by the velocity classes along each axis.
   Each non-zero spatial axis that is to be summed over is pre-pended as an axis to the EOM tensor, as described in :doc:`Stacking Conventions <stacking_conventions>`.
#. Solve the entire stack of EOMs.
#. Weight the EOMs according to their velocity classes via the Maxwell-Boltzmann distribution and the discrete velocity volume element, as described above.
#. Sum the solutions along the velocity axes.

Of particular note is the somewhat unconventional definition that Rydiqule uses for the "k-vector" of each optical field.
To begin, all quantities in the EOMs are given in units of Mrad/s, so the "k-vector" must be defined so that multiplication by the velocity in m/s will produce these scaled units.
Second, the "k-vector" defined for each coupling is *not* the optical k-vector, but rather the associated vector of most probable Doppler shift.

.. math:: K_i = k_i v_p

where :math:`k_i` is the optical k_vector component along the :math:`i`-th axis, :math:`v_p` is the most probable speed.
The Doppler shift is found by multiplying :math:`K_i` by :math:`d_i`, the normalized velocity along the :math:`i`-th axis.
The velocity along the :math:`i`-th axis is given by :math:`v_i = v_p d_i`.

This construction has two benefits.
First, it allows for meshes (ie velocity classes) to be defined in a general way relative to the distribution width :math:`v_p`,
making them easily re-usable for any velocity distribution that obeys the Maxwell-Boltzmann distribution.
Second, it allows the user the flexibility to define non-symmetric Doppler distributions, such as would be found in an atomic beam.
This is done by defining the optical field "k-vectors" as :math:`K_i = k_i v_{p_i}`, where :math:`v_{p_i}` is the most probable speed along each axis.
When doing this, the prefactor applied to the sum in :func:`gaussian3d` will need to be modified for quantitative accuracy.