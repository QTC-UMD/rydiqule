Time-Dependence
===============

This document discusses how rydiqule implements time-dependent couplings between states.
It discusses the how to define these couplings in terms of the relevant coupling parameters
as well as some theoretical considerations when working in the rotating wave approximation.

Time-Dependent Couplings
------------------------

The general form for a time-dependent field is

.. math::

    A(t)\cos(\omega(t)t + \phi(t))

As it will be helpful later, we will break each time-dependent component into relevant static and time-dependent parts.
So :math:`A(t)\equiv A_0(1 + A_t(t))`,
:math:`\phi(t)\equiv \phi_0 + \phi_t(t)`,
and :math:`\omega(t)\equiv \omega_0 + \Delta +\omega_t(t)`.
Note that we have made an explicit choice to allow for a static detuning relative to a static transition frequency :math:`\omega_0`.

Rydiqule Coupling Parameters for Time dependence
------------------------------------------------

When defining a coupling in rydiqule, there are four parameters that define the time-dependence, all specified using specific keys in the coupling dictionary.

#. **The amplitude scale factor**: `'rabi_frequency'` or `'e_field'`. A constant scalar that multiplies the time-dependence function.
   Only one can be defined. If `'e_field'` is defined, the corresponding dipole moment is used internally to convert it to a Rabi frequency.
   Rydiqule will automatically apply the factor of 1/2 from the RWA when building the Hamiltonian.
#. **The normalized time dependence function**: `'time_dependence'`. A python function that takes a single argument, :math:`t`.
   It is normalized such that a value of 1 corresponds to the field amplitude set by `'rabi_frequency'` or `'e_field'`.
#. **The static detuning from the transition resonance**: `'detuning'`,
   A constant scalar that defines a fixed detuning relative to the transition frequency.
   When set, this implicitly defines the coupling in the rotating frame defined by the field frequency :math:`\omega_0 + \Delta` with the Rotating Wave Approximation applied.
   Rydiqule's convention is that positive detunings represent a blue detuning relative to the atomic transition
   (ie photon has more energy than the energy difference between the levels).
#. **or the transition frequency**: `'transition_frequency'`.
   A constant scalar that defines the atomic transition frequency.
   When set, this implicitly defines the coupling to *not* be in a rotating frame.
   As such, the time-dependence will need to use the full field frequency.
#. **The static phase offset**: `'phase'`. A constant scalar that results in a factor of :math:`e^(i\phi)` applied to the amplitude scale factor
   for fields defined in the rotating frame.
   Note that this scale factor can be incorporated directly into the `rabi_frequency`, which is allowed to be a complex number.
   This factor *cannot* be directly incorporated into alternate Rabi definitions of `Cell` such as `e_field`,
   which is why this parameter exists.
   Rydiqule defaults to a phase of 0 for all couplings in the rotating frame (i.e. `detuning` is defined) if phase is not provided.
   An error is raised when phase is defined outside the rotating frame.

Defining the time-dependence in this way allows us to efficiently construct the time-dependent equations of motion (EOMs) as an expansion of EOMs proportional to each time-dependent function.
If we let :math:`M_i` be an EOM tensor, :math:`A_i` the amplitude scale factor,
:math:`\Delta_i` the detuning, and :math:`f_i(t)` the time-dependent function,
we can express this expansion as

.. math::

    M_{tot} = M_{0}(\Delta_i, ...) + \sum M_i(A_i)\cdot f_i(t)

Note that :math:`M_0` represents the steady-state EOMs which includes the static detunings for all of the couplings, time-dependent or not.

Example Time-Dependencies
-------------------------

We now provide a few examples of how to write a time-dependent field coupling into the parameters exposed by rydiqule.

RF Heterodyne
+++++++++++++

In this situation, we want to couple a single transition with two fields with a small detuning between them.
One field is the local oscillator (LO), which has constant amplitude, frequency, and phase.
It is detuned from the transition resonance by :math:`\Delta_{LO}`.
The second field is the signal (S), and is detuned from the LO by a frequency :math:`\delta_S`.
It's amplitude is fixed and smaller than the LO amplitude.
The phase and frequency of this field is fixed.

One can write this total field as

.. math::

    \label{eq:heterodyne_field}
    E_{tot} = E_{LO} + E_{S} = E_{LO}\cos((\omega_0+\Delta_{LO}) t) + E_S\cos((\omega_0+\Delta_{LO} +\delta_S)t)

To convert to rydiqule parameters, we first move to the rotating frame defined by the field frequency :math:`\omega=\omega_0+\Delta_{LO}`.
Mathematically this is done by multipling the total field by :math:`e^{-i\omega t}`
and dropping the fast-rotating terms (i.e. that have components like :math:`e^{-i2\omega_0 t}`).
We then separate the constant amplitude prefactor from the normalized time-dependence.

.. math::
    E_{tot-rwa} = \frac{E_{LO}}{2}\left(1+\frac{E_S}{E_{LO}} e^{i \delta_S t}\right)

Note that the rotating wave approximation has discarded the counter-rotating term from :math:`\cos`,
leaving the single complex exponential in the time-dependence.

The rydiqule parameters are now defined as 

#. Constant amplitude: :math:`E_{LO}`
#. Time-dependence: :math:`\left(1+\frac{E_S}{E_{LO}} e^{i \delta_S t}\right)`
#. Detuning: :math:`\Delta_{LO}`

The constant amplitude does not include the factor of 2 from the rotating wave approximation
because rydiqule will automatically add it if the detuning coupling parameter is provided.
Note that the detuning parameter corresponds to the diagonal term of the resulting RWA hamiltonian.
Formally, this term is found by adding the (negative) rotating-frame frequency to the atomic energy on the diagonal.
In this simplified system, that gives :math:`\omega_0-\omega=-\Delta_{LO}`.
Rydiqule handles applying the negative sign internally so the user-specified detuning is :math:`\Delta_{LO}`.

Non-Rotating Frame
++++++++++++++++++

In some instances, one may wish to solve for a time-dependent coupling outside the rotating frame approximation.
This situation is signaled to rydiqule by not defining the `'detuning'` parameter of the relevant coupling.

As an example, we can consider the rf heterodyne field coupling in Eq. \\ref{eq:heterodyne_field}.
The rydiqule parameters without the rotating wave approximation would be

#. Constant amplitude: :math:`E_{LO}`
#. Time-dependence: :math:`\cos((\omega_0+\Delta_{LO}) t) + \frac{E_{LO}}{E_S}\cos((\omega_0+\Delta_{LO} +\delta_S)t)`
#. Detuning: undefined
#. Transition Frequency: :math:`\omega_0`

Note that in this case, a factor of :math:`1/2` will not be applied by rydiqule to the amplitude
and the `'transition_frequency'` is now a required parameter that will be used on the hamiltonian diagonal.

Frequency Sweep in the Rotating-Frame
+++++++++++++++++++++++++++++++++++++

In this situation, we want to work in a rotating frame which will remove the bulk of the field's frequency,
relaxing the time solver's timesteps.
We assume a fixed amplitude and a linear frequency sweep through resonance at a rate :math:`b`,
starting at detuning :math:`-\delta_0`.

This field coupling is written as

.. math::

    \Omega(t) = \Omega_0\cos(\phi(t))

where 

.. math::

    \phi(t) = \int_{0}^{t} \omega(\tau) d\tau

and :math:`\omega(t) = \omega_0 + bt - \delta_0`.

The field coupling with the integration performed is

.. math::

    \Omega(t) = \Omega_0\cos((\omega_0 - \delta_0)t  + \frac{bt^2}{2})

Moving to the rotating frame and re-writing to match rydiqule's inputs we have

.. math::

    \Omega_{rwa}(t) = \frac{\Omega_0}{2} e^{\frac{i b t^2}{2}}

The rydiqule parameters are now defined as 

#. Constant amplitude: :math:`\Omega_0`
#. Time-dependence: :math:`e^{\frac{i b t^2}{2}}`
#. Detuning: :math:`-\delta_0`

Static Phase Offsets
++++++++++++++++++++

When doing time-dependent calculations of multi-photon coherent effects to study steady-state spectra (eg studying response to frequency modulations),
it can be helpful to set random static phase offsets to the couplings to help the solution converge to steady-state faster.
If this isn't done, you often observe are large transient at :math:`t=0` due to all fields being approximately coherent even if their frequencies are different.

The field coupling is

.. math::

    \Omega(t) = \Omega_0\cos((\omega_0 + \Delta) t + \phi_0)

This phase offset can be moved to the static amplitude scaling factor,
reducing the computational complexity of the time-dependence.

.. math::

    \Omega_{rwa}(t) = \frac{\Omega_0}{2} e^{i \phi_0}

The rydiqule parameters are now defined as 

#. Constant amplitude: :math:`\Omega_0`
#. Time-dependence: undefined
#. Detuning: :math:`\Delta`
#. Constant phase offset: :math:`\phi_0`

An equivalent definition would be

#. Constant amplitude: :math:`\Omega_0 e^{i \phi_0}`
#. Time-dependence: undefined
#. Detuning: :math:`\Delta`

Note that this coupling does not have time-dependence
and would be solved as a steady-state field by not setting the `time_dependence` coupling parameter.

Closed-Loops
++++++++++++

If your system involves a closed-loop of couplings (ie there is a circular coupling path),
you have to track the overall phase of the circular path when moving to a rotating frame.
In particular, a time-dependent phase will accumulate in the loop if any of the couplings in the loop have non-zero detuning from atomic resonance.

Modelling a diamond scheme in a four-level atom would have the following four couplings.

.. math::

    \begin{align}
    \Omega^{(a)}(t) &= \Omega_a\cos((\omega_1 + \delta_a)t + \phi_a)\\
    \Omega^{(b)}(t) &= \Omega_b\cos((\omega_2 + \delta_b)t + \phi_b)\\
    \Omega^{(c)}(t) &= \Omega_c\cos((\omega_3 + \delta_c)t + \phi_c)\\
    \Omega^{(d)}(t) &= \Omega_d\cos((\omega_4 + \delta_d)t + \phi_d)
    \end{align}

The atomic transition frequenicies obey the relationship :math:`\omega_1 + \omega_2 - \omega_3 - \omega_4 = 0`,
with fields 1 and 4 coupling to the ground state,
and fields 2 and 3 coupling the highest excited state.
Note that the detunings for each field are defined such that a positive value corresponds to a blue detuning from atomic resonance.
The field frequencies obey the relationship :math:`\omega_a + \omega_b - \omega_c - \omega_d - \Delta = 0`,
where :math:`\Delta = \delta_a + \delta_b - \delta_c - \delta_d`.

Moving to a rotating frame is a non-unique transformation
(ie there are many equally valid choices).
This means that the time-dependent phase due to non-zero detuning of any field could be accounted for on any of the field couplings in a self-consistent way.
However, rydiqule makes an explicit choice for the rotating frame via its shortest path determination of the graph for each state.
Accurate modelling requires writing the couplings in the specific rotating frame chosen by rydiqule.

The basic choice made by rydiqule is to use the shortest path from the lowest index node of the connected sub-graph (typically 0).
If there are multiple shortest paths (ie multiple paths with the same shortest length),
only one is returned.
Typically it is the first equal-length path traversed by the algorithm.
Which one that is depends on internals of python (namely dictionary ordering).

Because of these choices,
rydiqule will choose multiple branching paths from the ground state in a closed-loop.
In order to correctly define the rotating frame,
you must ensure that your couplings are defined in rydiqule such that each path is relative to the ground state.
An example can demonstrate this subtlety.

In rydiqule code, the above couplings would be defined as (ignoring time dependence)

.. code-block:: python

    fa = {'states':(0,1), 'detuning':delta_a, 'rabi_frequency':Omega_a, 'phase':phi_a}
    fb = {'states':(1,2), 'detuning':delta_b, 'rabi_frequency':Omega_b, 'phase':phi_b}
    fc = {'states':(3,2), 'detuning':delta_c, 'rabi_frequency':Omega_c, 'phase':phi_c}
    fd = {'states':(0,3), 'detuning':delta_d, 'rabi_frequency':Omega_d, 'phase':phi_d}

    s = rq.Sensor(4)
    s.add_couplings(fa,fb,fc,fd)

Note that we have set the `fc` coupling with reversed ordering to indicate state 2 has higher energy than state 3.
We can use rydiqule to tell us which rotating frames will be chosen by calling :meth:`~Sensor.get_rotating_frames`.
This will return

.. code-block:: python

    {<networkx.classes.digraph.DiGraph at 0x1ef491e1910>: {0: [0],
    1: [0, 1],
    3: [0, 3],
    2: [0, 1, 2]}}

Note that state 3 has been defined directly from ground,
instead of the path `[0, 1, 2, -3]` as is often done when solving this problem on paper.
As a result, we have defined the `fd` coupling to be `(0,3)` instead of `(3,0)`
to match this convention.
This is important since all states need to rotate in a frame that starts from the same state.
If we instead defined coupling `fd` with `'states':(3,0)`,
the resulting path for state 3 is `[0, -3]` indicating state 3 is lower in energy than state 0
because all paths must start at 0.
Put another way, all coupling `'states'` tuples are assumed to be ordered
such that the second state has higher energy than the first.

Rotating the above couplings into rydiqule's default frame is accomplished using the unitary rotation operator

.. math::

    \left(
        \begin{array}{cccc}
        1 & 0 & 0 & 0 \\
        0 & e^{-i t \omega_a} & 0 & 0 \\
        0 & 0 & e^{-i t (\omega_a+\omega_b)} & 0 \\
        0 & 0 & 0 & e^{-i t \omega_d} \\
        \end{array}
    \right)

The couplings now in the rotating wave approximation are

.. math::

    \begin{align}
    \Omega^{(a)}_{rwa}(t) &= \frac{\Omega_a}{2} e^{i \phi_a}\\
    \Omega^{(b)}_{rwa}(t) &= \frac{\Omega_b}{2} e^{i \phi_b}\\
    \Omega^{(c)}_{rwa}(t) &= \frac{\Omega_c}{2} e^{-i \phi_c} e^{i (\delta_a+\delta_b-\delta_c-\delta_d) t}\\
    \Omega^{(d)}_{rwa}(t) &= \frac{\Omega_d}{2} e^{i \phi_d}
    \end{align}

Note that only coupling `fc` has any time-dependence for these CW fields.
Obviously, if any of the fields are not CW,
that extra time-dependence will need to be accounted for as described in the above examples
in addition to the time-dependence described here.

The rydiqule coupling parameters would be written as (letting :math:`i=[a,b,c,d]`)

#. Constant amplitude: :math:`\Omega_i e^{i \phi_i}` with `fc` differing by a sign :math:`\Omega_d e^{-i \phi_c}`
#. Time-dependence (coupling `fc` only): :math:`e^{i (\delta_a+\delta_b-\delta_c-\delta_d) t}`
#. Detuning: :math:`\delta_i` with :math:`\delta_c` not actually being inserted on the diagonal of the hamiltonian

An equivalent definition using the `phase` parameter is

#. Constant amplitude: :math:`\Omega_i`
#. Time-dependence (coupling `fc` only): :math:`e^{i (\delta_a+\delta_b-\delta_c-\delta_d) t}`
#. Detuning: :math:`\delta_i` with :math:`\delta_c` not actually being inserted on the diagonal fo the hamiltonian
#. Constant phase offset: :math:`\phi_i`, with `fc` differing by a sign :math:`-\phi_c`
