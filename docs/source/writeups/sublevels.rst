Fine and Hyperfine Structure with Sublevels
===========================================

In many situtations, accurate modeling requires solving for the entire atomic structure (i.e. explicit handling of the magnetic sublevels).
These situations include calculating changes in probing polarization and ellipticity angles,
non-degenerate states due to ambient magnetic fields,
or inhomogeneous coupling strengths due to sublevel structure.
As described in :doc:`nlj`, each sublevel must be treated independently within the basis,
leading to much larger basis sizes and an associated increase in the complexity of defining the system.
With the release of rydiqule v2, the ability to handle calculations involving sublevels has been greatly improved,
and this document will discuss those features and the associated physics conventions we use.

Sublevels in Sensor
-------------------

In :class:`~.Sensor`, states can be defined using arbitrary tuples,
and groups of states can be readily specified by using nested lists within the tuple.
Groups of "states" readily represent a manifold of sublevels.

For example, the following code creates a system with four states: a single ground state and a manifold of three excited states.
It then adds a coupling from the ground to each excited state.

.. code-block:: python

    g = (0, 0)
    e = (1, [-1, 0, 1])
    s = rq.Sensor([g, e])
    s.add_coupling((g, e), detuning=1, rabi_frequency=2, label='probe')

Here we see that the excited state manifold can be specified by a single object (representing 3 distinct sublevel states),
both at `Sensor` creation, and when applying couplings (as well as decoherences).

Another key feature is that the couplings and decoherences can define `coupling_coefficients`
which allow for scaling prefactors to be applied to a Rabi frequency for each "sublevel" coupling in the manifold.
Modifying the above example

.. code-block:: python

    g = (0, 0)
    e = (1, [-1, 0, 1])
    (e1, e2, e3) = rq.expand_statespec(e)

    cc = {
          (g,e1): 1/sqrt(2),
          (g,e2): 0,
          (g,e3): -1/sqrt(2)
    }

    s = rq.Sensor([g, e])
    s.add_coupling((g, e), detuning=1, rabi_frequency=2, coupling_coefficients=cc, label='probe')

The above represents a typical :math:`F=0\rightarrow F'=1` hyperfine transition
that is probed with linearly polarized light,
in a quantization axis that is aligned with the optical propagation axis.

Sublevels in Cell
-----------------

In :class:`~.Cell`, we combine the features of :class:`~.Sensor`
with a specialized named tuple class to track quantum numbers of states (:class:`~.A_QState`)
and :external+arc:doc:`ARC <index>` integration wrapped by an internal :class:`~.RQ_AlkaliAtom` interface
to enable automatic calculation of many atomic parameters directly from :class:`~.A_QState` state specifications.

State Definition
++++++++++++++++

We support three bases for defining the atomic states using :class:`~.A_QState`:

- NLJ: which averages over sublevel structure, described :doc:`here <nlj>`
- FS: the fine structure basis, where :math:`J` and :math:`m_J` are good quantum numbers
- HFS: the hyperfine structure basis, where :math:`f` and :math:`m_f` are good quantum numbers

In each case, :math:`n,l,j` are mandatory arguments in the definition of the :class:`~.A_QState`.
Providing the `'all'` argument to the other parameters will instruct :class:`~.Cell` to
expand the allowed fine or hyperfine states. for example, to define the entire D2 hyperfine transition structure in `Cell`

.. code-block:: python

    g = rq.A_QState(5, 0, 0.5, f='all', m_f='all')
    e = rq.A_QState(5, 1, 1.5, f='all', m_f='all')
    c = rq.Cell('Rb85', [g,e])

.. note::

    While it is simple to define large atomic bases this way,
    the hamiltonian size grows very quickly when using sublevels.
    This is especially true when setting `f='all'`.
    Be sure your model actually needs all these levels.

Coherent Coupling Definition
++++++++++++++++++++++++++++

We support four classes of transitions between states in these bases:

- NLJ :math:`\rightarrow` NLJ
- FS :math:`\rightarrow` FS
- HFS :math:`\rightarrow` HFS
- HFS :math:`\rightarrow` FS (and the inverse)

Note that we can perform models where different bases are used to describe different states, namely HFS and FS.
This is particularly useful for Rydberg atoms,
where the ground states are best described in the hyperfine basis,
but Rydberg states are best described in the fine structure basis.
An example of a typical, simplified definition would be

.. code-block:: python

    g = rq.A_QState(5, 0, 0.5, f=3, m_f='all')
    i = rq.A_QState(5, 1, 1.5, f=4, m_f='all')
    r = rq.A_QState(50, 2, 2.5, m_J='all')
    c = rq.Cell('Rb85', [g,i,r])
    c.add_coupling((g,i), 
        beam_power=5e-6, # watts
        beam_waist=200e-6, # m, 1/e^2
        detuning=0, q=0, label='probe')
    c.add_coupling((i,r),
        beam_power=50e-3,
        beam_waist=180e-6,
        detuning=0, q=0, label='couple')

Rydiqule is handling quite a bit automatically here.
First, it assumes a gaussian beam profile of waist `beam_waist` and total power `beam_power` to calculate the field strength,
which is then used to calculate Rabi frequencies between the outer product of all possible sublevels between the two manifolds.
Dipole-allowed transitions will have the associated quantities saved to the graph:

- `dipole_moment`: the transition dipole moment, in units of :math:`a_0 e`
- `coherent_cc`: the angular part of the dipole moment, used to scale the base Rabi frequency, in units of :math:`\langle J||d||J'\rangle/2`
- `rabi_frequency`: the reduced Rabi frequency, i.e. :math:`E\cdot\langle J||d||J'\rangle/2\hbar`

The transition Rabi frequency is given by `rabi_frequency*coherent_cc`.
The reason for breaking this up is because the `coherent_cc`
(which are at least proportional to Clebsch-Gordon coefficients)
are used when calculating observables to properly weight
various density matrix components corresponding to a single field.
Our convention of defining the base `rabi_frequency` relative to the
reduced J matrix element ensures that a common Rabi frequency can be defined for a field spanning many manifolds of sublevels.
The functions that calculate these quantities in rydiqule are
:meth:`~.RQ_AlkaliAtom.get_dipole_matrix_element`,
:meth:`~.RQ_AlkaliAtom.get_reduced_rabi_frequency`,
:meth:`~.RQ_AlkaliAtom.get_reduced_rabi_frequency2`,
and :meth:`~.RQ_AlkaliAtom.get_spherical_dipole_matrix_element`.
Details are given below on how the reduced Rabi frequency, spherical matrix element, and total dipole matrix element are defined.

.. note::

    NLJ transitions use a slight different specification internally.
    While the angular part is well defined,
    rydiqule does not use it since there are no
    other states to meaningfully compare against.
    For NLJ states, we instead enforce `coherent_cc=1`
    and the saved Rabi frequency is the full Rabi frequency.

There are alternate methods of specifying the coupling strength in `Cell`.
In all cases, the `dipole_moment`, `coherent_cc`, and `rabi_frequency` are defined the same way.

- The first is the `beam_power`/ `beam_waist` definition used above.
  The function :meth:`~.RQ_AlkaliAtom.gaussian_center_field` calculates the field amplitude at the center of the gaussian spatial mode.
  Then :meth:`~.RQ_AlkaliAtom.get_reduced_rabi_frequency2` calculates the reduced Rabi frequency.
- The second is by providing the `e_field` directly, which is largely equivalent the above `beam_power`/ `beam_waist` definition,
  but does not require assuming a gaussian profile (primarily used for RF transitions between Rydberg states).
- The third is by providing the `rabi_frequency` directly.
  In this case, rydiqule will assume the reduced Rabi frequency has been provided,
  and calculate the spherical matrix elements accordingly.

Incoherent Coupling Definition
++++++++++++++++++++++++++++++

Much like coherent couplings, decoherences between states can be specified between manifolds,
using a similar base value/coefficient paradigm.
In :class:`~.Cell`, however,
all decoherences due to state natural lifetimes are automatically calculated from the provided basis states.
This is done by leveraging ARC to automatically calculate the natural lifetimes of each state
as well as the dephasing rates between all states provided in the system definition.
These calculations are provided by :meth:`~.RQ_AlkaliAtom.get_state_lifetime` and :meth:`~.RQ_AlkaliAtom.get_transition_rate`.

Of course, it is common to not provide all possible states that every state could decay to.
In this situation, rydiqule has three configurable methods for dealing with discrepancies between
the natural lifetime of the state
and the total sum of dephasing rates out of a state.
Selecting between these options is controlled by the `gamma_mismatch` argument to :class:`~.Cell`.

1. `'ground'`: Send extra dephasing to the ground state.
   Here the "ground state" is defined as all atomic states defined in the system that share the same `n,l,j` quantum numbers with the lowest energy in the system.
   This is rydiqule's default behavior.
2. `'all'`: Proportionally scale existing dephasings so the sum matches the natural lifetime.
3. `'none'`: Ignore the discrepancy.

Dipole Matrix Element Definitions
---------------------------------

Here we define how the dipole matrix elements are defined for different kinds of transitions.
In each case, we follow ARC's general model of using the Wigner-Eckart theorem to divide the dipole matrix element into angular and radial parts.
In particular, we reference
all calculations to the symmetric reduced matrix element J
for the transition :math:`\langle J||d||J'\rangle`.
It is calculated by :external+arc:meth:`~arc.alkali_atom_functions.AlkaliAtom.getReducedMatrixElementJ`.
It only depends on `n,l,j`, has no angular dependence, and is the common element for all types of transitions :class:`~.Cell` supports.

Note that we use a slightly different convention for defining
the spherical dipole matrix element and reduced matrix element than what ARC uses internally.
Namely, we move a factor of two off the reduced matrix element onto the spherical matrix element.
This results in the `coherent_cc` parameter being closer to 1,
making the Rabi frequency more natural to define.
This convention is chosen merely for convenience.
The final Rabi frequency in the hamiltonian is identical to what ARC provides.

NLJ
+++

NLJ dipole matrix elements are defined as
the average magnitude of all dipole-allowed transitions
between sublevels of the two manifolds:

.. math::

    d_\text{NLJ} = \frac{1}{N}\sum_{m_j=-j}^{j}|\langle n,l,j,m_j | d | n',l',j',m_j+q \rangle|

Here, :math:`N` is defined as the number of non-zero elements in the sum.
The spherical matrix element is simply defined as the total matrix element
divided by the reduced matrix element in the J basis.

.. math::

    s_\text{NLJ} = d_\text{NLJ}/\langle J||d||J'\rangle


This is equivalent to taking the average of the Clebsch-Gordon coefficients for each dipole-allowed transition.

.. note::

    While the spherical matrix element is well defined,
    rydiqule does not use it in :class:`~.Cell` since there are no
    other states to meaningfully compare against.
    For NLJ states, we instead enforce `coherent_cc=1`
    and the saved Rabi frequency is the full Rabi frequency.

FS to FS
++++++++

Fine structure dipole matrix elements are calculated using
:external+arc:meth:`~arc.alkali_atom_functions.AlkaliAtom.getDipoleMatrixElement`
and the spherical matrix element is calculated using
and :external+arc:meth:`~arc.alkali_atom_functions.AlkaliAtom.getSphericalDipoleMatrixElement` with arguments `j,m_j,j',m_j',q`.

The functional definition (using Wigner3J symbols) is

.. math::

    d_\text{FS} = s_\text{FS}\cdot\langle j||d||j'\rangle\
    = (-1)^{j-m_j}\
    \begin{pmatrix}
    j & 1 & j'\\
    -m_j &-q &m_j'
    \end{pmatrix}\
    \langle j||d||j'\rangle

The Clebsch-Gordon coefficients are related to the spherical matrix element by

.. math::

    \langle j', m_j'; 1, q|j m_j\rangle = \sqrt{2j+1}\cdot s_\text{FS}

.. note::

    Rydiqule's convention is for :math:`\texttt{coherent_cc}=2s_\text{FS}`.
    This results in `coherent_cc=1` for stretch states between some :math:`J\rightarrow J'` manifolds.
    
    For example, :math:`\langle J=1/2, m_J=\pm1/2|d|J'=3/2, m_J'=\pm3/2\rangle` has `coherent_cc=1`.
    
    Note, however, that :math:`\langle J=1/2, m_J=\pm1/2|d|J'=1/2, m_J'=\mp1/2\rangle` has `coherenct_cc=4/3`.
    

HFS to HFS
++++++++++

Hyperfine structure dipole matrix elements are calculated using
:external+arc:meth:`~arc.alkali_atom_functions.AlkaliAtom.getDipoleMatrixElementHFS`.
The spherical matrix element is calculated using
:external+arc:meth:`~arc.alkali_atom_functions.AlkaliAtom.getSphericalDipoleMatrixElement` (using arguments `f,m_f,f',m_f',q`)
and :external+arc:meth:`!_reducedMatrixElementFJ` (which gives the reduced F matrix element in terms of the reduced J matrix element).

The functional definition is

.. math::

    d_\text{HFS} = s_\text{HFS}\cdot\langle n l j||d || n' l' j' \rangle\
    = (-1)^{f-m_{f}} \
    \left( \
    \begin{matrix} \
    f & 1 & f' \\ \
    -m_{f} & -q & m_{f}' \
    \end{matrix}\right) \
    \langle n l j f|| d || n' l' j' f' \rangle
    
where the reduced F matrix element is defined as

 .. math::

    \langle n l j f ||d|| n' l' j' f' \rangle \
    = (-1)^{j+I+f'+1}\sqrt{(2f+1)(2f'+1)} ~ \
    \left\{ \begin{matrix}\
    f & 1 & f' \\ \
    j' & I & j \
    \end{matrix}\right\}~ \
    \langle n l j||d || n' l' j' \rangle

Clebsch-Gordon coefficients for these transitions are related to the spherical matrix element by

.. math::

    \langle f', m_f'; 1, q|f m_f\rangle = \frac{(-1)^{j+I+f'+1}}{ \
    \sqrt{2f'+1} \
    \left\{\begin{matrix} f & 1 & f' \
    \\ j' & I & j\end{matrix}\right\}} \
    s_\text{HFS}

.. note::

    Rydiqule's convention is for :math:`\texttt{coherent_cc}=2s_\text{HFS}`.
    This generally results in `coherent_cc=1` for stretch states between :math:`J\rightarrow J'` manifolds.
    
    For example, in a :math:`J=1/2\rightarrow J'=3/2` manifold, the
    :math:`\langle F=2, m_F=\pm2|d|F'=3, m_F'=\pm3\rangle` moments have `coherent_cc=1`.
    
    For a :math:`J=1/2\rightarrow J'=1/2` manifold, the
    :math:`\langle F=1, m_F=\pm1|d|F'=2, m_F'=\pm2\rangle` and
    :math:`\langle F=2, m_F=\pm2|d|F'=1, m_F'=\pm1\rangle` moments have `coherent_cc=1`.

HFS to FS
+++++++++

Dipole matrix elements between fine and hyperfine structure sublevels are calculated using
:external+arc:meth:`~arc.alkali_atom_functions.AlkaliAtom.getDipoleMatrixElementHFStoFS`.
The spherical matrix element is calculated using
:external+arc:meth:`~arc.alkali_atom_functions.AlkaliAtom.getSphericalMatrixElementHFStoFS`.

The spherical part is calculated by expanding the fine basis state into its hyperfine components
and summing the elements weighted by Clebsch-Gordon coefficients.

.. math::

    s_\text{HFS-FS} = \sum_{f'}\langle j', m_j'; I, m_I|f' m_f' \rangle\cdot s_\text{HFS}


.. note::
    For these transitions, our definition of reduced Rabi frequency can result in `coherent_cc>1`,
    in similar situations as FS to FS transitions.