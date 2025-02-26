Stacking Conventions
====================

For many probelms, Rydiqule is designed to implicitly handle multiple possible values for a
single parameter. For example, sweeping over a range of detuning values is handled in Rydiqule
simply by specifying the value of interest a list or array rather than a single value. This
enables a tremendous amount of flexibility in the problems that Rydiqule can solve naturally,
but there are some things worth noting about how Rydiqule specifically handles these problems, 
which are outlined in this document. 

Numpy arrays
------------

Typically, python lists are quite slow to perform operations on since they are dynamically sized
and typed. This allows tremendous flexibility in what can be put into a list but some problems
with how fast elements are accessed from that list and operating on. `Numpy arrays <https://numpy.org/doc/stable/user/whatisnumpy.html>`_
were created to address this limitation and Rydiqule makes extensive use of them to make
its calculations fast without losing the ease-of-use benefits of a Python interface. Fundamentally,
a numpy ``ndarray`` is a grid of numbers that has dimensionality `m1` by `m2` by `m3` and so on.
Numpy routines are written to operate on these arrays very quickls for large numbers of dimensions.

Stacking
--------

While numpy's own way of handling arrays via matrix broadcasting is `well-documented <https://numpy.org/doc/>`_,
and most of Rydiqule's own functions use the standard numpy conventions, there are some additional
assumptions Rydiqule makes when performing these operations that are worth outlining. Fundamentally,
Rydiqule thinks about these `ndarray` objects as groups of matrices, meaning that calculations 
are performed assuming, for example, that an array of shape ``(25, 3, 3)`` represents 25 :math:`3\times 3`
matrices. This is the array that would be generated if a list of 25 values were provided for a 
detuning value in a 3-level `Sensor`, and that `Sensor`'s `get_hamiltonian` function were called.
Rydiqule seamlessly handles all the work of generating those Hamiltonian matrices for each value, and returns
a single array object as an output. Similarly, if 2 values are specified as lists of length 25, 
a single arary of shape ``(25, 25, 3, 3)`` would be returned, with a different :math:`3\times 3`
Hamiltonian matrix for every combination of parameter values, for a total of 625 Hamiltonian matrices. 
Rydiqule terms this array a "stack" of Hamiltonians, and the "stack shape" are the axes preceding the actual
matrix value axes (in this case ``(25, 25)``), and is typically, denoted in Rydiqule as ``*l`` to
make clear that it could be any length of set of values depending on the problem.

Hamiltonian generations is created using this convention, and that carries through to generation of 
equations of motion, and any other quantities that may have a different matrix for each parameter
value. A Hamiltonian stack of shape ``(*l, 3, 3)`` will generate an equation of motion (eom) stack
of shape ``(*l, 8, 8)``, with all stack demensions remaining consistent. Rydiqule's internals
are, broadly, agnostic to exaclty what the dimensions ``*l`` represent, and work regardless, as long as the
dimensions corresponding to the actual quantities are in the expected position at the end.

Parameter Ordering
------------------

Given that any number of parameters may be defined as a list, Rydiqule needs a convention to ensure, in the final
result, the values represent what is expected and has not been turned around. It is important to Rydiqule's
design philosophy that internal variables not by tracked opaquely, and that quantities are, to the extent
possible, generated on the fly in a predictable and reprodicuble way. This begs the questions, which
axis corresponds to which value? Suppose the coupling between states 0 and 1 is swept in detuning over 25
values, as is the coupling between states 1 and 2, the stack shape will be ``(25, 25)``, but there 
are some uncertainties. One might assume that the first axis corresponds the the first laser, and the second
axis corresponds to the second laser. However, this is not necessarily obvious, and it might be the other
way around without a unifying convention. Rydiqule's solution to this problem turns out to be simple: python's
``.sort()`` function. Since it always orders things accordning to the same rules, there is a predictable outcome
to which axis is which. The parameters are represent by tuples: ``((0,1),"detuning")`` and ``((1,2),"detuning")``.
``.sort()`` will sort them first by the lower state of a transition, then by the upper state, then alphabetically by
the string parameter name (in this case detuning for both). 

With this simple convention, Rydiqule makes these arrays consistent accross functions. One can be sure that all 
values will be exaclty what is expected and line up properly for all quantities. Hamiltonians, equations of motion,
and solutions will all use the same rules. To avoid figuring this out manually for every system, the ``Sensor`` module
contains the ``.axis_labels()`` method, which returns a list of which axes are which in string form for
results interpretation. Note that the internal functions which calculate these values don't actually care
what the axis are, but they do keep them consistent between calculations.

Doppler
-------

It is worth a quick note how Rydiqule handles doppler broadening, because it leverages the same conventions
around stacking as other parameter scans, and it may be encountered and cause confusion if you use Rydiqule
enough. If doppler is accounted for in a solve, that typically is not invoked until the relevant ``solve`` 
function is called. Given a case of ``n_doppler`` velocity classes in 1 dimension, a new axis will be prepended
to the stack, resuling in a ``n_doppler`` new dopple-shifted Hamiltonian matrices matrix that was previously
in the stack. Typically, this is done under the hood, and these other solutions are averaged over before
a result is returned, but examining intermediate values may ultimately result in seeing these axes, even
if they are not present in the solution that is returned. Importantly, the solver internals are still agnostic
to what these preceding doppler axes represent, giving flexibility and allowing a single function to handle
all cases. Again, this is an intermediate step that typically does not affect how results are interpreted, 
it just helps to understand the internals a little better.