Changelog
=========

v2.0.0
------

Improvements
++++++++++++

- Added ability to use arbitrary tuples to define a state.
- Added concept of `StateSpec`, which is a tuple with a nested list for at least one element.
  Rydiqule will interpret this as a list of state tuples expanded along the nested element.
  For example, `(0, [-1, 0, 1])` is interpreted as the group of states `[(0, -1), (0, 0), (0, 1)]`.
- Added ability to couple groups of states to a single field using a single function. This involves several modifications:

  - The `Sensor.add_coupling_group` has been added which takes two lists of states and couples all combinations of states.
    Relative scaling of each sub-coupling is controlled via a `coupling_coefficients` dictionary.
    If an entry is missing from this dictionary, that sub-coupling will not be added.
  - The previous `Sensor.add_coupling` has has been renamed to `Sensor.add_single_coupling`.
  - The new `Sensor.add_coupling` function now takes either a pair of states or a pair of `StateSpec` (which define multiple states), and dispatches appropriately.
  - Identical functionality now exists for the `add_decoherence`, `add_self_broadening`, and `add_energy_shift` functions.
  - Added a string representation of `Sensor` so that a summary of the data on its graph is shown when calling `print(<sensor>)`.

- Complete overhaul of `Cell`.
  
  - States are now specified using the named tuple class `A_QState`, which tracks quantum numbers explicitly.
  - `Cell` now supports calculations with and without sublevels defined.
    
    - If sublevels are not defined (the NLJ basis), results now use an averaged dipole moment.
      Otherwise, calculations should be identical to `Cell` from v1.
      See :ref:`nlj update` for further details.
    - Sublevels can now be defined, in either the fine structure (FS) or hyperfine structure (HFS) bases.
      Couplings between bases are also supported. This support heavily relies on new coupling group functionality of `Sensor` described above.
  
  - A new interface to ARC calculations has been defined, `RQ_AlkaliAtom`,
    which defines methods for calculating atomic parameters between pairs of `A_QState`.
  - The `kvec` parameter for couplings is replaced with `kunit`, the unit propagation axis.
    Necessary prefactors for calculation are automatically applied using ARC functionality.
    See :ref:`kvec update` for details.

- Definition of `kvec` has been redefined to be the field k-vector,
  instead of the most probable velocity vector.
  The most probable speed `vP` must now be provided separately either via init keyword argument
  (i.e. `Sensor(4, vP=250)`) or by setting the `Sensor.vP` attribute.
  See :ref:`kvec update` for details.
- Allow the `about` function to optionally print numpy backend information.
- Created custom `RydiquleError` and `RydiquleWarning` classes and subclasses to denote
  errors and warnings specific to Rydiqule.
  By default, `RydiquleError` will suppress the raise statement in the traceback via patches to exception hooks.
  This can be disabled by calling `rq.set_debug_state(True)`.
  `rq.set_debug_state` also controls debugging print statements during Sensor/Cell creation.
- Added utility functions for converting density matrices between a standard complex basis
  and rydiqule's real with ground removed computational basis (`convert_dm_to_complex` and `convert_complex_to_dm`).
  Added utility to add ground state to real computational basis results `convert_to_full_dm`.
  Also added utility to confirm density matrices are physical (`check_positive_semi_definite`).
  This functionality is now used in `solve_time` to ensure user-provided `init_cond` are physical.
- Improved the ergonomics of `Sensor.zip_parameters` so that parameters to be zipped are specified by a dictionary keyed by coupling with entries that specify the parameter.
- Changed `Sensor.add_energy_shift` to to be more in line with couplings, so that it works as a dispatch function for single shifts or groups.
  Also added `add_energy_shift_group` which will add and zip a group of energy shifts.
- Package versioning is now handled by `setuptools_scm` which introspects the version based on git tags (if present).
  We also use this functionality to dynamically update the version on import when running an from an editable install, to account for local development.
- Overhaul of the observable functions of `sensor_solution` to use a more physical definition of observable defined by the trace of the density matrix times an operator.
  Additionally, those functions are transparent to allow more flexible definitions of observables.
- `sensor_solution` now stores the `Sensor.couplings` graph directly.
- Added `Sensor.get_time_hamiltonian` method which returns the system hamiltonian at a specific time `t`.
- Reworked `Sensor`'s time hamiltonian generation function structure to be more clear.
- `draw_diagram` now scales the linewidth of coupling arrows based on the magnitude of the Rabi frequency.
- Improved accuracy of language regarding rotating frame choices in rydiqule's physics documentation.
- Greatly over-hauled and expanded example notebooks and documentation to cover new features and clarify old ones.
- Added a `Sensor.zip_zips` method to zip axes already containing multiple zipped parameters.
- Updated CyRK timesolver backend to use `pysolve_ivp`. 
  Added an improved differential equation generation method `'flat'` which improves performance by ~30%.
  This new method is currently not compatible with doppler solves.
- Extended the automated test suite to check docstring examples.
- Added an analytic 1D doppler-averaged steady-state solver `doppler_1d_exact`.
  This solver is significantly faster for Doppler-averaged solves.
  For now, this solver is considered experimental.

Bug Fixes
+++++++++

- Fix bug where re-adding a coupling that had a zipped parameter did not invalidate the zip.
- `transition_frequency` is now correctly marked as a non-scannable parameter
- Fixed bugs in `draw_diagram` with un-coupled states and dephasings not toggling correctly.
- Fixed issue where passing the same numpy array to two zipped parameters would result in incorrect tensor broadcasts.

Deprecations
++++++++++++

- Overhaul of `Cell` is likely to change results of code that used `Cell` in v1,
  if not fail outright. Please see documentation for migration guide between v1 and v2.
- Previously deprecated experiment functions have been deleted from `rydiqule.experiments`.
  These deprecated functions are: `get_transmission_coef`, `get_susceptibility`, `get_phase_shift`,
  `get_solution_element`, and `get_OD`.
  Since v1.1.0, this functionality has been incorporated directly into `Sensor_Solution`.
- Internally-used utility functions have been removed from the top-level namespace.
  All these functions can still be accessed by importing from their sub-module locations.
  Functions removed from the top-level namespace are `generate_eom`, `get_basis_transform`,
  `solve_eom_stack`, `generate_eom_time`,
  `get_doppler_equations`, `generate_doppler_shift_eom`, `doppler_classes`, `doppler_mesh`,
  `apply_doppler_weights`, `compute_grid`, `matrix_slice`, `memory_size`, `get_slice_num`,
  and `get_slice_num_t`
- Removed deprecated `Cell.add_states` method.
- `suppress_rwa_warn` kwarg for `Sensor.add_coupling` is deprecated.
  Now use `warnings.simplefilter('ignore', rq.RWAWarning)` to suppress the warning.
- Renamed `Sensor.get_time_couplings` to `Sensor.get_time_hamiltonian_components`.
- Removed `Sensor.get_time_hamiltonians`. Instead call `Sensor.get_hamiltonian` and
  `Sensor.get_time_hamiltonian_components` directly.
- `suppress_dipole_warn` kwarg for `Cell.add_coupling` is deprecated.
  It is no longer possible to add a non-dipole allowed coupling in Cell.
- `Solution` object is no longer a bunch/dict object.
- Dropped support for numba-only timesolver backends.

  - `numbakit-ode` was never much of an improvement, if any for our types of problems
  - `nbrk_ode` (and it's modern replacement `nbsolve_ivp`) are not actively being maintained by CyRK.
    They also have not provided significant improvements for our types of problems.

v1.2.3
------

- Minor hotfix release to pin down incompatible versions of numpy and cyrk dependencies.

v1.2.2
------

Improvements
++++++++++++

- Now also distribute rydiqule via an `anaconda channel <https://anaconda.org/rydiqule/repo>`_.

Bug Fixes
+++++++++

- Fixed bug where t=0 time-dependent hamiltonians calculated in `solve_steady_state`
  were double counted if more than one time-dependent coupling was present.


v1.2.1
------

Bug Fixes
+++++++++

- Fixed bug in energy level shifts where shifts overwrote detunings instead of adding.

v1.2.0
------

Improvements
++++++++++++

- Level diagrams now use `Sensor.get_rotating_frames` to provide better plotting of energy ordering of levels.
- Level diagrams now allow for optional control of plotting parameters by manually specifying `ld_kw` options on nodes and edges.
- Added the ability to specify energy level shifts (additional Hamiltonian digonal terms) not accounted for by the coupling infrastructure.


Bug Fixes
+++++++++

- `Sensor.make_real` now returns correct sized `const` array when ground is not removed.
- Many updates to type hints to improve their accuracy.

Deprecations
++++++++++++

- Remove `Solution._variable_parameters` in favor of property checking the observable parameters.
- Renamed `Sensor.basis()` and `Solution.basis` to `Sensor.dm_basis()` and `Solution.dm_basis`
  to disambiguate physical basis from computational basis.

v1.1.0
------

Improvements
++++++++++++

- Added the ability to specify hyperfine states in a `Cell`. They are distiguished by having 5 quantum numbers `[n, l, j, f, m_f]`.
- `kappa` and `eta` are now proprties of `Cell` which are calculated on the fly.
- Separated rotating frame logic from hamiltonian diagonal generation into a new function `Sensor.get_rotating_frames()`.
  Allows for simple inspection of what rotating frame rydiqule is using in a solve.
- Reworked the under-the-hood parameter zipping framework. This should have minimal impact on user-facing functionality.

  - Hamiltonians with zipped parameters are no longer generated with a `diag` operation.
  - Zipped parameters are now handled with a dictionary rather than a list.
  - Zipped parameters can now be given a shorthand label rather than the default behavior of concatenating individual labels.

- The rearrangement of axes in a stack is now defined completely by the behavior of `axis_labels()`.
- Added a `diff_nearest` boolean argument to `get_snr`. When true, calculates SNR based on nearest neighbor diff.
  This is in contrast to the default behavior of taking the difference relative to the first element.
  One case where this is necessary is when getting SNR vs LO Rabi frequency of a heterodyne measurement.
- Added the ability to label states of a sensor with the `label_states` method. States with a label matching a particular pattern can be accessed with the `states_with_label` function.
- Timesolver now allows for returning doppler-averaged solutions without applying the doppler weight factors.
  This is mostly useful for internal testing.
- `solve_steady_state` now treats time-dependent couplings as having their :math:`t=0` value.
  Most importantly, this affects the default behavior for timesolve initial condition generation and should limit large transient behavior.
  This also allows the user to specify if time-dependent couplings should be solved with field on or off in steady-state
  by altering their :math:`t=0` value (eg changing between sin and cos).
- Added unit tests for observables, (susceptibility, optical depth, transmission coefficient, and phase shift).
- All Observables (susceptibility, optical depth, etc) now only require a `Solution` object to run.
- `rq.D1_states` and `rq.D2_states` can now specify the atom via string with any isotope specification (including none)
- `get_snr` now warns if any couplings have time-dependence, which are ignored.
- Zipped parameter labels may now include underscores
- `about` function now conceals the user's home directory by default when printing paths
- Moved level diagram plotting to use an external library

Bug Fixes
+++++++++

- Fixed return units of `get_snr` to actually return in 1s BW. Previously was returning in 1us BW.
- Sign errors when specifying detunings both in and out of the rotating frame have been fixed.
  All detuning signs now follow the convention that positive = blue detuned from atomic resonance,
  so long as the couplings are added correctly (ie second state of `states` tuple is always the higher energy one).
- Fixed potential issue in `get_snr` where output results could be overwritten to views of intermediate arrays
- Fixed numerical bugs in observables: phase shift, susceptibility, optical depth, transmission coef.  Now unit tested 
  against Steck Quantum Optics notes.
- Ensure that non-dipole-allowed transitions are properly warned about in `Cell.add_coupling` with ARC==3.4


Deprecations
++++++++++++

- The new `kappa` and `eta` properties of `Cell` directly calculate from Cell properties.
- Time-solver backends (except scipy) are now optional dependencies that are no longer installed by default. To install them, use the `pip install rydiqule[backends]` command.
- The uncollapsed stack shape can no longer be accessed to avoid confusion.
- Removed the ability to pass additional parameters to `np.meshgrid` through the `get_parameter_mesh` function. 
- `get_snr` no longer returns in units of 1us.
- Default timesolver initial conditions no longer assume time-dependent couplings have the value of `rabi_frequency`.
  It is now `rabi_frequency` times the `time_dependence`.
- Multiple sign errors have been corrected in `Sensor` and `Cell` with regards to detunings.
  Results that are asymmetric about zero detuning are likely to change.
  Please ensure all couplings are following correct sign conventions for consisten results
  (ie second state of `states` tuple has higher energy).
- most of the functions in experiments.py have been moved to become methods of `Solution` class.

v1.0.0
------

Improvements
++++++++++++

- Steady-state behavior for time-dependent fields (and thus initial conditions for time solves) is now computed as a static value rather than zero (previous behavior).
- Added a flag in `scipy_solve` to specify how to define the right-hand function of the differential equation, to use either loops (the newer method) or list comprehension (the older method).
- Implemented `ruff` linting rules as an action for new PRs to help enforce good coding practices.
- Implemented unit-testing action for new PRs to help automate catching regression bugs.

Bug Fixes
+++++++++

- Fixed a broken uinit test that did not affect package functionality.
- Fixed issue where level diagrams don't draw correctly if all non-zero dephasings are equal.


Deprecations
++++++++++++


v1.0.0rc2
---------

Improvements
++++++++++++

- Added a `copy` method to solution.
- Expanded the `Solution` object to include more clear axis labels and the basis of the sensor used.
- Begin hosting public documentation on readthedocs.

Bug Fixes
+++++++++

- Changed an `isinstance` check to `hasattr`, fixing an occasional issue with reloading `rydiqule` in jupyter notebooks.
- Fixed issue where submodules wree not installed outside of editable mode.
- Fixed a bug where additional arguments like warning suppression could not be passed to Sensor.add_couplings

Deprecations
++++++++++++


v1.0.0rc1
---------

Improvements
++++++++++++

- Added a warning in cell if `add_coupling` is called a dipole-forbidden transition.
- The zip_parameters function can now be called on parameters of different types (e.g. detuning with rabi_frequency)
- The time solver now can call ivp solvers outside its own module. This allows for more quickly using different backend solvers for time-dependent problems. 
- Implement timesolver backends based on CyRK's cython and numba ode solvers
- Optimize scipy backend of the timesolver for smaller dimensional problems

Bug Fixes
+++++++++

- Fixed issue where solvers would save doppler axes labels and values even when they are summed over to the solution object
- Fixed a bug where energy level diagrams broke when decochernce rates were scanned.
- Fixed issue where compiled timesolvers could not solve doppler averaged problems.
- Fixed issue where certain doppler solves could not be sliced correctly


Deprecations
++++++++++++



v0.5.0
------

Improvements
++++++++++++

- Add isometric-population meshing option to `doppler_mesh`
- Allow `get_rho_ij` to accept a `Solution` object directly, in addition to solution numpy arrays
- Add `get_rho_populations` helper function to efficiently get the trace of density matrix solutions
- Allow `beam_power` or `beam_waist` to be scanned parameters in a `Cell` coupling
- Add more information to `Solution` objects returned by the solvers
- Allow dephasings to be scannable parameters.
- Updated the framework for scanning parameters to generate relevant lists on the fly

  - Note: This changes the order of axes in a stack. Previously, the axes would be ordered based on the order they were added to the system.
    They are now ordered based on python's `sort()` applied to a tuple of ((low_state, high_state), parameter_name).
    As a result, they will be ordered first by lower state, then by upper state, then alphabetically by parameter name (e.g. "detuning", "rabi_frequency")
    In cases where the code was being used for simulations, this may affect cases where axes were defined specifically by number, and these may need to be updated.
    
- Added a distinction between stack shapes in steady-state vs time-dependent. For example, a steady-state hamiltonian stack may have shape `(10,1,3,3)` while the time dependent portion may have shape `(1,25,3,3)`.
- Renamed the `ham_slice` function to `matrix_slice` and allowed it to iterate over any number of matrices.
  - Updated internals of solver functions to use this framework.
- `zip_parameters` function no longer enforces parameters be the same type.

Bug Fixes
+++++++++

- Fixed several issues with parameter zipping functionality producing errors when sensor methods were called multiple times.
- Fixed issue where `get_rho_ij` incorrectly calculated the `rho_00` element
- Allow `Cell.add_coupling` to accept a list of e-field values
- Fixed an bug where specifying a list of `rabi_frequency` in a coupling with `time-dependence` would raise an error when solved
- Fixed issue with dephasing broadcasting preventing hamiltonian slices for large solves

Deprecations
++++++++++++

- Removed all `sensor_management` functionality as too difficult to maintain generally and securely.
- Removed the internal `_variable_couplings`, `_variable_parameters`, and `_variable_values` attributes from sensor.

v0.4.0
------

Improvements
++++++++++++

- Changed the handling of decoherent transitions to be stored on graph edges rather than as a separate attribute.
  
  - Gamma matrix is now calculated on the fly with the `decoherence__matrix()` method.
  - Decoherent transitions are now added with with the `add_decoherence()` function in `Sensor`.
  - `Cell` now calculates tranistion frequencies and decay rates automatically and places them on the appropriate graph edges.

- Changed the `Sensor.couplings` attribute from a `nx.Graph` to an `nx.DiGraph`. This has multiple advantages:
  
  - A less vague definition of detuning convention.
  - Precise definition of energy ordering: couplings now always point from lower to higher absolute energy.
  - More flexibility in decoherence. Decoherent transions now point "from" one state "to" another rather than just "between" 2 states. This fixes a limitation where gamma matrices no longer must be lower triangular.

- `get_snr()` function in `rq.experiments` now takes `kappa` and `eta` as optional arguments to allow for running on any `Sensor` object. They can still be inferred from a `Sensor` subclass that has them as attributes if unspecified.
- time solver now properly handles complex time dependences in the rotating wave approximation
- Added type hints to code base that can be used to static type check with mypy
- Added functions `rq.calc_kappa` and `rq.calc_eta` to properly calculate kappa and eta constants for experimental parameters.
- Added function `rq.get_OD` that calculates the optical depth of a solution
- Improved accuracy of the solver memory estimates
- Increased input validation unit test coverage
- Generalized handling of transit broadening to allow for multiple repopulation states with varying branching ratios

Bug Fixes
++++++++++++
- Fixed an issue with time dependence in the probe laser
- Modified solver to allow for complex time dependence
- Fixed non-hermitian hamiltonians in time solver
- Fixed error with multiple time-dependences in time solver
- Added functionality to solver error with complex time dependences
- Modified experimental return functions (`get_transmission_coef()`, `get_phase_shift()`, and `get_susceptibility()``) to allow scanning of probe rabi frequency
- Fixed `get_rho_ij` so that it correctly calculates the `(0,0)` population element
- Fix error in `test_sensor_management` which fails if temporary directory does not exist.
- Tighten `test_decoherences` tolerances to the 2pi*100Hz level to catch errors in decoherence matrix generation.
- Fixed issue where `get_snr` ignored the optical path length input parameter
- Fixed issue where calling `solve_steady_state` with `sum_doppler=False` would double memory footprint.
- Fixed issue where `solve_steady_state` could be called with `weight_doppler=False` and `sum_doppler=True`.

Deprecations
++++++++++++

- `get_snr` no longer allows manually specifying `Sensor.eta` and `Sensor.kappa`, these values must be passed as args for Sensor input
- Removed unused `gamma_transit` argument from Sensor init
- Re-ordered argument list to `Cell.add_coupling` to match order of `Sensor.add_coupling`
- `Sensor.add_fields` has been fully removed and no longer works as a deprecated alias of `Sensor.add_couplings`

v0.3.0
------

Improvements
++++++++++++

- Expanded documention
- Removed restrictions on ARC and numpy versions during installation.
- Vectorized equation of motion generation to support prepending axes to a hamiltonian
- Updated the internal mechanism for sensor handling fields of various type

  - Fields are now internally called couplings
  - Fields are specified as either having rabi_frequency or transition_frequency, corresponding to RWA or non-RWA fields
  - Fields are specified as either having detuning or transition_frequency, corresponding to steady-state or time-dependent fields
  - Fields with specific traits can be accessed with the `couplings_with()` function

- Added a feature to save/load sensors/cells
- Implemented NumbaKitODE which considerably speeds up solve_time. This feature can be enabled by setting parameter compile=True of solve_time.
- Improved logic for building diagonal terms of Hamiltonian using NetworkX graph library that allows for diagonal terms to be built from any set of values.
- Generalized doppler averaging to support prepended axes on hamiltonians.
- Improved time solver logic for improved modularity across doppler solving and multivalue parameters.
- Added a feature to draw level diagram
- Seamlessly generate all Hamiltonians from lists of parameters in sensor.
- Added ability to label couplings.
- Added capability to make any coupling time-dependent
- Sped up time solving considerably by simultaneously solving all equations rather than looping.
- Allow for user to specify fields by beam power, beam waist, and electric field, in the Cell framework.
- Solve functions now return a bunch-type object rather than a tuple.
- Added functionality that breaks equations into slices based on memory requirements
- Quantum numbers and absolute energies are now stored on the nodes of a Cell couplings graph
- Cell now adds decay rates and decoherences to the nodes and edges of the Cell couplings graph
- Cell now calculates the gamma matrix in an arbitrary way, and is no longer limited to two laser, ladder schemes
- Added function to calculate sensor SNR with repect to any varied sensor coupling parameter
- Added function to return sensor parameter mesh

Bug Fixes
+++++++++

- Fixed example notebook.
- Fixed issue where doppler averaging breaks if there are uncoupled levels.
- Fixed doppler averaging so that doppler shifts are applied with signs consistent with the hamiltonian.
- Fixed a bug where doppler averaging did not properly solve separately for each doppler class.
- Fixed issue where spatial dimension of doppler averaging is not introspected correctly in the presence of round-off errors.

Deprecations
++++++++++++

- All "field" functionality are being deprecated in favor of "coupling"
- The `rf_couplings`, `target_state`, and `rf_dipole_matrix` arguments of `solve_time()`
- All functions relating to sensor.transtion_map are deprecated
- Cell now does not accept gamma_excited or gamma_Rydberg as these are always calculated or Sensor can be used with a given gamma matrix
- Cell now does not accept  gamma_doppler as Doppler broadening width is given by mutiplying the most proable velocity and the laser k-vector

v0.2.0
------

Beta release. Contains very large number of backwards-incompatible changes over alpha release.

v0.1.0
------

Alpha release. Minimum viable product release that does basic modeling tasks slowly.
