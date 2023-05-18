Changelog
=========

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
