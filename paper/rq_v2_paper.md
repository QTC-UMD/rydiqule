---
title: "RydIQule Version 2: Enhancing Graph-Based modeling of Rydberg atoms"
tags:
  - Python
  - Rydberg
  - Atomic Physics
  - Quantum Sensing
  - Quantum
  - Graph
authors:
  - name: Benjamin N. Miller
    orcid: 0000-0003-0017-1355
    equal-contrib: true
    affiliation: 1 
  - name: David H. Meyer
    orcid: 0000-0003-2452-2017
    equal-contrib: true
    affiliation: 1
  - name: Omar Nagib
    affiliation: 2
  - name: Teemu Virtanen
    orcid:
    affiliation: 3
  - name: Kevin C. Cox
    orcid: 0000-0001-5049-3999
    affiliation: 1
affiliations:
  - index: 1
    name: DEVCOM Army Research Laboratory, 2800 Powder Mill Rd, Adelphi, MD, 20783, USA
    ror: 011hc8f90
  - index: 2
    name: Department of Physics, University of Wisconsin-Madison, 1150 University Avenue, Madison, WI, 53706, USA
    ror: 01y2jtd41
  - index: 3
    name: Naval Air Warfare Center, 1 Administration Circle, China Lake, CA, 93555, USA
    ror: 01f0pxq13

date: 27 February 2025
bibliography: paper.bib
aas-doi: 
aas-journal:
---
# Summary
<!--- This is an HTML comment in Markdown -->
<!--- EACH SENTENCE SHOULD BE A NEWLINE --->
Rydberg atomic radio-frequency (rf) sensors are an emerging technology platform that relies on vaporous atoms, interrogated with laser beams and nearly ionized, to receive rf signals.
Rydberg rf sensors have a number of interesting fundamental distinctions from traditional receiver technologies,
such as those based on metallic antennas,
since they are goverened by the quantum physics of atom-light interactions [@fancher_rydberg_2021].
As Rydberg sensors quickly advance from laboratory experiments into fieldable devices,
there is a need for a general software modeling tool that fully encompasses the internal physics of the sensor.
The Rydberg Interactive Quantum Module (RydIQule) is a Python package designed to fill this need.

RydIQule calculates the dynamics of atomic quantum states,
with a particular emphasis on modeling thermal vapors of Rydberg atoms coupled to optical and radio-frequency electromagnetic fields.
To accomplish this, a unique graph-based paradigm is used to represent the complex quantum system consisting of multiple energy levels and multiple electromagnetic fields,
where each energy level is stored as a graph node and each electromagnetic coupling as a graph edge.
RydIQule then generates a set of differential equations for the quantum state dynamics from the graph,
using the Lindblad master equation formalism [@manzano_short_2010].
Finally, RydIQule leverages linear equation solvers,
such as those provided by NumPy [@harris_array_2020], SciPy [@virtanen_scipy_2020] or CyRK [@cyrk]
to efficiently solve these systems and recover the quantum system response to arbitrary input fields.
During the numerical solving, systems of equations are represented as tensor objects,
allowing for efficient parameterization and computation of large sets of equations.
All together, RydIQule provides a flexible platform for forward modeling Rydberg sensors while also providing a widely useful set of theoretical tools for fundamental exploration of atomic physics concepts.

The initial public release of RydIQule in late 2023 built the core functionality described above [@miller_rydiqule_2024].
Here we outline RyIQule's version 2 release which expands on its capability to more accurately model real-world atoms.

# Statement of Need

The unique quantum properties of Rydberg atoms offer distinct advantages in the fields of sensing, communication,
and quantum information [@adams_rydberg_2019].
However, the breadth of possible configurations and experimental parameters makes general modeling of an experiment difficult.
One challenge is that many atomic energy levels consist of numerous magnetic sublevels that arise from the different possible orientations of the electron's and nucleus's angular momentum.
These sublevels have different responses to applied magnetic and electric fields which leads measureable differences for most real-world atomic sensors.
In some cases, this sublevel structure can be treated in average and ignored.
More often, they are ignored due to the significant complexity inherent in expanding the model size to account for them.

For example, in accounting for nuclear magnetic splitting,
the commonly-used $5\text{S}_{1/2}$ ground state of rubidium 85 breaks into 12 magnetic sublevels divided between two manifolds of degerate sublevels.
In RydIQule's initial release, users would have no choice but to individually add each sublevel and the many associated electromagnetic couplings, making it of little functional use.
For this reason, RydiQule was not easily scaled to realistic scenarios involving several atomic states and typically many tens, or possibly even hundreds, of sublevels.

The main advance of RydIQule version 2 is to allow user-friendly inclusion of large atomic manifolds that include the complete set of electronic and magnetic sublevels.
In particular, this release introduces a new paradigm for structured labeling of states using arbitrary tuples,
and expands the automated calculation of relevant atomic properties on alkali atoms commonly used in Rydberg physics
to include sublevels.
This release also includes a new steady-state Doppler-averaging method that improves speed and accuracy,
along with many other optimizations and improvements to the code-base.

## Handling Sublevel Structure

RydIQule's primary improvement in version 2 is in handling state manifolds: degenerate sets of sublevel states defined by a magnetic interaction with the electron.
It handles this sublevel structure by expanding the way nodes are labelled.
Rather than only using integers, tuples of numbers can now be used as graph nodes.
When lists, which we call specifications, are used within a tuple, they are automatically interpreted as all corresponding states individually
(ie `(0, [-1, 0, 1])` maps to the group `[(0,-1), (0, 0), (0, 1)]`).
RydIQule's core functions relating to graph operations have been updated to interchangeably use individual states or entire manifolds.
Its internals have been overhauled to not only ensure that all relevant states/couplings are added, but tracked as originating from a single manifold.

## Improved Calculation of Atomic Properties

Version 2 of RydIQule also completely overhauls the `Cell` class, which provides automatic calculation of atomic properties of alkali atoms
using the Alkali.ne Rydberg Calculator (ARC) package [@sibalic_arc_2017; @robertson_arc_2021].
In version 1, this class could only handle simplified atomic models that treated manifolds of atomic sublevels as a single approximate state.
Though this type of model is very fast and can be effective in many situtations,
it breaks down for systems in the presence of magnetic fields (including those as weak as Earth's background magnetic field)
or for large electric field amplitudes that result in inhomogeneous couplings due to sublevel structure.
With the native ability to specify states using tuples, `Cell` can now define states by their quantum numbers directly,
which allows for natural definition and coupling of manifolds.

Version 2 also greatly enhances the leveraging of ARC to calculate more system parameters automatically.
In particular, there is automatic calculation of coupling strengths between manifolds defined in incommensurate fine and hyperfine bases.
This feature allows for more efficient modeling of Rydberg atoms since low energy and high energy states can be defined in their natural bases,
fine and hyperfine respectively, lowering the total number of sublevels that need to be calculated.

## Analytic Doppler Averaging

Experimental support for Doppler-averaged models using an exact analytic solution has been added.
This functionality is based on the theoretical work presented in [@nagib_exact_2025].
That work derived a method for solving the Linblad master equation in the eigenbasis of the system,
which allows for a separation of the Doppler-averaging integration from system diagonalization enabling a general, analytic result.
Replacing RydIQule's approximate numeric integration with this exact method results in faster and significantly higher accuracy solutions.
At present, only 1-dimensional Doppler averages are supported, with extensions to higher dimensions planned for the next minor release.

# Related Packages and Work

Modeling quantum systems using the semi-classical Lindblad formalism is a common task that has been implemented by many physicists for their bespoke problems.
Other tools that implement this type of simulation for specific types of problems include: qubits in QuTiP [@johansson_qutip_2013], atomic magnetometers in ElecSus [@keaveney_elecsus_2018], and laser cooling in PyLCP [@eckel_pylcp_2020].
Ultimately, the goal of RydIQule has not been to develop a new modeling technique,
but rather to make a common, flexible, and most importantly efficient tool that solves a ubiquitous problem.

RydIQule's version 2 release aims to capture the functionality of the Atomic Density Matrix (ADM) package [@rochester_atomicdensitymatrix_2008] written in Mathematica.
While very capable,
it is built on a proprietary platform requiring a paid license which limits its accessibility.
And since Mathematica is an interpreted language,
it can lack the speed that complied libraries like NumPy enable, especially when exploring a large parameter space.

Since RydIQule version 1 has been publically released,
it has been used in several publications to model both general Rydberg atom physics [@backes_performance_2024; @su_two-photon_2024]
as well as Rydberg sensor development [@santamaria-botello_comparison_2022; @elgee_satellite_2023; @richardson_study_2023; @gokhale_deep_2024].

# Acknowledgements

Financial support for the development of RydIQule version 2 was provide by the DEVCOM Army Research Laboratory.
The views, opinions and/or findings expressed are those of the authors and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government.
