---
title: "RydIQule Version 2: Enhancing Graph-Based modelling of Rydberg atoms"
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
  - name: Kevin C. Cox
    orcid: 0000-0001-5049-3999
    affiliation: 1
affiliations:
  - index: 1
    name: DEVCOM Army Research Laboratory, 2800 Powder Mill Rd, Adelphi, MD, 20783, USA
    ror: 011hc8f90

date: 17 December 2024
bibliography: paper.bib
aas-doi: 
aas-journal:
---
# Summary

The Rydberg Interactive Quantum Module (RydIQule) is a Python package which uses a novel paradigm of representing quantum systems as NetworkX graphs [@SciPyProceedings_11],
with nodes representing quantum states and edges representing couplings between them.
From this representation it generates equations of motion that govern the interaction between the quantum system and classical field perturbations,
using the Lindblad master equation formalism [@manzano_short_2010].
Finally, RydIQule leverages general linear equation solvers, such as those provided by NumPy [@harris_array_2020], SciPy [@virtanen_scipy_2020] or CyRK [@cyrk]
to efficiently solve these systems and recover the quantum system response to arbitrary input fields.
Internally, RydIQule heavily relies on array broadcasting techniques to efficiently parameterize and solve many systems of equations as collective tensor objects.

The initial target problem space for RydIQule was modeling of Rydberg electric field sensors [@schlossberger_rydberg_2024].
These sensors are based on thermal ensembles of alkali atoms
that are excited to Rydberg quantum states using multiple laser fields.
Atoms in these high-energy electronic states have a large response to ambient electric fields,
providing a massively wideband response spanning from quasi-DC to a THz [@fancher_rydberg_2021].
Creating a flexible, performant forward model for these types of sensors allows for rapid iteration and exploration of system designs and fundamental atomic physics.
Ultimately, the theoretical tools required to model these systems are common to many quantum platforms and applications,
allowing for RydIQule to be useful to a much wider range of common problems in quantum science.

The initial public release of RydIQule in late 2023 built the core functionality described above [@miller_rydiqule_2024],
but some necessary functionality for readily handling the problems of the wider community was missing.
Here we outline RyIQule's version 2 release which significantly expands on its capability to model more detailed real-world atoms.
In this paper we highlight the key improvements necessary to handle the more arbitrary and complex structure of general quantum systems.

# Statement of Need

The unique quantum properties of Rydberg atoms offer distinct advantages in the fields of sensing, communication,
and quantum information which make them distinct from any classical analogue [@adams_rydberg_2019].
However, the breadth of possible configurations and experimental parameters makes general modeling of an experiment difficult.
The initial release of RydIQule in late 2023 [@miller_rydiqule_2024] built the foundational structure of generating differential equations from a computational graph.
Though this design is highly flexible and technically capable of modeling a huge variety of systems,
it required a significant amount of manual book-keeping by the user to track the many complexities of real atoms (compared with simplified models).
In particular, many atomic energy levels consist of magnetic sublevels, which are normally degenerate (ie have the same energy) and therefore are not resolved.
In many cases, this sublevel structure can be treated in average and ignored.
But these sublevels have different responses to applied magnetic and electric fields which leads measureable differences for even moderately strong fields.
In order to accurately model an atomic system response to truly arbitrary fields, accounting for this sublevel structure is necessary.

RydIQule's initial release could, in principle, handle these complicated systems,
but practical limitations would arise as system sizes quickly reach dozens or even hundreds of quantum states.
For example, in accounting for nuclear magnetic splitting,
the $5\text{S}_{1/2}$ ground state of rubidium 85 breaks into 12 sublevels divided between two manifolds.
Users would have no choice but to individually add each sublevel and the many associated couplings,
making RydIQule of little functional use.

The primary goal of RydIQle version 2 is to allow modeling such systems with little additional code beyond what was required to solve very simple systems in version 1. 
This release introduces a new paradigm for structured labeling of states using arbitrary tuples,
and greatly improves the automated calculation of relevant atomic properties on alkali atoms commonly used in Rydberg physics.

## Handling Sublevel Structure

RydIQule's primary improvement in version 2,
is in handling state manifolds: degenerate sets of sublevel states defined by a magnetic interaction with the electron.
RydIQule v2 handles this sublevel structure by expanding the way nodes are labelled.
Rather than only using integers, tuples of numbers can now be used as graph nodes.
When lists, which we call specifications, are used within a tuple, they are automatically interpreted as all corresponding states individually
(ie `(0, [-1, 0, 1])` maps to the group `[(0,-1), (0, 0), (0, 1)]`).
RydIQule's core functions relating to graph operations have been updated to interchangeably use individual states or entire manifolds.
RydIQule's internals have been overhauled to not only ensure that all relevant states/couplings are added, but tracked as originating from a single manifold.

## Improved Calculation of Atomic Properties

Version 2 of RydIQule also completely overhauls the `Cell` class, which provides automatic calculation of atomic properties of alkali atoms
using the Alkali.ne Rydberg Calculator (ARC) package [@sibalic_arc_2017, @robertson_arc_2021].
In version 1, this class could only handle simplified atomic models that treated manifolds of atomic sublevels as a single approximate state.
Though this type of model is very fast and can be effective in many situtations,
it breaks down for systems in the presence of magnetic fields (including those as weak as Earth's background magnetic field)
or for large electric field amplitudes that result in inhomogeneous couplings due to sublevel structure.
With the native ability to specify states using tuples, `Cell` can now define states by their quantum numbers directly,
which allows for easy definition and coupling of manifolds.
Version 2 also greatly enhances the leveraging of ARC to calculate more system parameters automatically.
In particular, there is automatic calculation of coupling strengths between manifolds defined in incommensurate fine and hyperfine bases.
This feature allows for more efficient modeling of Rydberg atoms since low energy and high energy states can be defined in their natural bases,
fine and hyperfine respectively, lowering the total number of degenerate sublevels that need to be calculated.

# Related Packages and Work

Modeling quantum systems using the semi-classical Lindblad formalism is a common task that has been implemented by many physicists for their bespoke problems.
Other tools that implement this type of simulation for specific types of problems; including qubits in QuTiP [@johansson_qutip_2013], atomic magnetometers in ElecSus [@keaveney_elecsus_2018], and laser cooling in PyLCP [@eckel_pylcp_2020].
Ultimately, the goal of RydIQule has not been to develop a new modeling technique,
but rather to make a common, flexible, and most importantly efficient tool that solves a ubiquitous problem.

RydIQule's version 2 release is specifically designed to capture the functionality of the Atomic Density Matrix (ADM) package [@rochester_atomicdensitymatrix_nodate] written in Mathematica.
While very capable, it suffers from a couple of limitations.
Firstly, it is built on a proprietary platform requiring a paid license.
Second, since mathematica is an interpreted language,
it lacks the speed that complied libraries like NumPy enable, especially when exploring a large parameter space.

Since RydIQule version 1 has been publically released,
it has been used in several recent publications to model both general Rydberg atom physics [@backes_performance_2024, @su_two-photon_2024,]
as well as Rydberg sensor development [@santamaria-botello_comparison_2022, @elgee_satellite_2023, @richardson_study_2023, @gokhale_deep_2024].

# Acknowledgements

Financial support for the development of RydIQule version 2 was provide by the DEVCOM Army Research Laboratory.
The views, opinions and/or findings expressed are those of the authors and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government.