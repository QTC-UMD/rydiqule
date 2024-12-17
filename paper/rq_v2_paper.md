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
affiliations:
  - index: 1
    name: DEVCOM Army Research Laboratory, 2800 Powder Mill Rd, Adelphi, MD, 20783, USA
    ror: 011hc8f90

date: 11 Dec 2024
bibliography: paper.bib
aas-doi: 
aas-journal:
---
# Summary

The Rydberg Interactive Quantum Module (RydIQule) is a Python package which uses a novel paradigm of representing semiclassical quantum systems as graphs, with nodes representing quantum states and edges representing couplings between them. It uses this intuitive representation to generate system dynamic equations, then wraps an efficient differential equation solver in to produce solutions to these equations in a single function call at the top level. While the initial release of RydIQule in late 2023 built this core functionality, its version 2 release significantly expands on its capability to model more detailed real-world atoms. In this paper we highlight these key improvements, and discuss the new types of systems whose modelling they enable.

# Statement of Need

The unique quantum properties of Rydberg atoms offer distinct advantages in the fields of sensing, communication, and quantum information which make them distinct from any classical analogue [@adams_rydberg_2019]. The initial release of RydIQule in early 2024 [@miller_rydiqule_2024] built the foundational structure of generating differential equations from graph. While technically capable of modelling a huge variety of systems, it was missing functionality both in easy of use and capabilities in some more complex systems. The version 2 release of RydIQule introduces a new paradigm for labeling states by their quantum number, which trivializes creation systems with huge numbers of states, and can automatically calculate dipole moments for allowed transitions in real atoms such as Rubidium and Cesium. 

Many atomic systems have their levels divided into magnetic sublevels, which enable types of experiments not possible without them (example?). While many such systems could in principle be modeled with RydIQule's initial release, practical limitations would arise from accounting for systems which could quickly reach dozens or even hundreds of quantum states. For example, in accounting for nuclear magnetic splitting, users would have no choice but to individually add each state and coupling, making RydIQule of little functional use for modelling such systems. The primary goal of RydIQle version 2 is to allow modelling such effects with little additional code beyond what was required to solve very simple systems in Version 1. 

# Handling Sublevel Structure

RydIQule's primary improvements in version 2, and the focus of this paper, are in handling state manifolds: degenerate states split by a magnetic interaction with the electron. In RydIQule, we are interested in the splitting created by either orbital effects or the nuclear magnetic moment. In the latter case especially, single states can often be broken into a dozen or more individual sublevels which all must be accounted for to get accurate results. 

RydIQule v2 handles this sublevel structure by expanding the way nodes are labelled. Rather than only using integers, tuples of quantum numbers numbers can now be used as graph nodes. All of RydIQule's core functions relating to graph operations can now interchangeably use individual states or entire manifolds by using lists of quantum numbers. When lists, which we call specifications, are used, they are automatically interpreted as all corresponding states individually. RydIQule's internals have been overhauled to not only ensure that all relevant states/couplings are added, but tracked as originating from a single manifold.

# Related Packages and Work

RydIQule, especially in its v2 realease, is designed to capture the functionality of the Atomic Density Matrix (ADM) package [@rochester_atomicdensitymatrix_nodate] written in Mathematica. While very capable, and used by many labs as a modelling aid, it suffers from a couple of limitations. Firstly, it is built on a proprietary platform requiring a paid license. Second, since mathematica is a completely interpreted language, it lacks the speed that complied libraries like numpy enable, especially when exploring a large parameter space.

Calculation of physical properties of real atoms is handled by the ARC Rydberg package [@sibalic_arc_2017, @robertson_arc_2021]. Once calculated, RydIQule populates these values where appropriate onto its own graph model.

Additionally, RydIQule version 1 has been used as a modelling aid in several recent publications [@santamaria-botello_comparison_2022, @elgee_satellite_2023, @richardson_study_2023, @backes_performance_2024, @su_two-photon_2024, @gokhale_deep_2024]

# Acknowledgements

We acknowledge funding from our employers, the Army Research Lab in Adelphi, MD.