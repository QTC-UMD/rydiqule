
.. image:: img/Rydiqule_Logo_Transparent_300.png
   :width: 800

.. raw:: html

   <div style="visibility: hidden;">

rydiqule
========

.. raw:: html

   </div></div>

A python library for calculating Rydberg electrometer response to arbitrary RF fields in steady-state or time domains.
It is a general density matrix-based master equation solver,
optimized for speed to solve problems with large parameter spaces
while maintaining flexibility to define novel problems.
It leverages a graph-based system definition,
computationally-efficient equation "stacking" in the form of tensors,
and external computational libraries such as `numpy`, `scipy`, and `ARC`.

For more details, see the :doc:`overview`.

For detailed usage examples, see the :doc:`_intro_nbs/Introduction_To_Rydiqule/Introduction_To_Rydiqule` Jupyter notebook.

.. toctree::
   :maxdepth: 2
   :hidden:
   :glob:
   :caption: GETTING STARTED

   installation
   overview
   _intro_nbs/**/*
   changelog   

.. toctree::
   :maxdepth: 2
   :glob:
   :hidden:
   :caption: DETAILED DOCUMENTATION

   writeups/writeups_index
   api/api_index
   dev/dev_index

.. toctree::
   :maxdepth: 2
   :hidden:
   :glob:
   :caption: EXAMPLE NOTEBOOKS

   _examples/**/*

.. todolist::
