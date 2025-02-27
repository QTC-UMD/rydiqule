
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

For detailed usage examples, see the :doc:`intro_nbs/Introduction_To_Rydiqule` Jupyter notebook.

If you use rydiqule in your work, please cite as

.. raw:: html

   <details>
     <summary>B. N. Miller, <em>et. al.</em>, <u><a href="https://doi.org/10.1016/j.cpc.2023.108952">RydIQule: A Graph-based paradigm for modeling Rydberg and atomic sensors</a>,</u> <em>Computer Physics Communications</em> <b>294</b>, 108952 (2024). arXiv:<a href="http://arxiv.org/abs/2307.15673">2307.15673</a>.</summary>

.. code-block:: bibtex

   @article{rydiqule_2024,
      author = {Miller, B. N. and Meyer, D. H. and Virtanen, T. and O'Brien, C. M. and Cox, K. C.},
      title = {RydIQule: A Graph-based paradigm for modeling Rydberg and atomic sensors},
      journal = {Computer Physics Communications},
      volume = {294},
      pages = {108952},
      year = {2024},
      doi = {10.1016/j.cpc.2023.108952},
      url = {https://doi.org/10.1016/j.cpc.2023.108952},
      eprint = {https://doi.org/10.1016/j.cpc.2023.108952}
   }

.. raw:: html

   </details>

.. toctree::
   :maxdepth: 2
   :hidden:
   :glob:
   :caption: GETTING STARTED

   installation
   overview
   intro_nbs/Introduction_To_Rydiqule
   intro_nbs/Cell_Basics
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

   examples/*

.. todolist::
