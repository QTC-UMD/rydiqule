Building the Documentation
==========================

The Rydiqule documentation can be built locally from the source repository using `sphinx`.
To do so, you will need to install the `sphinx` and `sphinx-rtd-theme` packages.

html
----

An html webpage version of the documentation formatted in the read-the-docs style can be made by running the following command from the `docs/` subdirectory.

.. code-block:: shell

  make html

The output will be located in the `docs/build/html/` subdirectory.
The home page is `index.html`.
The html documentation has the best formatting by default and is the easiest to use.

latexpdf
--------

A pdf version of the documentation can be built using

.. code-block:: shell

  make latexpdf

The output will be located in `docs/build/latex` and is called `rydiqule.pdf`.
Note that building the pdf requires `perl` and a functioning latex installation with the `latexmk` package.
You will also require the GNU FreeFont collection.
On Windows, these can be `installed manually at the system level <https://www.gnu.org/software/freefont/>`_
or via the MikTeX package `gnu-freefont`.
This build also requires a great many other latex packages in addition to `latexmk`.
It is easiest to install these packages on the fly as needed, if your latex distribution supports that.

Given the difficulty of building this type of documentation, 
we attempt to include an updated pdf with each relase.
It is locaed in the `docs\build\latex` directory.

epub
----

There is also the ability to build the documentation in the EPUB format, if desired.

.. code-block:: shell

  make epub