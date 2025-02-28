Building the Documentation
==========================

This section describes how to build the documentation locally.
A web-hosted copy of the docs is avialable at `https://rydiqule.readthedocs.io/ <https://rydiqule.readthedocs.io/>`_
and should generally be used.
These instructions are provided for local testing and development purposes.

The Rydiqule documentation is built locally from the source repository using `sphinx`.
To do so, you will need to install the `docs` optional dependencies.

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

Given the difficulty of building this type of documentation locally, 
a copy can be downloaded from the `documentation website. <https://rydiqule.readthedocs.io/_/downloads/en/stable/pdf/>`_

latex
-----

It is also possible to build the pdf docs in stages, allowing for more control of the process.
This is also how the docs are built on readthedocs, allowing for more accurate reproduction of results there.

First, build the latex for the docs pdf.

.. code-block:: shell

  make latex

Then change directory to the `docs/build/latex` directory and run the following `latexmk` command.

.. code-block:: shell

  latexmk -r latexmkrc -pdf -f -dvi- -ps- -jobname=rydiqule -interaction=nonstopmode

This workflow largely recreates the `latexpdf` workflow, but invokes options that ensure errors do not stop the build.

epub
----

There is also the ability to build the documentation in the EPUB format, if desired.

.. code-block:: shell

  make epub

This version of the documentation is also available for download on `ReadTheDocs <https://rydiqule.readthedocs.io/_/downloads/en/stable/epub/>`_.