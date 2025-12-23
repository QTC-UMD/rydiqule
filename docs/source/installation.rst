Installation
============

Installation is done via pip or conda.
See below for detailed instructions.

In all cases, it is highly recommended to install rydiqule in a virtual environment.

Installation via conda is recommended for rydiqule.
It handles dependency installation as well as a virtual environment to ensure packages do not conflict with other usages on the same system.
Finally, the `numpy` provided by anaconda has been compiled against optimized BLAS/LAPACK implementations,
which results in much better performance in rydiqule itself.

.. note::

  RydIQule currently requires python >3.8.
  For a new installation, it is recommended to use the newest supported python.
  Currently supported versions are |pythons|

Regular Installation
--------------------

.. tab:: conda

  Assuming you have not already created a separate environment for RydIQule (recommended), run the following to create a new environment:

  .. code-block:: shell

    (base) ~/> conda create -n rydiqule python=3.11
    (base) ~/> conda activate rydiqule

  Now install via rydiqule's anaconda channel.
  This channel provides rydiqule as well as its dependencies that are not available in the default anaconda channel.
  If one of these dependencies is outdated, please raise an issue with the 
  `vendoring repository <https://github.com/QTC-UMD/rydiqule-vendored-conda-builds>`_.

  .. code-block:: shell

    (rydiqule) ~/> conda install -c rydiqule rydiqule

.. tab:: pip

  To install normally, run:

  .. code-block:: shell

    pip install rydiqule

  This command will use pip to install all necessary dependencies.

.. tab:: uv

  The `uv` package and project manager allows for unified project management in a way
  that is reproducible and easy to share.
  You create a new simulation project that relies on `rydiqule` by running the following commands.

  .. code-block:: shell

    uv init new-project
    cd new-project
    uv venv --python 3.12  # this line optionally sets python version to use for the venv

  These commands create a project template in the sub-directory `new-project`.
  This directory is readily version controlled and contains all information needed to reproduce your environment.

  You finish configuring the project by adding required dependencies, starting with `rydiqule`.

  .. code-block:: shell

    uv add rydiqule

  You can now create a script in the project and run it directly:

  .. code-block:: shell

    uv run example.py

  You can also add jupyter kernel support and run jupyter notebooks from VS Code.
  This support is added as a development dependency.

  .. code-block:: shell

    uv add --dev ipykernel

  Once added, VS Code can launch a jupyter notebook using the project's virtual environment
  located in the `/new-project/.venv/` sub-directory.

  You can also run jupyter lab directly from the virtual environment with the following command

  .. code-block:: shell

    uv run --with jupyter jupyter lab

Editable Installation
---------------------

If you would like to install rydiqule in editable mode to locally modify its source,
use the following commands.

.. tab:: conda

  Follow the above to install rydiqule and its dependencies,
  then run the following to uninstall rydiqule as provided by conda
  and install the editable local repository.

  .. code-block:: shell

    (rydiqule) ~/> conda remove rydiqule --force
    # following must be run from root of local repository
    (rydiqule) ~/> pip install -e .

.. tab:: pip
 
  Run the following from the root directory of the cloned repository:

  .. code-block:: shell

    pip install -e .

.. tab:: uv

  Using `uv` on an existing python package automatically installs it in editable mode.
  Run the following from the root of the local repository.

  .. code-block:: shell

    uv venv --python 3.12  # this line optionally sets a python version for the venv
    uv sync

  Note that `uv sync` automatically installs the `dev` dependency group which includes `ipykernel` and `pytest`.

  You can now use VS Code to run jupyter notebooks with the virtual environment at `rydiqule/.venv/`.
  To use jupyter lab, run the following command

  .. code-block:: shell

    uv run --with jupyter jupyter lab

Note that editable installations should have `git` available if you want dynamic versioning (via `setuptools-scm`),
either by a system-wide installation or via conda in the virtual environment (`conda install git`).

.. note::

    While rydiqule is a pure python package (ie it is platform independent), its core dependency ARC is not.
    If a pre-built package of ARC is not available for your platform in our anaconda channel,
    you will need to install ARC via `pip` to build it locally before installing `rydiqule`.
    To see what architectures are supported, please see the 
    `vendoring repository <https://github.com/QTC-UMD/rydiqule-vendored-conda-builds>`_.


Confirm installation
--------------------

Proper installation can be confirmed by executing the following commands in a python terminal.

.. code-block:: shell

  >>> import rydiqule as rq
  >>> rq.about()

          Rydiqule
      ================

  Rydiqule Version:     1.1.0
  Installation Path:    ~\Miniconda3\envs\rydiqule\lib\site-packages\rydiqule

        Dependencies
      ================

  NumPy Version:        1.24.3
  SciPy Version:        1.10.1
  Matplotlib Version:   3.7.1
  ARC Version:          3.3.0
  Python Version:       3.9.16
  Python Install Path:  ~\Miniconda3\envs\rydiqule
  Platform Info:        Windows (AMD64)
  CPU Count:            12
  Total System Memory:  128 GB

Updating an existing installation
---------------------------------

Upgrading an existing installation is simple.
Simply run the appropriate upgrade command for the installation method used.

Regular Installation Upgrade
++++++++++++++++++++++++++++

.. tab:: conda

  .. code-block:: shell

    conda upgrade rydiqule

.. tab:: pip

  .. code-block:: shell

    # standard upgrade
    pip install rydiqule
    # greedy upgrade: ie update dependencies too
    pip install -U rydiqule

.. tab:: uv

  .. code-block:: shell

    uv lock --upgrade-package rydiqule
    # greedy upgrade
    uv lock --upgrade

Editable Installation Upgrade
+++++++++++++++++++++++++++++

If using an editable install, simply replacing the files in the same directory is sufficient.
Though it is recommended to also run the appropriate pip update command as well to capture updated dependencies.

.. tab:: conda

  .. code-block:: shell

    pip install -U -e .

.. tab:: pip

  .. code-block:: shell

    pip install -U -e .

.. tab:: uv

  .. code-block:: shell

    uv sync

  Note that any `uv run` command will automatically sync and thereby capture updated dependencies.

Dependencies
------------

This package requires installation of the excellent `ARC <https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator>`_ 
package, which is used to get Rydberg atomic properties. 
It also requires other standard computation dependencies, such as `numpy`, `scipy`, `matplotlib`, etc.
These will be automatically installed if not already present.

.. note::

    Rydiqule's performance does depend on these dependencies.
    In particular, `numpy` can be compiled with a variety of backends that implement
    BLAS and LAPACK routines that can have different performance for different computer architectures.
    When using Windows, it is recommended to install `numpy` from the conda default channel,
    which is built against the IntelMKL and has generally shown the best performance for Intel-based PCs.

Optional timesolver backend dependencies include the `numba`
and `CyRK <https://github.com/jrenaud90/CyRK>`_ packages.
Both are available via `pip`, `conda`, or our anaconda channel.

.. tab:: conda

  For conda installations, these dependencies must be installed manually

  .. code-block:: shell

    conda install -c rydiqule CyRK

.. tab:: pip

  Backends can be installed automatically via the optional extras specification for the `pip` command.

  .. code-block:: shell

    pip install rydiqule[backends]

.. tab:: uv

  Backends can be installed automatically via the optional extras specification for the `uv sync` command.

  .. code-block:: shell

    uv sync --extra backends

  Note that these dependencies will be uninstalled if `uv sync` is called without the extras flag.

.. |pythons| image:: https://img.shields.io/pypi/pyversions/rydiqule.svg
