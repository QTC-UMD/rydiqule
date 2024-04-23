Installation
============

Installation is done via pip or conda.
See below for detailed instructions.

In all cases, it is highly recommended to install rydiqule in a virtual environment.

Conda installation
------------------

Installation via conda is recommended for rydiqule.
It handles dependency installation as well as a virtual environment to ensure packages do not conflict with other usages on the same system.
Finally, the `numpy` provided by anaconda has been compiled against optimized BLAS/LAPACK implementations, which results in much better performance in rydiqule itself.

Assuming you have not already created a separate environment for RydIQule (recommended), run the following to create a new environment:

.. code-block:: shell

  (base) ~/> conda create -n rydiqule python=3.11
  (base) ~/> conda activate rydiqule

RydIQule currently requires python >3.8.
For a new installation, it is recommended to use the newest supported python.

Now install via rydiqule's anaconda channel.
This channel provides rydiqule as well as its dependencies that are not available in the default anaconda channel.
If one of these dependencies is outdated, please raise an issue with the 
`vendoring repository <https://github.com/QTC-UMD/rydiqule-vendored-conda-builds>`_.

.. code-block:: shell

  (rydiqule) ~/> conda install -c rydiqule rydiqule


If you would like to install rydiqule in editable mode to locally modify its source,
this must be done using pip.
Follow the above to install rydiqule and its dependencies,
then run the following to uninstall rydiqule as provided by conda
and install the editable local repository.

.. code-block:: shell

  (rydiqule) ~/> conda remove rydiqule --force
  # following must be run from root of local repository
  (rydiqule) ~/> pip install -e .

Note that editable installations require `git`.
This can be provided by a system-wide installation or via conda in the virtual environment (`conda install git`).

.. note::

    While rydiqule is a pure python package (ie it is platform independent), its core dependency ARC is not.
    If a pre-built package of ARC is not available for your platform in our anaconda channel,
    you will need to install ARC via `pip` to build it locally before installing `rydiqule`.
    To see what architectures are supported, please see the 
    `vendoring repository <https://github.com/QTC-UMD/rydiqule-vendored-conda-builds>`_.

Pure pip installation
---------------------

To install normally, run:

.. code-block:: shell

  pip install rydiqule

This command will use pip to install all necessary dependencies.

To install in an editable way (which allows edits of the source code), 
run the following from the root directory of the cloned repository:

.. code-block:: shell

  pip install -e .

Editable installtion requires `git` to be installed.

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

For conda installations, run the following command to upgrade rydiqule

.. code-block:: shell

  conda upgrad rydiqule

For `pip`, you can use the same installation command to upgrade.
Optionally, include the update flag to greedily update dependencies as well.

.. code-block:: shell

  pip install -U rydiqule

This command will also install any new dependencies that are required.

If using an editable install, simply replacing the files in the same directory is sufficient.
Though it is recommended to also run the appropriate pip update command as well to capture updated depedencies.

.. code-block:: shell

  pip install -U -e .


Dependencies
------------

This package requires installation of the excellent `ARC <https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator>`_ 
package, which is used to get Rydberg atomic properties. 
It also requires other standard computation dependenices, such as `numpy`, `scipy`, `matplotlib`, etc.
These will be automatically installed if not already present.

.. note::

    Rydiqule's performance does depend on these depedencies.
    In particular, `numpy` can be compiled with a variety of backends that implements
    BLAS and LAPACK routines that can have different performance for different computer architectures.
    When using Windows, it is recommended to install `numpy` from conda,
    which is built against the IntelMKL and has generally shown the best performance for Intel-based PCs.

Optional timesolver backend dependencies include the `numbakit-ode <https://github.com/hgrecco/numbakit-ode>`_
and `CyRK <https://github.com/jrenaud90/CyRK>`_ packages.
Both are available via `pip`.
They can be installed automatically via the optional extras specification for the `pip` command.

.. code-block:: shell

  pip install rydiqule[backends]

For conda installations, these dependencies must be installed manually

.. code-block:: shell

  conda install -c rydiqule CyRK numbakit-ode