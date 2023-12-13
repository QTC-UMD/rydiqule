Installation
============

Presently, installation is done via pip.
See below for detailed instructions.

In all cases, it is highly recommended to install rydiqule in a virtual environment.

Conda/pip installation
----------------------

If using a conda to provide the virtual environmnet,
it can often be advantageous to install as many packages as possible via conda before running the pip installation command.

Assuming you have not already created a separate environment for RydIQule (recommended), run the following to create a new environment:

.. code-block:: shell

  (base) ~/> conda create -n rydiqule python=3.9
  (base) ~/> conda activate rydiqule

RydIQule currently requires python >3.8.

Now install dependencies that are available via conda.
To capture as many dependencies as possible,
we will add the `conda-forge` channel at a lower priority.

.. code-block:: shell

  (rydiqule) ~/> conda config --env --append channels conda-forge
  (rydiqule) ~/> conda config --set channel_priority strict
  (rydiqule) ~/> conda install numpy scipy matplotlib networkx numba psutil
  # ARC specific dependencies available via conda
  (rydiqule) ~/> conda install sympy asteval lmfit uncertainties


Now use pip to install rydiqule and remaining dependencies.

.. code-block:: shell

  # for normal installation
  (rydiqule) ~/Rydiqule> pip install rydiqule
  # for editable installation of cloned repo, so source can be modified locally
  (rydiqule) ~/Rydiqule> pip install -e .

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
Simply run the pip installation commands described above.
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
These will be automatically installed by pip if not already present.

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