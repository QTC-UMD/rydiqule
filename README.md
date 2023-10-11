<img src="https://raw.githubusercontent.com/QTC-UMD/rydiqule/main/docs/source/img/Rydiqule_Logo_Transparent_300.png" alt="rydiqule" style="max-width: 100%;">

The Rydberg Interactive Quantum module is a modelling library designed to simulate
the response of a Rydberg atoms to arbitrary input RF waveforms.
It also functions as a general master equation solver based on the semi-classical density matrix method.

[![PyPI](https://img.shields.io/pypi/v/rydiqule.svg)](https://pypi.org/project/rydiqule)
[![Python Version](https://img.shields.io/pypi/pyversions/rydiqule.svg)](https://python.org)
[![License](https://img.shields.io/pypi/l/rydiqule.svg)](https://github.com/QTC-UMD/rydiqule/raw/main/LICENSE)
[![Docs](https://readthedocs.org/projects/rydiqule/badge/?version=latest)](https://rydiqule.readthedocs.io/en/latest)

## Installation

Presently, installation is done via pip.
See below for detailed instructions.

In all cases, it is highly recommended to install rydiqule in a virtual environment.

### Conda/pip installation

If using a conda to provide the virtual environmnet,
it can often be advantageous to install as many packages as possible via conda before running the pip installation command.

Assuming you have not already created a separate environment for RydIQule (recommended), run the following to create a new environment:
```shell
(base) ~/> conda create -n rydiqule python=3.9
(base) ~/> conda activate rydiqule
```
RydIQule currently requires python >3.8.

Now install dependencies that are available via conda.
To capture as many dependencies as possible,
we will add the `conda-forge` channel at a lower priority.
```shell
(rydiqule) ~/> conda config --env --append channels conda-forge
(rydiqule) ~/> conda config --set channel_priority strict
(rydiqule) ~/> conda install numpy scipy matplotlib networkx numba psutil
# ARC specific dependencies available via conda
(rydiqule) ~/> conda install sympy asteval lmfit uncertainties
```

Now use pip to install rydiqule and remaining dependencies.
```shell
# for normal installation
(rydiqule) ~/> pip install rydiqule
# for editable installation, so source can be modified locally
(rydiqule) ~/> pip install -e rydiqule
```

### Pure pip installation

To install normally, run:
```shell
pip install rydiqule
```
This command will use pip to install all necessary dependencies.

To install in an editable way (which allows edits of the source code), run:
```shell
pip install -e rydiqule
```

### Confirm installation

Proper installation can be confirmed by executing the following commands in a python terminal.
```shell
>>> import rydiqule as rq
>>> rq.about()

        Rydiqule
    ================

Rydiqule Version:     1.0.0
Installation Path:    C:\Users\naqsL\Miniconda3\envs\rydiqule\lib\site-packages\rydiqule

      Dependencies
    ================

NumPy Version:        1.24.3
SciPy Version:        1.10.1
Matplotlib Version:   3.7.1
ARC Version:          3.3.0
Python Version:       3.9.16
Python Install Path:  C:\Users\naqsL\Miniconda3\envs\rydiqule
Platform Info:        Windows (AMD64)
CPU Count:            12
Total System Memory:  128 GB
```

### Updating an existing installation

Upgrading an existing installation is simple.
Simply run the pip installation commands described above with the update flag.
```shell
pip install -U rydiqule
```
This command will also install any new dependencies that are required.

If using an editable install, simply replacing the files in the same directory is sufficient.
Though it is recommended to also run the appropriate pip update command as well.
```shell
pip install -U -e rydiqule
```

### Dependencies

This package requires installation of the excellent [ARC](https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator) package, which is used to get Rydberg atomic properties.
It also requires other standard computation dependenices, such as `numpy`, `scipy`, `matplotlib`, etc.
These dependencies will be automatically installed by pip if not already present.

Rydiqule's performance does depend on these depedencies.
In particular, `numpy` can be compiled with a variety of backends that implements
BLAS and LAPACK routines that can have different performance for different computer architectures.
When using Windows, it is recommended to install `numpy` from conda,
which is built against the IntelMKL and has generally shown the best performance for Intel-based PCs.

Optional timesolver backend dependencies include the [numbakit-ode](https://github.com/hgrecco/numbakit-ode)
and [CyRK](https://github.com/jrenaud90/CyRK) packages.
Both are available via `pip`.
They can be installed automatically via the optional extras specification for the `pip` command.
```shell
pip install rydiqule[backends]
```

## Documentation

Documentation is available online at [readthedocs](https://rydiqule.readthedocs.io/en/latest).
PDF or EPUB formats of the documentation can be downloaded from the online documentation.

### Examples

Example jupyter notebooks that demonstrate RydIQule can be found in the `examples` subdirectory.
Printouts of these notebooks are available in the documentation as well.

## Support

Creation of this software was supported in part by the Defense Advanced Research Projects Agency (DARPA) Quantum Apertures program, DEVCOM Army Research Laboratory, and the Quantum Technology Center at the University of Maryland.

## Disclaimer

The views, opinions and/or findings expressed are those of the authors and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government.

## Contact

The github repository is for code distribution only.
While we monitor it, 
we will not directly respond to issues or pull requests posted to it.
If you would like a response from the developers, please e-mail
david.h.meyer3.civ@army.mil or kevin.c.cox29.civ@army.mil