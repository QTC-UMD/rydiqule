![Rydiqule Logo](docs/source/img/Rydiqule_Logo_Transparent_300.png)

The Rydberg Interactive Quantum module is a modelling library designed to simulate
the response of a Rydberg atoms to arbitrary input RF waveforms.
It also functions as a general master equation solver based on the semi-classical density matrix method.

## Installation

Presently, installation must be done manually using a copy of the repository.
After downloading or cloning this repository,
follow the directions below for installation.

### Pure pip installation

To install in an editable way (which allows edits of the source code), run:
```shell
pip install -e .
```
from within the top level `rydiqule` directory (i.e. where the `setup.cfg` file resides).
This command will use pip to install all necessary dependencies.

To install normally, run:
```shell
pip install .
```
from the same directory.

### Conda/pip installation

If using a conda environment, it can often be advantageous to install as many packages via conda before running the pip installation command.

Assuming you have not already created a separate environment for RydIQule (recommended), run the following to create a new environment:
```shell
(base) ~/Rydiqule> conda create -n rydiqule python=3.9
(base) ~/Rydiqule> conda activate rydiqule
```
RydIQule currently requires python >3.8.

Now install dependencies that are available via conda.
To capture as many dependencies as possible,
we will add the `conda-forge` channel at a lower priority.
```shell
(rydiqule) ~/Rydiqule> conda config --append channels conda-forge
(rydiqule) ~/Rydiqule> conda config --set channel_priority strict
(rydiqule) ~/Rydiqule> conda install numpy scipy matplotlib networkx psutil
# ARC specific dependencies available via conda
(rydiqule) ~/Rydiqule> conda install sympy asteval lmfit uncertainties
```

Now use pip to install rydiqule and remaining dependencies.
```shell
# for editable installation, so source can be modified locally
(rydiqule) ~/Rydiqule> pip install -e .
# for normal installation
(rydiqule) ~/Rydiqule> pip install .
```

### Confirm installation

Proper installation can be confirmed by executing the following commands in a python terminal.
```shell
>>> import rydiqule as rq
>>> rq.about()

        Rydiqule
    ================

Rydiqule Version:     0.3.0
Installation Path:    c:\users\naqsl\src\rydiqule\src\rydiqule

      Dependencies
    ================

NumPy Version:        1.20.3
SciPy Version:        1.7.1
Matplotlib Version:   3.5.0
ARC Version:          3.2.0
Python Version:       3.8.12
Python Install Path:  C:\Users\naqsL\Miniconda3\envs\arc
Platform Info:        Windows (AMD64)
CPU Count:            12
Total System Memory:  128 GB
```

### Updating an existing installation

Upgrading an existing installation is simple.
Simply run the pip installation commands described above with the update flag.
```shell
pip install -U .
```
This command will also install any new dependencies that are required.

If using an editable install, simply replacing the files in the same directory is sufficient.
Though it is recommended to also run the appropriate pip update command as well.
```shell
pip install -U -e .
```

### Dependencies

This package requires installation of the excellent [ARC](https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator) package, which is used to get Rydberg atomic properties.
It also requires other standard computation dependenices, such as `numpy`, `scipy`, `matplotlib`, etc.
These dependencies will be automatically installed by pip if not already present.

The timesolver backend dependencies include the `numbakit-ode` and `CyRK` packages.
Both are available via `pip`.

## Documentation

Documentation is available in the `docs\build` directory.
A precompiled version of the pdf documentation is available in `docs\build\latex\rydiqule.pdf`.

### Examples

Example jupyter notebooks that demonstrate RydIQule can be found in the `examples` subdirectory.
Printouts of these notebooks are available in the documentation as well.

## Support

Creation of this software was supported in part by the Defense Advanced Research Projects Agency (DARPA) Quantum Apertures program, DEVCOM Army Research Laboratory, and the Quantum Technology Center at the University of Maryland.

## Disclaimer

The views, opinions and/or findings expressed are those of the authors and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S.~Government.