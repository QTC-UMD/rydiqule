<img src="https://raw.githubusercontent.com/QTC-UMD/rydiqule/main/docs/source/img/Rydiqule_Logo_Transparent_300.png" alt="rydiqule" style="max-width: 100%;">

The Rydberg Interactive Quantum module is a modeling library designed to simulate
the response of Rydberg atoms to arbitrary input RF waveforms.
It also functions as a general master equation solver based on the semi-classical density matrix method.

[![PyPI](https://img.shields.io/pypi/v/rydiqule.svg)](https://pypi.org/project/rydiqule)
[![Conda Version](https://img.shields.io/conda/v/rydiqule/rydiqule)](https://anaconda.org/rydiqule/rydiqule)
[![Python Version](https://img.shields.io/pypi/pyversions/rydiqule.svg)](https://python.org)
[![License](https://img.shields.io/pypi/l/rydiqule.svg)](https://github.com/QTC-UMD/rydiqule/raw/main/LICENSE)
[![Docs](https://readthedocs.org/projects/rydiqule/badge/?version=latest)](https://rydiqule.readthedocs.io/en/latest)

### Please cite as

B. N Miller, D. H. Meyer, T. Virtanen, C. M O'Brien, and K. C. Cox,
RydIQule: A Graph-based paradigm for modeling Rydberg and atomic sensors,
*Computer Physics Communications*, **294**, 108952 (2024)
[https://doi.org/10.1016/j.cpc.2023.108952](https://doi.org/10.1016/j.cpc.2023.108952)

## Installation

Installation is done via pip or conda.
See below for detailed instructions.

In all cases, it is highly recommended to install rydiqule in a virtual environment.

### Conda installation (recommended)

Installation via conda is recommended for rydiqule.
It handles dependency installation as well as a virtual environment to ensure packages do not conflict with other usages on the same system.
Finally, the `numpy` provided by anaconda has been compiled against optimized BLAS/LAPACK implementations, which results in much better performance in rydiqule itself.

Assuming you have not already created a separate environment for RydIQule (recommended), run the following to create a new environment:
```shell
(base) ~/> conda create -n rydiqule python=3.11
(base) ~/> conda activate rydiqule
```
RydIQule currently requires python >3.8.
For a new installation, it is recommended to use the newest supported python.

Now install via rydiqule's anaconda channel.
This channel provides rydiqule as well as its dependencies that are not available in the default anaconda channel.
If one of these dependencies is outdated, please raise an issue with the [vendoring repository](https://github.com/QTC-UMD/rydiqule-vendored-conda-builds).
```shell
(rydiqule) ~/> conda install -c rydiqule rydiqule
```

If you would like to install rydiqule in editable mode to locally modify its source,
this must be done using pip.
Follow the above to install rydiqule and its dependencies,
then run the following to uninstall rydiqule as provided by conda
and install the editable local repository.
```shell
(rydiqule) ~/> conda remove rydiqule --force
# following must be run from root of local repository
(rydiqule) ~/> pip install -e .
```

Note that editable installations require `git`.
This can be provided by a system-wide installation or via conda in the virtual environment (`conda install git`).

While rydiqule is a pure python package (ie it is platform independent), its core dependency ARC is not.
If a pre-built package of ARC is not available for your platform in our anaconda channel,
you will need to install ARC via `pip` to build it locally before installing `rydiqule`.
To see what architectures are supported, please see the [vendoring repository](https://github.com/QTC-UMD/rydiqule-vendored-conda-builds).


### Pure pip installation

To install normally, run:
```shell
pip install rydiqule
```
This command will use pip to install all necessary dependencies.

To install in an editable way (which allows edits of the source code),
run the following from the root directory of the cloned repository:
```shell
pip install -e .
```

Editable installation requires `git` to be installed.

### Confirm installation

Proper installation can be confirmed by executing the following commands in a python terminal.
```shell
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
```

### Updating an existing installation

Upgrading an existing installation is simple.
Simply run the appropriate upgrade command for the installation method used.

For conda installations, run the following command to upgrade rydiqule
```shell
conda upgrade rydiqule
```

For `pip`, you can use the installation command to upgrade.
Optionally, include the update flag to greedily update dependencies as well.
```shell
pip install -U rydiqule
```
This command will also install any new dependencies that are required.

If using an editable install, simply replacing the files in the same directory is sufficient.
Though it is recommended to also run the appropriate pip update command as well to capture updated dependencies.
```shell
pip install -U -e .
```

### Dependencies

This package requires installation of the excellent [ARC](https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator) package, which is used to get Rydberg atomic properties.
It also requires other standard computation dependenices, such as `numpy`, `scipy`, `matplotlib`, etc.
These dependencies will be automatically installed if not already present.

Rydiqule's performance does depend on these depedencies.
In particular, `numpy` can be compiled with a variety of backends that implements
BLAS and LAPACK routines that can have different performance for different computer architectures.
When using Windows, it is recommended to install `numpy` from conda,
which is built against the IntelMKL and has generally shown the best performance for Intel-based PCs.

Optional timesolver backend dependencies include the [numbakit-ode](https://github.com/hgrecco/numbakit-ode)
and [CyRK](https://github.com/jrenaud90/CyRK) packages.
Both are available via `pip` or our anaconda channel.
They can be installed automatically via the optional extras specification for the `pip` command.
```shell
pip install rydiqule[backends]
```

For conda installations, these dependencies must be installed manually.
```shell
conda install -c rydiqule CyRK numbakit-ode
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