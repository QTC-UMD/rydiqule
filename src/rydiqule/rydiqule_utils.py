"""
General rydiqule package utilities
"""

import platform
import rydiqule
import inspect
import os
import sys
import psutil
from pathlib import Path
from importlib.metadata import version


def about(obscure_paths: bool = True,
          show_numpy_config: bool = True):
    """About box describing Rydiqule and its core dependencies.

    Prints human readable strings of information about the system.

    Parameters
    ----------

    obscure_paths: bool, optional
        Remove user directory from printed paths. Default is True.
    show_numpy_config: bool, optional
        Show the numpy config for BLAS/LAPACK backends. Default is True.

    Examples
    --------
    >>> rq.about() # doctest: +SKIP
    <BLANKLINE>
            Rydiqule
        ================
    <BLANKLINE>
    Rydiqule Version:     2.1.1
    Installation Path:    ~\\rydiqule\\src\\rydiqule
    <BLANKLINE>
          Dependencies
        ================
    <BLANKLINE>
    NumPy Version:        2.2.5
    SciPy Version:        1.16.0
    Matplotlib Version:   3.10.0
    ARC Version:          3.9.0
    Python Version:       3.11.10
    Python Install Path:  ~\\miniconda3\\envs\\arc
    Platform Info:        Windows (AMD64)
    CPU Count:            16 @ 3.91 GHz
    Total System Memory:  256 GB
    <BLANKLINE>
         NumPy backends
        ================
    <BLANKLINE>
    blas: provided by mkl-sdl (2023.1)
    lapack: provided by mkl-sdl (2023.1)
    """
    home = Path.home()
    install_path = inspect.getsourcefile(rydiqule)
    assert install_path is not None
    rydiqule_install_path = Path(install_path).parent
    try:
        if obscure_paths:
            ryd_path = '~' / rydiqule_install_path.relative_to(home)
        else:
            ryd_path = rydiqule_install_path
    except ValueError:
        ryd_path = rydiqule_install_path

    python_install_path = Path(sys.executable).parent
    try:
        if obscure_paths:
            py_path = '~' / python_install_path.relative_to(home)
        else:
            py_path = python_install_path
    except ValueError:
        py_path = python_install_path

    header = """
        Rydiqule
    ================
    """
    print(header)
    print(f'Rydiqule Version:     {rydiqule.__version__:s}')
    
    print(f'Installation Path:    {ryd_path}')
    dep_header = """
      Dependencies
    ================
    """
    print(dep_header)
    print(f'NumPy Version:        {version("numpy"):s}')
    print(f'SciPy Version:        {version("scipy"):s}')
    print(f'Matplotlib Version:   {version("matplotlib"):s}')
    print(f"ARC Version:          {version('arc-alkali-rydberg-calculator'):s}")
    print(f'Python Version:       {platform.python_version():s}')
    print(f'Python Install Path:  {py_path}')
    print(f'Platform Info:        {platform.system():s} ({platform.machine():s})')
    print(f'CPU Count and Freq:   {os.cpu_count():d} @ {psutil.cpu_freq().current*1e-3:.2f} GHz')
    print(f'Total System Memory:  {psutil.virtual_memory().total/1024**3:.0f} GB')

    if show_numpy_config:
        print('\n     NumPy backends\n    ================\n')
        import numpy
        config = numpy.show_config('dicts')
        for dep, info in config['Build Dependencies'].items():
            print(f"{dep}: provided by {info['name']} ({info['version']})")


if __name__ == "__main__":
    about()
