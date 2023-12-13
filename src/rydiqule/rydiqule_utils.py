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


def about(obscure_paths: bool = True):
    """About box describing Rydiqule and its core dependencies.

    Prints human readable strings of information about the system.

    Parameters
    ----------

    obscure_paths: bool, optional
        Remove user directory from printed paths. Default is True.

    Examples
    --------
    >>> import rydiqule as rq
    >>> rq.about()
    <BLANKLINE>
            Rydiqule
        ================
    <BLANKLINE>
    Rydiqule Version:     0.4.0
    Installation Path:    C:\\~\\rydiqule\\src\\rydiqule
    <BLANKLINE>
        Dependencies
        ================
    <BLANKLINE>
    NumPy Version:        1.21.5
    SciPy Version:        1.7.3
    Matplotlib Version:   3.5.2
    ARC Version:          3.2.1
    Python Version:       3.9.12
    Python Install Path:  C:\\~\\miniconda3\\envs\\arc
    Platform Info:        Windows (AMD64)
    CPU Count:            16
    Total System Memory:  256 GB

    """
    home = Path.home()
    install_path = inspect.getsourcefile(rydiqule)
    assert install_path is not None
    rydiqule_install_path = Path(install_path).parent
    try:
        ryd_path = '~' / rydiqule_install_path.relative_to(home)
    except ValueError:
        ryd_path = rydiqule_install_path

    python_install_path = Path(sys.executable).parent
    try:
        py_path = '~' / python_install_path.relative_to(home)
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
    print(f'CPU Count:            {os.cpu_count()}')
    print(f'Total System Memory:  {psutil.virtual_memory().total/1024**3:.0f} GB')


if __name__ == "__main__":
    about()
