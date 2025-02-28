Type Hinting
============

Rydiqule employs the `optional type hinting capabilities of python <https://peps.python.org/pep-0484/>`_.
These type annotations are not checked or enforced at runtime by python itself.
Rather, they provide hints to fellow programmers and users about the types of function arguments, return types, and class variables.

We use the `pyright <https://microsoft.github.io/pyright/#/>`_ static type checking library to read these hints and catch type errors within the code base.
The easiest way to use this checker locally is via the VS Code Pylance extensions,
which defaults to the pyright type checker and will automatically run when configuration options are detected in the root `pyproject.toml`.

Note that python is still a dynamic, duck-typed language,
and rydiqule employs some features that are perfectly valid python which are not expressable in the type hinting system.
In these situations, we err on the side of type hints being primarily documentation,
and do our best to not obfuscate functioning code merely to satisfy the type checker.

We also have configurations for using the `mypy <https://mypy.readthedocs.io/en/stable/>`_ static type checking library.
To run this check locally, install the `mypy` python package and run the following command from the package root directory.

.. code-block:: shell

    mypy

This command will automatically read configuration options set in the `mypy.ini` file.
Further optional flags can be passed to the command to override or add optional behaviors.
Initial run of the mypy takes some time, however subsequent runs take advantage of local caching to increase analysis speed.
Using the mypy daemon mode can further increase analysis speed if necessary.

An html report of the mypy coverage can be generated using the following command.

.. code-block:: shell

    mypy --html-report .mypy_report

This command will store the html pages in the specified directory `.mypy_report`.
Note that this command takes a long time to run every time, as it cannot use the cache.