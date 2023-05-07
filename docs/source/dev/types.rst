Type Hinting
============

Rydiqule employs the `optional type hinting capabilities of python <https://peps.python.org/pep-0484/>`_.
These type annotations are not checked or enforced at runtime by python itself.
Rather, they provide hints to fellow programmers and users about the types of function arguments, return types, and class variables.

We use the `mypy <https://mypy.readthedocs.io/en/stable/>`_ static type checking library to read these hints and catch type errors within the code base.
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