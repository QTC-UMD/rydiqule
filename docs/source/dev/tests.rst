Unit Tests
==========

Rydiqule comes bundled with a suite of unit tests that confirm basic functionality of the various components
and checks for robustness to erroneous or unexpected arguments.
We strive to follow the testing methodology and practices employed by `numpy <https://numpy.org/doc/stable/reference/testing.html>`_.
We agree with the stipulation made there, that 

  "Long experience has shown that by far the best time to write the tests is before you write or change the code - 
  this is `test-driven development <https://en.wikipedia.org/wiki/Test-driven_development>`_"

pytest
******

Rydiqule takes advantage of the `pytest` testing framework to run unit and integration tests of the code base.
The full test suite is run from the project base directory with the command

.. code-block:: shell
  
  pytest

This command will run all tests in the `tests/` subdirectory as well as docstring examples.
These tests cover a wide range of functionality as well as a number of representative integrations that demonstrate how the code can be used to generate end results.

Marks
-----

The tests are marked based on the type of test that is being performed, and `pytest` can be told to only run certain tests.
For example, this command will only run tests relating to the steady state solving functionality:

.. code-block:: shell

  pytest -m steady_state

You can also exclude a specific group of tests.
For example, this command will exlcude tests marked as slow.

.. code-block:: shell

  pytest -m "not slow"

Marks specifications can be combined using standard boolean keywords as well.
The following will run all the time tests that are not slow.

.. code-block:: shell

  pytest -m "time and not slow"

The available marks can be listed using `pytest --markers`.
The markers we use are

.. list-table:: Markers
  :widths: 25 100
  :header-rows: 1

  * - Marker
    - Description
  * - slow
    - Marks a test as taking a long time to run
  * - high_memory
    - Marks a test needing a lot of RAM
  * - steady_state
    - Marks a test as using the steady-state solver
  * - time
    - Marks a test as using the time solver
  * - doppler
    - Marks a test that incorporates Doppler averaging.
  * - experiments
    - Marks a test that represents a full experiment.
  * - util
    - Marks a test of the ancillary utilties.
  * - structure
    - Marks a test of the definition of the atomic system.
  * - exception
    - Marks a test of error handling.
  * - dev
    - Used to temporarily mark a single test that is being developed so it can run independently.

Docstring Rounding
------------------

Pytest also runs all examples in the docstrings using python's built-n `doctest` module.
This module runs the provided code and does simple string matching of the output to confirm results.
Given rydiqule is a numerical computation package that deals heavily with floating point numbers,
there will be small differences in results based on platform and even exact dependency libraries used.
To prevent spurious errors, doctest has been configured to ignore digits of higher precision in the result
than what is provided in the source.
For example, a computed result of `9.333453` will successfully match against the docstring value of `9.333`.

Coverage
--------

If you install the `pytest-cov` plugin, you can check code coverage of the tests by modifying the command to read.

.. code-block:: shell

  pytest --cov=rydiqule

Durations
---------

If you want to see which tests take the longest to complete, you can use the `--durations=n` flag to give the `n` longest time tests:

.. code-block:: shell

  pytest --durations=3

Settings the `durations` flag to 0 will cause pytest to report the time taken for all tests run.
