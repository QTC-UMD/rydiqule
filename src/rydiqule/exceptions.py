"""
Rydiqule Custom Exceptions and Warnings

This module defines a number of custom Errors and Warnings for use in Rydiqule.

It ensures that all RydiquleWarnings are always shown.
This behavior can be changed by calling
:external+python:func:`warnings.simplefilter`.

It also defines custom exception handlers to hide the
raise statement associated with RydiquleErrors.
This behavior can be suppressed by calling
:func:`rq.debug_state(True) <debug_state>`.
"""

import warnings
import traceback
import sys


###### Rydiqule Exception Definitions
class RydiquleError(Exception):
    """A *rydiqule* error
    
    Indicates a Rydiqule-specific error.
    It is a thin wrapper around Exception.
    """
    pass

class AtomError(RydiquleError):
    """An error in interacting with ARC"""
    pass

class CouplingNotAllowedError(RydiquleError):
    """Indicated coupling is not allowed (eg dipole-forbidden)"""
    pass

# setup concealing of final frame in RydiquleError tracebacks unless in DEBUG mode
DEBUG = False

def set_debug_state(state: bool):
    """Controls DEBUG state of rydiqule.
    
    Parameters
    ----------
    state: bool
        If True, full error tracebacks for RydiquleErrors will be shown.
        If False (default behavior), final raise statement is suppresed in the traceback.
    """

    global DEBUG

    DEBUG = state

def debug_state():
    """Returns current rydiqule debug state
    
    Returns
    -------
    bool
        Current debug state
    """

    return DEBUG

# suppress final frame of RydiquleError tracebacks for normal usage
# ensure we use current excepthook as fallback, in case someone has overrideen already
current_excepthook = sys.excepthook
def quiet_exception_handler(exception_type, exception, tb):

    if issubclass(exception_type, RydiquleError) and not DEBUG:

        tb_counter = 1
        tb_han = tb
        while tb_han.tb_next is not None:
            tb_counter += 1
            tb_han = tb_han.tb_next
        # -1 strips off the final raise statement from the traceback
        traceback.print_exception(exception_type, exception, tb, limit=tb_counter-1)
    else:
        current_excepthook(exception_type, exception, tb)

sys.excepthook = quiet_exception_handler

# suppress RydiquleError tracebacks in IPython (ie Jupyter)
try:
    from IPython import get_ipython

    ipython = get_ipython()
    if ipython is not None:
        # don't patch if IPython isn't being used

        def exception_handler(self, etype, evalue, tb, tb_offset=None):
            if not DEBUG:
                tb_han = tb
                # assumes at least 2 levels to the traceback
                while tb_han.tb_next.tb_next is not None:
                    tb_han = tb_han.tb_next
                # stop the traceback at the 2nd to last level
                tb_han.tb_next = None
            # standard IPython's printout
            self.showtraceback((etype, evalue, tb), tb_offset=tb_offset)  

        ipython.set_custom_exc((RydiquleError,), exception_handler)

        del ipython, get_ipython
except ImportError:
    # if ipython not available, don't patch
    pass


###### Rydiqule Warning Definitions
class RydiquleWarning(UserWarning):
    """Indicates a rydiqule-specfic warning.
    
    All other rydiqule warnings derive from this class.

    Disable globally by calling
    :external+python:func:`warnings.simplefilter('ignore', rq.RydiquleWarning) <warnings.simplefilter>`
    or locally using the :external+python:class:`warnings.catch_warnings` context manager.
    """
    pass

class RWAWarning(RydiquleWarning):
    """Indicates the coupling is using a large transition frequency outside the rotating wave approximation.
    """
    pass

class PopulationNotConservedWarning(RydiquleWarning):
    """Indicates population will not be conserved in the model."""
    pass

class TimeDependenceWarning(RydiquleWarning):
    """Indicates a time-dependent coupling is being used in a steady-state context"""
    pass

class NLJMWarning(RydiquleWarning):
    """Indicates `Cell` has likely been called with the old state specification [n, l, j, m]
    for a solve that did not intend to use fine-structure magnetic sublevels.
    """
    pass

# Ensure Rydiqule warnings are always raised
# can be overridden by user after import
warnings.filterwarnings('always', category=RydiquleWarning, append=True)
