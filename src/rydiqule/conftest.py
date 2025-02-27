import pytest
import rydiqule
import numpy

@pytest.fixture(autouse=True)
def add_rq(doctest_namespace):
    doctest_namespace["rq"] = rydiqule

@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace["np"] = numpy
