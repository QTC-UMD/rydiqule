"""
Bunch-like object use to store aspects of a solution when calling rydiule.solve()
Adds essential keys with "None" entries
"""
from __future__ import annotations
from typing import Optional

import copy

import numpy as np

# have to import this way to prevent circular imports
from rydiqule import sensor_utils


class Solution(dict):
    """
    Manual implementation of a bunch object which fuctions as a dictionary with
    the ability to access elements.

    For now, little additional funcitonality exists
    on top of this, but some may be added in the future.
    """
    # common attributes
    rho: np.ndarray
    """numpy.ndarray : Solutions returned by the solver."""
    eta: Optional[float]
    """float, optional : Eta constant from the Cell.
    Not generally defined when using a Sensor."""
    kappa: Optional[float]
    """float, optional : Kappa constant from the Cell.
    Not generally defined when using a Sensor."""
    couplings: dict
    """dict : Dictionary of the couplings."""
    axis_labels: list[str]
    """list of str : Labels for the axes of scanned parameters.
    If doppler averaging but not summing, doppler dimensions are prepended."""
    axis_values: list
    """list : Value arrays corresponding to each axis.
    If doppler averaging but not summing, doppler classes in internal units are added."""
    rq_version: str
    """str : Version of rydiqule that created the Solution."""
    basis: list[str]
    """list of str: The list of density matrix elements in the order they appear in the solution.
    See :meth:`Sensor.basis` for details."""

    # doppler specific
    doppler_classes: Optional[np.ndarray]
    """numpy.ndarray, optional : Doppler classes used to perform the doppler average.
    Will be None if doppler averaging was not used."""

    # time_solver specific
    t: np.ndarray
    """numpy.ndarray : Times the solution is returned at, when using the time solver.
    Undefined otherwise."""
    init_cond: np.ndarray
    """numpy.ndarray : Initial conditions, when using the time solver.
    Undefined otherwise."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__ = self

    def rho_ij(self, i: int, j: int) -> np.ndarray:
        """
        Gets the i,j element(s) of the density matrix solutions.

        See :func:`~.get_rho_ij` for details.

        Parameters
        ----------
        i: int
            density matrix element `i`
        j: int
            density matrix element `j`

        Returns
        -------
        numpy.ndarray
            `[i,j]` elments of the density matrix
        """

        return sensor_utils.get_rho_ij(self.rho, i, j)

    def copy(self):
        return copy.copy(self)
    
    def deepcopy(self):
        return copy.deepcopy(self)