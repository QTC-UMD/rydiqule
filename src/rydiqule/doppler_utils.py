"""
Utilities for implementing Doppler averaging
"""

import numpy as np
from scipy import special

from .sensor_utils import _hamiltonian_term, make_real, remove_ground
from .exceptions import RydiquleError

from typing import Tuple, Optional, TypedDict, Union, Literal, Sequence
from typing_extensions import Required


class UniformMethod(TypedDict, total=False):
    method: Required[Literal['uniform']]
    width_doppler: float
    n_uniform: int


class IsoPopMethod(TypedDict, total=False):
    method: Required[Literal['isopop']]
    n_isopop: int


class SplitMethod(TypedDict, total=False):
    method: Required[Literal['split']]
    width_doppler: float
    n_doppler: int
    width_coherent: float
    n_coherent: int


class DirectMethod(TypedDict, total=True):
    method: Literal['direct']
    doppler_velocities: Union[np.ndarray, Sequence]


MeshMethod = Union[UniformMethod, IsoPopMethod, SplitMethod, DirectMethod]


def get_doppler_equations(base_eoms: np.ndarray,
                          doppler_hamiltonians: np.ndarray, Vs: np.ndarray, ground_removed : bool = True) -> np.ndarray:
    """
    Returns the equations for each slice of the doppler profile.

    A new axes corresponding to these slices are appended to the beginning.
    For example, if equations are of shape `(m,m)`
    and there are `n_doppler` doppler values being sampled,
    the return will be of shape `(n_doppler, m, m)`.

    Parameters
    ----------
    base_eoms : numpy.ndarray
        Stacked square arrays representing the unshifted equations,
        i.e. the theoretical equations for an ensemble of atoms with zero momentum.
    doppler_hamiltonians : numpy.ndarray
        Arrays of hamiltonians with only doppler shifts present.
        One for each spatial dimension needed.
        See :meth:`~.Sensor.get_doppler_shifts` for details.
    Vs : numpy.ndarray
        Mesh of velocity classes to sample,
        with same spatial dimensions as `dop_ham`.
        See :func:`~.doppler_mesh` for details.
    ground_removed : bool, optional
        Whether to remove the ground state from the equations.
        Default is True.

    Returns
    -------
    numpy.ndarray
        An array of shape `(*Vs.shape[1:], *base_eoms.shape)` which is a,
        potentially multi-dimensional, stack of individual equations of shape `(m, m)`.
        Each slice of this stack is an equation of shape `(m, m)`
        with the corresponding doppler shifts applied.

    Note
    ----
    Each doppler shift is equal to `k_i*vP*det_i`, in units of Mrad/s,
    where `i` denotes the einstein summation along the spatial dimensions.
    `det` is the normalized velocity class, with `vP*det_i=v_i` giving the velocity.
    `vP` is the most probable speed from the Maxwell-Boltzmann distribution:
    sqrt(2*kB*T/m).
    `k_i` is the k-vector of the field along the same axis as `det_i`.
    `doppler_hamiltonians` provides `k_i*vP`, `Vs` provides `det_i`.

    """
    base_doppler_shift_eoms = generate_doppler_shift_eom(doppler_hamiltonians, ground_removed=ground_removed)
    # take outer product of velocities and base kvec eoms to get detunings.
    # Add shifts from each spatial dimension to each detuning
    doppler_shift_eoms = np.tensordot(Vs, base_doppler_shift_eoms, ((0),(0)))
    # broadcast up base_eom to add doppler shift dims
    # broadcast up doppler_shift_eoms to add base_eom stack dimensions
    n_stacks = len(base_eoms.shape[:-2])
    spatial_dim = base_doppler_shift_eoms.shape[0]
    exp_dims = tuple(range(spatial_dim, spatial_dim+n_stacks))
    doppler_eqns = np.expand_dims(base_eoms, 0) + np.expand_dims(doppler_shift_eoms, exp_dims)

    return doppler_eqns


def generate_doppler_shift_eom(doppler_hamiltonians: np.ndarray, ground_removed : bool = True) -> np.ndarray:
    """
    Generates the EOMs for the supplied doppler shifts.

    Multiply the output by the velocity in each dimension,
    then add to the normal EOMs to get the full Doppler shifted EOMs.

    Parameters
    ----------
    doppler_hamiltonians : numpy.ndarray
        Hamiltonians of only the doppler shifts,
        one for each spatial dimension to be averaged over.
    ground_removed : bool, optional
        Whether to remove the ground state from the equations.
        Default is True.

    Returns
    -------
    numpy.ndarray
        Corresponding LHS EOMs with ground removed and in the real basis.

    """
    obes = _hamiltonian_term(doppler_hamiltonians)
    if ground_removed:
        obes, const = remove_ground(obes)
        doppler_shift_obes = make_real(obes, const)[0]
    else:
        n_squared = doppler_hamiltonians.shape[-1]**2
        const = np.zeros(n_squared)
        doppler_shift_obes = make_real(obes, const, ground_removed=False)[0]

    return doppler_shift_obes


def gaussian3d(Vs: np.ndarray) -> np.ndarray:
    """
    Evaluate a multi-dimensional gaussian for the given detunings (in units of most probable speed).

    This is equivalent to a gaussian distribution with rms width :math:`\\sigma=1/\\sqrt{2}`.

    Parameters
    ----------
    Vs : numpy.ndarray
        Array of normalized velocity classes for which to get
        the gaussian weighting.

    Returns
    -------
    numpy.ndarray
        Gaussian weights for the velocity classes.
        Has same shape as `Vs`.

    """
    spatial_dim = Vs.shape[0]
    if spatial_dim > 3:
        raise RydiquleError(f"Too many axes supplied: {spatial_dim}")

    prefactor = np.power(1/(np.pi),spatial_dim*0.5)

    return prefactor*np.exp(-np.square(Vs).sum(axis=0))


def doppler_classes(method: Optional[MeshMethod] = None
                    ) -> np.ndarray:
    """
    Defines which velocity classes to sample for doppler averaging.

    These are defined in units of the most probable speed of the Maxwell-Boltzmann
    distribution.

    Note
    ----

    To avoid issues, optical detunings should not leave densely sampled velocity classes.
    To avoid artifacts, the density of points should provide >~10 points over the
    narrowest absorptive feature. The default is a decent first guess, but for many
    problems the sampling mesh should be adjusted.

    Parameters
    ----------
    method : dict
        Specifies method to use and any control parameters.
        Must contain the key `"method"` with one of the following options.
        Each method has suboptions that also need to be specified.
        Valid options are:

            - `"uniform"`: Defines a uniformly spaced, dense grid.
              Configuration parameters include:

                - `"width_doppler"`: Float that specifies one-sided width of gaussian
                  distribution to average over, in units of most probable speed. Defaults to 2.0.
                - `"n_uniform"`: Int that specifies how many points to use. Defaults to 1601.

            - `"isopop"`: Defines a grid with uniform population in each interval.
              This method highly emphasises physics happening near the 0 velocity class.
              If stuff is happening for non-zero velocity classes,
              it is likely to alias it unless `n_isopop` is large.
              See Ref [1]_ for details.
              Configuration parameters include:

                - `"n_isopop"`: Int that specifies how many points to use. Defaults to 400.

            - `"split"`: Defines a grid with a dense central spacing and wide spacing wings.
              This method provides a decent compromise between uniform and isopop.
              It uses fewer points than uniform, but also works well for non-zero velocity class
              models (like Autler-Townes splittings).
              This is the default meshing method.
              Configuration parameters include:

              - `"width_doppler"`: Float that specifies one-sided width of coarse grided portion
                of the gaussian distribution.
                Units are in most probable speed. Defaults to 2.0.
              - `"width_coherent"`: Float that specifies one-sided width of fine grided portion
                of gaussian distribution. Units are in most probable speed. Defaults to 0.4.
              - `"n_doppler"`: Int that specifies how many points to use for the coarse grid.
                Note that points of the coarse grid that fall within the fine grid are
                dropped. Default is 201.
              - `"n_coherent"`: Int that specifies how many points to use for the fine grid.
                Default is 401.

              .. note::
                For the "split" method, a union of 2 samplings is taken,
                so the number of total points will not necessary be equal to the sum
                of `"n_coherent"` and `"n_doppler"`.

            - `"direct"`: Use the supplied 1-D numpy array to build the mesh.

              - `"doppler_velocities"`: Mandatory parameter that holds the 1-D numpy array
                to use when building the mesh grids. Given in units of most probably speed.

    Returns
    -------
    numpy.ndarray
        1-D array of velocities to be sampled.

    Examples
    --------
    The defaults will sample more densely near the center of the distribution,
    (the "split" method) with a total of 561 classes.

    >>> classes = rq.doppler_utils.doppler_classes() #use the default values
    >>> print(classes.shape)
    (561,)

    Specifying "uniform" with no additional arguments produces 1601 evenly spaced
    classes by default.

    >>> m = {"method":"uniform"}
    >>> classes = rq.doppler_utils.doppler_classes(method=m)
    >>> print(classes.shape)
    (1601,)

    Further specifying the number of points allows more dense or sparse sampling
    of the velocity distribution.

    >>> m = {"method":"uniform", "n_uniform":801}
    >>> classes = rq.doppler_utils.doppler_classes(method=m)
    >>> print(classes.shape)
    (801,)

    The "split" method also has further specifications

    >>> m = {"method":"split", "n_coherent":301, "n_doppler":501}
    >>> classes = rq.doppler_utils.doppler_classes(method=m)
    >>> print(classes.shape)
    (701,)

    References
    ----------
    .. [1] Andrew P. Rotunno, et. al.
        Inverse Transform Sampling for Efficient Doppler-Averaged Spectroscopy Simulation,
        AIP Advances 13, 075218 (2023)
        https://doi.org/10.1063/5.0157748

    """
    if method is None:
        method = SplitMethod(method="split")

    implemented_methods = ["uniform","isopop","split","direct"]
    try:
        if method["method"] not in implemented_methods:
            raise RydiquleError(f"Method {method['method']} is not a recognized meshing method.")
    except KeyError as err:
        raise RydiquleError("Meshing method must be a dictionary with at least key 'method'") from err

    if method["method"] == "uniform":
        # use default options if not provided
        width_doppler = method.get("width_doppler",2.0)
        n_uniform = method.get("n_uniform",1601)
        doppler_velocities = np.linspace(-width_doppler,width_doppler,n_uniform)
    elif method["method"] == "isopop":
        # define such that each slice has equal population distribution
        n_isopop = method.get("n_isopop", 400)
        bin_edges = np.linspace(0, 1.0, n_isopop+1)
        bin_centers = (bin_edges - bin_edges[1]/2)[1:]
        doppler_velocities: np.ndarray = special.erfinv(2*bin_centers-1)*np.sqrt(2)
    elif method["method"] == "split":
        # use default options if not provided
        width_doppler = method.get("width_doppler",2.0)
        width_coherent = method.get("width_coherent",0.4)
        n_doppler = method.get("n_doppler",201)
        n_coherent = method.get("n_coherent",401)
        # Doppler shifts to sample
        range_doppler = np.linspace(-width_doppler,width_doppler,n_doppler)
        # Finer mesh near zero velocity class where coherent stuff is happening
        range_coherent = np.linspace(-width_coherent,width_coherent,n_coherent)
        # Combine and get deltas for calculating the lazy integral
        # mask out points from course grid where fine grid is, avoids aliasing artifacts
        doppler_velocities = np.union1d(range_doppler[np.abs(range_doppler) > width_coherent],
                                        range_coherent)
    elif method["method"] == "direct":
        try:
            doppler_velocities = np.asarray(method["doppler_velocities"])
        except KeyError as err:
            raise RydiquleError("Method 'direct' must specify a 'doppler_velocities' config parameter") from err
        # assert shape is 1-D
        assert len(doppler_velocities.shape) == 1, "doppler_velocities must be 1-D"

    return doppler_velocities  # pyright: ignore[reportPossiblyUnboundVariable]


def doppler_mesh(doppler_velocities: np.ndarray, spatial_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates meshgrids of evaluation points and point "volumes" for doppler averaging.

    Parameters
    ----------
    dop_velocities : numpy.ndarray
        A 1-D array of velocities to evaluate over.
        These should be normalized to the most probable velocity used by
        :func:`gaussian3d`.
    spatial_dim : int
        Number of spatial dimensions to grid over.

    Returns
    -------
    Vs : numpy.ndarray
        Velocity evaluation points array of shape `(spatial_dim,spatial_dim*[len(dop_vel)])`.
    Vols : numpy.ndarray
        "Volume" of each meshpoint. Has same shape as `Vs`.

    Examples
    --------
    >>> m = {"method":"uniform", "n_uniform":801}
    >>> classes = rq.doppler_utils.doppler_classes(method=m)
    >>> mesh, vols = rq.doppler_utils.doppler_mesh(classes, 2)
    >>> print(type(mesh), type(vols))
    <class 'numpy.ndarray'> <class 'numpy.ndarray'>
    >>> mesh_np = np.array(mesh)
    >>> vols_np = np.array(vols)
    >>> print(mesh_np.shape, vols_np.shape)
    (2, 801, 801) (2, 801, 801)

    """
    dop_volumes = np.gradient(doppler_velocities)  # smoothly handles irregular arrays
    # generate the velocity meshgrids
    dets = [doppler_velocities for _ in range(spatial_dim)]
    diffs = [dop_volumes for _ in range(spatial_dim)]
    Vs = np.array(np.meshgrid(*dets,indexing="ij"))
    Vols = np.array(np.meshgrid(*diffs,indexing="ij"))

    return Vs, Vols


def apply_doppler_weights(sols: np.ndarray,
                          velocities: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """
    Calculates and applies the weight for each doppler class given unweighted solutions
    to doppler-shifted equations.

    Works for both time-domain and stead-state solutions.

    Parameters
    ----------
    sols : numpy.ndarray
        The array of solutions over which to calculate weights.
    velocities : numpy.ndarray
        Array of shape `(n_dim, *n_dop)` where n_dim
        is the number of dimensions over which doppler shifts are being considered
        and `*n_dop` is a number of axes equal to n_dim with length equal to the number
        of doppler velocity classes which are being considered. The values correspond
        the velocity class in units of most probable speed.
    volumes : numpy.ndarray
        Array of shape equal to `velocities`.
        The values correspond to the spacings between doppler classes on each axis.

    Returns
    -------
    numpy.ndarray
        The weighted solution array of shape equal to that of `sols`.

    Raises
    ------
    RydiquleError
        If the shapes of `velocities` and `volumes` do not match.

    """
    spatial_dim = volumes.shape[0]

    if volumes.shape != velocities.shape:
        raise RydiquleError((f"velocity shape {velocities.shape} does not match "
                             f"volume shape {volumes.shape}"))

    weights = gaussian3d(velocities)
    volumes = np.prod(volumes, axis=0)

    # calculate axes to append for array broadcasting to work in return
    solution_dim = len(sols.shape) - spatial_dim
    expand_axes = tuple(-1*np.arange(solution_dim)-1)

    weights = np.expand_dims(weights, expand_axes)
    volumes = np.expand_dims(volumes, expand_axes)

    return weights*volumes*sols
