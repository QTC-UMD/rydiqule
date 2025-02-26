import numpy as np
import rydiqule as rq
from rydiqule.doppler_utils import gaussian3d, doppler_classes, doppler_mesh
from rydiqule.slicing.slicing import matrix_slice
import pytest


@pytest.mark.doppler
@pytest.mark.high_memory
@pytest.mark.parametrize("spatial_dim", [1,2,3])
def test_gaussian(spatial_dim):
    """Confirms that the gaussian averaging function integrates to 1 in multiple dimensions"""

    if spatial_dim == 3:
        # for 3D, lower the meshing density to avoid RAM overflows
        dop_vels = doppler_classes({'method':'split','n_doppler':51,'n_coherent':201})
    else:
        dop_vels = doppler_classes({'method':'split'})

    Vs, Diffs = doppler_mesh(dop_vels,spatial_dim)

    weighting = Diffs.prod(axis=0)*gaussian3d(Vs)

    total = weighting.sum(axis=None)

    # confirm the integration is within 3%
    print(f'{spatial_dim:d}-D gaussian integration: {total:.2f}')
    assert total == pytest.approx(1.0, abs=3e-2), \
        f'{spatial_dim:d}-D gaussian integration not normalized'


@pytest.mark.doppler
@pytest.mark.util
def test_doppler_classes_assertion():
    """Confirms that doppler_classes correctly screens direct array inputs as 1-D"""

    # python list input
    l = [i for i in range(-10,11)]
    l_out = doppler_classes({'method':'direct','doppler_velocities':l})
    assert l_out == pytest.approx(l), 'Python list input mangled'

    # numpy 1-D array input
    arr = np.linspace(-1,1,100)
    arr_out = doppler_classes({'method':'direct','doppler_velocities':arr})
    assert arr_out == pytest.approx(arr), 'Numpy 1-D array input mangled'

    # numpy 2-D array input
    with pytest.raises(AssertionError,match='doppler_velocities must be 1-D'):
        arr2 = arr.reshape(20,5)
        _ = doppler_classes({'method':'direct','doppler_velocities':arr2})
        
        
@pytest.mark.util
def test_matrix_slice():
    """Uses a test set of matrices to ensure they slice correctly"""
    M1 = np.ones((1,15,5,5))
    M2 = np.ones((10,1,5,5))
    M3 = np.ones((10,15,5,5))

    shapes = [M.shape for M in [M1,M2,M3]]

    total_size_true = np.prod(np.broadcast_shapes(*shapes))
    total_size_calc = 0

    for _, m1, m2, m3 in matrix_slice(M1, M2, M3, n_slices=3):
        total_size_calc += np.prod(np.broadcast_shapes(m1.shape,m2.shape,m3.shape))

    assert total_size_calc == total_size_true


@pytest.mark.util
def test_density_matrix_conversions():
    '''Tests density matrix conversions between computational and complex bases.'''

    test_dms = np.array([[[0.1 , 0.  , 0.  , 0.25, 0.  , 0.  , 0.  , 0.25],
                          [0.1 , 0.  , 0.  , 0.25, 0.2 , 0.  , 0.  , 0.25]],
                         [[0. , 0. , 0.1, 0.5, 0.2, 0. , 0. , 0.5],
                          [0. , 0. , 0.1, 0.5, 0.2, 0. , 0.1, 0.5]]],
                          dtype=float)

    complex_dms = rq.sensor_utils.convert_dm_to_complex(test_dms)

    converted_dms = rq.sensor_utils.convert_complex_to_dm(complex_dms)

    np.testing.assert_allclose(test_dms, converted_dms)
