import spatial_image as si

import numpy as np

def test_is_spatial_image():
    array = np.random.random((3,4))
    assert(not si.is_spatial_image(array))

def test_to_spatial_image():
    array = np.random.random((3,4))
    image = si.to_spatial_image(array)
    # assert(si.is_spatial_image(image))
