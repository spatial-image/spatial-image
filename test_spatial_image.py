import pytest

import spatial_image as si

import numpy as np


def test_is_spatial_image():
    array = np.random.random((3, 4))
    assert not si.is_spatial_image(array)


def test_to_spatial_image():
    array = np.random.random((3, 4))
    image = si.to_spatial_image(array)
    assert si.is_spatial_image(image)


def test_2D_default_dims():
    array = np.random.random((3, 4))
    image = si.to_spatial_image(array)
    assert image.dims[0] == "y"
    assert image.dims[1] == "x"


def test_3D_default_dims():
    array = np.random.random((3, 4, 6))
    image = si.to_spatial_image(array)
    assert image.dims[0] == "z"
    assert image.dims[1] == "y"
    assert image.dims[2] == "x"


def test_4D_default_dims():
    array = np.random.random((3, 4, 6, 6))
    image = si.to_spatial_image(array)
    assert image.dims[0] == "z"
    assert image.dims[1] == "y"
    assert image.dims[2] == "x"
    assert image.dims[3] == "c"


def test_5D_default_dims():
    array = np.random.random((3, 4, 6, 6, 5))
    image = si.to_spatial_image(array)
    assert image.dims[0] == "t"
    assert image.dims[1] == "z"
    assert image.dims[2] == "y"
    assert image.dims[3] == "x"
    assert image.dims[4] == "c"


def test_catch_unsupported_dims():
    array = np.random.random((3, 4))
    with pytest.raises(ValueError):
        si.to_spatial_image(array, dims=("x", "purple"))
    image = si.to_spatial_image(array)
    image = image.rename({"x": "purple"})
    assert not si.is_spatial_image(image)


def test_2D_default_coords():
    array = np.random.random((3, 4))
    image = si.to_spatial_image(array)
    assert np.array_equal(image.coords["y"], np.arange(3, dtype=np.float64))
    assert np.array_equal(image.coords["x"], np.arange(4, dtype=np.float64))


def test_3D_default_coords():
    array = np.random.random((3, 4, 6))
    image = si.to_spatial_image(array)
    assert np.array_equal(image.coords["z"], np.arange(3, dtype=np.float64))
    assert np.array_equal(image.coords["y"], np.arange(4, dtype=np.float64))
    assert np.array_equal(image.coords["x"], np.arange(6, dtype=np.float64))


def test_4D_default_coords():
    array = np.random.random((3, 4, 6, 6))
    image = si.to_spatial_image(array)
    assert np.array_equal(image.coords["z"], np.arange(3, dtype=np.float64))
    assert np.array_equal(image.coords["y"], np.arange(4, dtype=np.float64))
    assert np.array_equal(image.coords["x"], np.arange(6, dtype=np.float64))
    assert np.array_equal(image.coords["c"], np.arange(6, dtype=np.float64))


def test_5D_default_coords():
    array = np.random.random((3, 4, 6, 6, 5))
    image = si.to_spatial_image(array)
    assert np.array_equal(image.coords["t"], np.arange(3, dtype=np.float64))
    assert np.array_equal(image.coords["z"], np.arange(4, dtype=np.float64))
    assert np.array_equal(image.coords["y"], np.arange(6, dtype=np.float64))
    assert np.array_equal(image.coords["x"], np.arange(6, dtype=np.float64))
    assert np.array_equal(image.coords["c"], np.arange(5, dtype=np.float64))


def test_spatial_coords_type_check():
    array = np.random.random((3, 4))
    coords = {"x": np.arange(4, dtype=np.float64), "y": np.arange(3, dtype=np.float64)}
    image = si.to_spatial_image(array, coords=coords)
    assert si.is_spatial_image(image)

    coords = {"x": np.arange(4, dtype=np.int16), "y": np.arange(3, dtype=np.int16)}
    image = si.to_spatial_image(array, coords=coords)

    assert not si.is_spatial_image(image)


def test_time_coord_type_check():
    array = np.random.random((2, 3, 4))
    coords = {
        "x": np.arange(4, dtype=np.float64),
        "y": np.arange(3, dtype=np.float64),
        "t": np.arange(2, dtype=np.int),
    }
    dims = ("t", "y", "x")
    image = si.to_spatial_image(array, dims=dims, coords=coords)
    assert si.is_spatial_image(image)

    coords = {
        "x": np.arange(4, dtype=np.float64),
        "y": np.arange(3, dtype=np.float64),
        "t": np.arange(2, dtype=np.unicode_),
    }
    dims = ("t", "y", "x")
    image = si.to_spatial_image(array, dims=dims, coords=coords)

    assert not si.is_spatial_image(image)


def test_uniform_coord_check():
    array = np.random.random((3, 4))
    coords = {
        "x": np.arange(4, dtype=np.float64),
        "y": np.arange(3, dtype=np.float64),
    }
    image = si.to_spatial_image(array, coords=coords)
    assert si.is_spatial_image(image)

    coords = {
        "x": np.arange(4, dtype=np.float64),
        "y": np.arange(3, dtype=np.float64),
    }
    coords["x"][1] = 1.1
    image = si.to_spatial_image(array, coords=coords)

    assert not si.is_spatial_image(image)
