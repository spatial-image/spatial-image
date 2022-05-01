import pytest

import spatial_image as si

import numpy as np
import xarray as xr


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


def test_spatial_coords_set_scale():
    array = np.random.random((3, 4))
    image = si.to_spatial_image(array, scale={"x": 4.0, "y": 3.0})
    assert si.is_spatial_image(image)
    assert np.array_equal(image.coords["y"], [0.0, 3.0, 6.0])
    assert np.array_equal(image.coords["x"], [0.0, 4.0, 8.0, 12.0])


def test_time_coords():
    array = np.random.random((2, 3, 4))
    dims = ("t", "y", "x")
    image = si.to_spatial_image(
        array, dims=dims, scale={"x": 4.0, "y": 3.0}, t_coords=[0, 2]
    )
    assert si.is_spatial_image(image)
    assert np.array_equal(image.coords["t"], [0, 2])


def test_c_coords():
    array = np.random.random((3, 4, 2))
    dims = ("y", "x", "c")
    image = si.to_spatial_image(
        array, dims=dims, scale={"x": 4.0, "y": 3.0}, c_coords=[0, 2]
    )
    assert si.is_spatial_image(image)
    assert np.array_equal(image.coords["c"], [0, 2])


def test_SpatialImage_type():
    si.SpatialImage is xr.DataArray


def test_SpatialImageXDataClass():
    array = np.random.random((3,))

    image = si.SpatialImageXDataClass.new(array)
    assert np.array_equal(image.data, array)
    assert np.array_equal(image.coords["x"].data, np.arange(3, dtype=np.float64))
    assert image.name == "image"
    assert image.x.long_name == "x"

    image = si.SpatialImageXDataClass.new(
        array,
        scale={"x": 2.0},
        translation={"x": 3.5},
        name="img",
        axis_names={"x": "left-right"},
    )
    assert np.array_equal(image.data, array)
    assert np.array_equal(
        image.coords["x"].data, np.arange(3, dtype=np.float64) * 2.0 + 3.5
    )
    assert image.name == "img"
    assert image.x.long_name == "left-right"

    image = si.SpatialImageDataClasses[("x",)].new(array)
    assert np.array_equal(image.data, array)
    assert np.array_equal(image.coords["x"].data, np.arange(3, dtype=np.float64))
    assert image.name == "image"
    assert image.x.long_name == "x"


def test_SpatialImageXCDataClass():
    array = np.random.random((3, 2))

    image = si.SpatialImageXCDataClass.new(array)
    assert np.array_equal(image.data, array)
    assert np.array_equal(image.coords["x"].data, np.arange(3, dtype=np.float64))
    assert np.array_equal(image.coords["c"].data, np.arange(2))
    assert image.name == "image"
    assert image.x.long_name == "x"
    assert image.c.long_name == "c"

    image = si.SpatialImageXCDataClass.new(
        array,
        scale={"x": 2.0},
        translation={"x": 3.5},
        name="img",
        axis_names={"x": "left-right", "c": "features"},
        c_coords=["fa", "fb"],
    )
    assert np.array_equal(image.data, array)
    assert np.array_equal(
        image.coords["x"].data, np.arange(3, dtype=np.float64) * 2.0 + 3.5
    )
    assert np.array_equal(image.coords["c"].data, ["fa", "fb"])
    assert image.name == "img"
    assert image.x.long_name == "left-right"
    assert image.c.long_name == "features"

    assert si.SpatialImageDataClasses[("x", "c")] is si.SpatialImageXCDataClass


def test_SpatialImageTXDataClass():
    array = np.random.random((3, 2))

    image = si.SpatialImageTXDataClass.new(array)
    assert np.array_equal(image.data, array)
    assert np.array_equal(image.coords["t"].data, np.arange(3))
    assert np.array_equal(image.coords["x"].data, np.arange(2, dtype=np.float64))
    assert image.name == "image"
    assert image.x.long_name == "x"
    assert image.t.long_name == "t"

    image = si.SpatialImageTXDataClass.new(
        array,
        scale={"x": 2.0},
        translation={"x": 3.5},
        name="img",
        axis_names={"x": "left-right", "t": "time"},
        t_coords=["ta", "tb", "tc"],
    )
    assert np.array_equal(image.data, array)
    assert np.array_equal(
        image.coords["x"].data, np.arange(2, dtype=np.float64) * 2.0 + 3.5
    )
    assert np.array_equal(image.coords["t"].data, ["ta", "tb", "tc"])
    assert image.name == "img"
    assert image.x.long_name == "left-right"
    assert image.t.long_name == "time"

    assert si.SpatialImageDataClasses[("t", "x")] is si.SpatialImageTXDataClass


def test_SpatialImageTXCDataClass():
    array = np.random.random((3, 2, 4))

    image = si.SpatialImageTXCDataClass.new(array)
    assert np.array_equal(image.data, array)
    assert np.array_equal(image.coords["t"].data, np.arange(3))
    assert np.array_equal(image.coords["x"].data, np.arange(2, dtype=np.float64))
    assert np.array_equal(image.coords["c"].data, np.arange(4))
    assert image.name == "image"
    assert image.x.long_name == "x"
    assert image.t.long_name == "t"
    assert image.c.long_name == "c"
    assert image.x.units == ""
    assert image.t.units == ""
    assert image.c.units == ""

    image = si.SpatialImageTXCDataClass.new(
        array,
        scale={"x": 2.0},
        translation={"x": 3.5},
        name="img",
        axis_names={"x": "left-right", "t": "time"},
        axis_units={"x": "millimeters", "t": "seconds"},
        t_coords=["ta", "tb", "tc"],
        c_coords=["fa", "fb", "fc", "fd"],
    )
    assert np.array_equal(image.data, array)
    assert np.array_equal(
        image.coords["x"].data, np.arange(2, dtype=np.float64) * 2.0 + 3.5
    )
    assert np.array_equal(image.coords["t"].data, ["ta", "tb", "tc"])
    assert np.array_equal(image.coords["c"].data, ["fa", "fb", "fc", "fd"])
    assert image.name == "img"
    assert image.x.long_name == "left-right"
    assert image.t.long_name == "time"
    assert image.x.units == "millimeters"
    assert image.t.units == "seconds"

    assert si.SpatialImageDataClasses[("t", "x", "c")] is si.SpatialImageTXCDataClass


def test_SpatialImageYXDataClass():
    array = np.random.random((3, 2))

    image = si.SpatialImageYXDataClass.new(array)
    assert np.array_equal(image.data, array)
    assert np.array_equal(image.coords["y"].data, np.arange(3, dtype=np.float64))
    assert np.array_equal(image.coords["x"].data, np.arange(2, dtype=np.float64))
    assert image.name == "image"
    assert image.x.long_name == "x"
    assert image.y.long_name == "y"
    assert image.x.units == ""
    assert image.y.units == ""

    image = si.SpatialImageYXDataClass.new(
        array,
        scale={"y": 3.4, "x": 2.0},
        translation={"y": 1.2, "x": 3.5},
        name="img",
        axis_names={"x": "left-right", "y": "anterior-posterior"},
        axis_units={"x": "millimeters", "y": "micrometers"},
    )
    assert np.array_equal(image.data, array)
    assert np.array_equal(
        image.coords["y"].data, np.arange(3, dtype=np.float64) * 3.4 + 1.2
    )
    assert np.array_equal(
        image.coords["x"].data, np.arange(2, dtype=np.float64) * 2.0 + 3.5
    )
    assert image.name == "img"
    assert image.x.long_name == "left-right"
    assert image.y.long_name == "anterior-posterior"
    assert image.x.units == "millimeters"
    assert image.y.units == "micrometers"

    assert si.SpatialImageDataClasses[("y", "x")] is si.SpatialImageYXDataClass


def test_SpatialImageYXCDataClass():
    array = np.random.random((3, 2, 1))

    image = si.SpatialImageYXCDataClass.new(array)
    assert np.array_equal(image.data, array)
    assert np.array_equal(image.coords["y"].data, np.arange(3, dtype=np.float64))
    assert np.array_equal(image.coords["x"].data, np.arange(2, dtype=np.float64))
    assert np.array_equal(image.coords["c"].data, np.arange(1))
    assert image.name == "image"
    assert image.x.long_name == "x"
    assert image.y.long_name == "y"
    assert image.c.long_name == "c"
    assert image.x.units == ""
    assert image.y.units == ""
    assert image.c.units == ""

    image = si.SpatialImageYXCDataClass.new(
        array,
        scale={"y": 3.4, "x": 2.0},
        translation={"y": 1.2, "x": 3.5},
        name="img",
        axis_names={"x": "left-right", "y": "anterior-posterior"},
        axis_units={"x": "millimeters", "y": "micrometers"},
        c_coords=[
            40,
        ],
    )
    assert np.array_equal(image.data, array)
    assert np.array_equal(
        image.coords["y"].data, np.arange(3, dtype=np.float64) * 3.4 + 1.2
    )
    assert np.array_equal(
        image.coords["x"].data, np.arange(2, dtype=np.float64) * 2.0 + 3.5
    )
    assert np.array_equal(
        image.coords["c"].data,
        [
            40,
        ],
    )
    assert image.name == "img"
    assert image.x.long_name == "left-right"
    assert image.y.long_name == "anterior-posterior"
    assert image.x.units == "millimeters"
    assert image.y.units == "micrometers"

    assert si.SpatialImageDataClasses[("y", "x", "c")] is si.SpatialImageYXCDataClass


def test_SpatialImageTYXDataClass():
    array = np.random.random((2, 3, 2))

    image = si.SpatialImageTYXDataClass.new(array)
    assert np.array_equal(image.data, array)
    assert np.array_equal(image.coords["y"].data, np.arange(3, dtype=np.float64))
    assert np.array_equal(image.coords["x"].data, np.arange(2, dtype=np.float64))
    assert np.array_equal(image.coords["t"].data, np.arange(2))
    assert image.name == "image"
    assert image.x.long_name == "x"
    assert image.y.long_name == "y"
    assert image.t.long_name == "t"
    assert image.x.units == ""
    assert image.y.units == ""
    assert image.t.units == ""

    image = si.SpatialImageTYXDataClass.new(
        array,
        scale={"y": 3.4, "x": 2.0},
        translation={"y": 1.2, "x": 3.5},
        name="img",
        axis_names={"x": "left-right", "y": "anterior-posterior"},
        axis_units={"x": "millimeters", "y": "micrometers"},
        t_coords=[
            20,
            40,
        ],
    )
    assert np.array_equal(image.data, array)
    assert np.array_equal(
        image.coords["y"].data, np.arange(3, dtype=np.float64) * 3.4 + 1.2
    )
    assert np.array_equal(
        image.coords["x"].data, np.arange(2, dtype=np.float64) * 2.0 + 3.5
    )
    assert np.array_equal(
        image.coords["t"].data,
        [
            20,
            40,
        ],
    )
    assert image.name == "img"
    assert image.x.long_name == "left-right"
    assert image.y.long_name == "anterior-posterior"
    assert image.x.units == "millimeters"
    assert image.y.units == "micrometers"

    assert si.SpatialImageDataClasses[("t", "y", "x")] is si.SpatialImageTYXDataClass


def test_SpatialImageTYXCDataClass():
    array = np.random.random((2, 3, 2, 1))

    image = si.SpatialImageTYXCDataClass.new(array)
    assert np.array_equal(image.data, array)
    assert np.array_equal(image.coords["y"].data, np.arange(3, dtype=np.float64))
    assert np.array_equal(image.coords["x"].data, np.arange(2, dtype=np.float64))
    assert np.array_equal(image.coords["t"].data, np.arange(2))
    assert np.array_equal(image.coords["c"].data, np.arange(1))
    assert image.name == "image"
    assert image.x.long_name == "x"
    assert image.y.long_name == "y"
    assert image.t.long_name == "t"
    assert image.c.long_name == "c"
    assert image.x.units == ""
    assert image.y.units == ""
    assert image.t.units == ""
    assert image.c.units == ""

    image = si.SpatialImageTYXCDataClass.new(
        array,
        scale={"y": 3.4, "x": 2.0},
        translation={"y": 1.2, "x": 3.5},
        name="img",
        axis_names={"x": "left-right", "y": "anterior-posterior"},
        axis_units={"x": "millimeters", "y": "micrometers"},
        t_coords=[
            20,
            40,
        ],
        c_coords=[
            3,
        ],
    )
    assert np.array_equal(image.data, array)
    assert np.array_equal(
        image.coords["y"].data, np.arange(3, dtype=np.float64) * 3.4 + 1.2
    )
    assert np.array_equal(
        image.coords["x"].data, np.arange(2, dtype=np.float64) * 2.0 + 3.5
    )
    assert np.array_equal(
        image.coords["t"].data,
        [
            20,
            40,
        ],
    )
    assert np.array_equal(
        image.coords["c"].data,
        [
            3,
        ],
    )
    assert image.name == "img"
    assert image.x.long_name == "left-right"
    assert image.y.long_name == "anterior-posterior"
    assert image.x.units == "millimeters"
    assert image.y.units == "micrometers"

    assert (
        si.SpatialImageDataClasses[("t", "y", "x", "c")] is si.SpatialImageTYXCDataClass
    )


def test_SpatialImageZYXDataClass():
    array = np.random.random((2, 3, 2))

    image = si.SpatialImageZYXDataClass.new(array)
    assert np.array_equal(image.data, array)
    assert np.array_equal(image.coords["z"].data, np.arange(2, dtype=np.float64))
    assert np.array_equal(image.coords["y"].data, np.arange(3, dtype=np.float64))
    assert np.array_equal(image.coords["x"].data, np.arange(2, dtype=np.float64))
    assert image.name == "image"
    assert image.x.long_name == "x"
    assert image.y.long_name == "y"
    assert image.z.long_name == "z"
    assert image.x.units == ""
    assert image.y.units == ""
    assert image.z.units == ""

    image = si.SpatialImageZYXDataClass.new(
        array,
        scale={"z": 1.8, "y": 3.4, "x": 2.0},
        translation={"z": 0.9, "y": 1.2, "x": 3.5},
        name="img",
        axis_names={
            "z": "inferior-superior",
            "x": "left-right",
            "y": "anterior-posterior",
        },
        axis_units={"z": "millimeters", "x": "millimeters", "y": "micrometers"},
    )
    assert np.array_equal(image.data, array)
    assert np.array_equal(
        image.coords["z"].data, np.arange(2, dtype=np.float64) * 1.8 + 0.9
    )
    assert np.array_equal(
        image.coords["y"].data, np.arange(3, dtype=np.float64) * 3.4 + 1.2
    )
    assert np.array_equal(
        image.coords["x"].data, np.arange(2, dtype=np.float64) * 2.0 + 3.5
    )
    assert image.name == "img"
    assert image.x.long_name == "left-right"
    assert image.y.long_name == "anterior-posterior"
    assert image.z.long_name == "inferior-superior"
    assert image.x.units == "millimeters"
    assert image.y.units == "micrometers"
    assert image.z.units == "millimeters"

    assert si.SpatialImageDataClasses[("z", "y", "x")] is si.SpatialImageZYXDataClass


def test_SpatialImageZYXCDataClass():
    array = np.random.random((2, 3, 2, 1))

    image = si.SpatialImageZYXCDataClass.new(array)
    assert np.array_equal(image.data, array)
    assert np.array_equal(image.coords["z"].data, np.arange(2, dtype=np.float64))
    assert np.array_equal(image.coords["y"].data, np.arange(3, dtype=np.float64))
    assert np.array_equal(image.coords["x"].data, np.arange(2, dtype=np.float64))
    assert np.array_equal(image.coords["c"].data, np.arange(1))
    assert image.name == "image"
    assert image.x.long_name == "x"
    assert image.y.long_name == "y"
    assert image.z.long_name == "z"
    assert image.c.long_name == "c"
    assert image.x.units == ""
    assert image.y.units == ""
    assert image.z.units == ""
    assert image.c.units == ""

    image = si.SpatialImageZYXCDataClass.new(
        array,
        scale={"z": 1.8, "y": 3.4, "x": 2.0},
        translation={"z": 0.9, "y": 1.2, "x": 3.5},
        name="img",
        axis_names={
            "z": "inferior-superior",
            "x": "left-right",
            "y": "anterior-posterior",
        },
        axis_units={"z": "millimeters", "x": "millimeters", "y": "micrometers"},
        c_coords=[
            3,
        ],
    )
    assert np.array_equal(image.data, array)
    assert np.array_equal(
        image.coords["z"].data, np.arange(2, dtype=np.float64) * 1.8 + 0.9
    )
    assert np.array_equal(
        image.coords["y"].data, np.arange(3, dtype=np.float64) * 3.4 + 1.2
    )
    assert np.array_equal(
        image.coords["x"].data, np.arange(2, dtype=np.float64) * 2.0 + 3.5
    )
    assert np.array_equal(
        image.coords["c"].data,
        [
            3,
        ],
    )
    assert image.name == "img"
    assert image.x.long_name == "left-right"
    assert image.y.long_name == "anterior-posterior"
    assert image.z.long_name == "inferior-superior"
    assert image.x.units == "millimeters"
    assert image.y.units == "micrometers"
    assert image.z.units == "millimeters"

    assert (
        si.SpatialImageDataClasses[("z", "y", "x", "c")] is si.SpatialImageZYXCDataClass
    )


def test_SpatialImageTZYXDataClass():
    array = np.random.random((1, 2, 3, 2))

    image = si.SpatialImageTZYXDataClass.new(array)
    assert np.array_equal(image.data, array)
    assert np.array_equal(image.coords["z"].data, np.arange(2, dtype=np.float64))
    assert np.array_equal(image.coords["y"].data, np.arange(3, dtype=np.float64))
    assert np.array_equal(image.coords["x"].data, np.arange(2, dtype=np.float64))
    assert np.array_equal(image.coords["t"].data, np.arange(1))
    assert image.name == "image"
    assert image.x.long_name == "x"
    assert image.y.long_name == "y"
    assert image.z.long_name == "z"
    assert image.t.long_name == "t"
    assert image.x.units == ""
    assert image.y.units == ""
    assert image.z.units == ""
    assert image.t.units == ""

    image = si.SpatialImageTZYXDataClass.new(
        array,
        scale={"z": 1.8, "y": 3.4, "x": 2.0},
        translation={"z": 0.9, "y": 1.2, "x": 3.5},
        name="img",
        axis_names={
            "z": "inferior-superior",
            "x": "left-right",
            "y": "anterior-posterior",
        },
        axis_units={"z": "millimeters", "x": "millimeters", "y": "micrometers"},
        t_coords=[
            20,
        ],
    )
    assert np.array_equal(image.data, array)
    assert np.array_equal(
        image.coords["z"].data, np.arange(2, dtype=np.float64) * 1.8 + 0.9
    )
    assert np.array_equal(
        image.coords["y"].data, np.arange(3, dtype=np.float64) * 3.4 + 1.2
    )
    assert np.array_equal(
        image.coords["x"].data, np.arange(2, dtype=np.float64) * 2.0 + 3.5
    )
    assert np.array_equal(
        image.coords["t"].data,
        [
            20,
        ],
    )
    assert image.name == "img"
    assert image.x.long_name == "left-right"
    assert image.y.long_name == "anterior-posterior"
    assert image.z.long_name == "inferior-superior"
    assert image.x.units == "millimeters"
    assert image.y.units == "micrometers"
    assert image.z.units == "millimeters"

    assert (
        si.SpatialImageDataClasses[("t", "z", "y", "x")] is si.SpatialImageTZYXDataClass
    )


def test_SpatialImageTZYXCDataClass():
    array = np.random.random((1, 2, 3, 2, 1))

    image = si.SpatialImageTZYXCDataClass.new(array)
    assert np.array_equal(image.data, array)
    assert np.array_equal(image.coords["z"].data, np.arange(2, dtype=np.float64))
    assert np.array_equal(image.coords["y"].data, np.arange(3, dtype=np.float64))
    assert np.array_equal(image.coords["x"].data, np.arange(2, dtype=np.float64))
    assert np.array_equal(image.coords["t"].data, np.arange(1))
    assert np.array_equal(image.coords["t"].data, np.arange(1))
    assert image.name == "image"
    assert image.x.long_name == "x"
    assert image.y.long_name == "y"
    assert image.z.long_name == "z"
    assert image.t.long_name == "t"
    assert image.c.long_name == "c"
    assert image.x.units == ""
    assert image.y.units == ""
    assert image.z.units == ""
    assert image.t.units == ""
    assert image.c.units == ""

    image = si.SpatialImageTZYXCDataClass.new(
        array,
        scale={"z": 1.8, "y": 3.4, "x": 2.0},
        translation={"z": 0.9, "y": 1.2, "x": 3.5},
        name="img",
        axis_names={
            "z": "inferior-superior",
            "x": "left-right",
            "y": "anterior-posterior",
        },
        axis_units={"z": "millimeters", "x": "millimeters", "y": "micrometers"},
        t_coords=[
            20,
        ],
        c_coords=[
            4,
        ],
    )
    assert np.array_equal(image.data, array)
    assert np.array_equal(
        image.coords["z"].data, np.arange(2, dtype=np.float64) * 1.8 + 0.9
    )
    assert np.array_equal(
        image.coords["y"].data, np.arange(3, dtype=np.float64) * 3.4 + 1.2
    )
    assert np.array_equal(
        image.coords["x"].data, np.arange(2, dtype=np.float64) * 2.0 + 3.5
    )
    assert np.array_equal(
        image.coords["t"].data,
        [
            20,
        ],
    )
    assert np.array_equal(
        image.coords["c"].data,
        [
            4,
        ],
    )
    assert image.name == "img"
    assert image.x.long_name == "left-right"
    assert image.y.long_name == "anterior-posterior"
    assert image.z.long_name == "inferior-superior"
    assert image.x.units == "millimeters"
    assert image.y.units == "micrometers"
    assert image.z.units == "millimeters"

    assert (
        si.SpatialImageDataClasses[("t", "z", "y", "x", "c")]
        is si.SpatialImageTZYXCDataClass
    )


def test_SpatialImageCXDataClass():
    array = np.random.random((2, 3))

    image = si.SpatialImageCXDataClass.new(array)
    assert np.array_equal(image.data, array)
    assert np.array_equal(image.coords["x"].data, np.arange(3, dtype=np.float64))
    assert np.array_equal(image.coords["c"].data, np.arange(2))
    assert image.name == "image"
    assert image.x.long_name == "x"
    # assert image.c.long_name == "c"

    image = si.SpatialImageCXDataClass.new(
        array,
        scale={"x": 2.0},
        translation={"x": 3.5},
        name="img",
        axis_names={"x": "left-right", "c": "features"},
        c_coords=["fa", "fb"],
    )
    assert np.array_equal(image.data, array)
    assert np.array_equal(
        image.coords["x"].data, np.arange(3, dtype=np.float64) * 2.0 + 3.5
    )
    assert np.array_equal(image.coords["c"].data, ["fa", "fb"])
    assert image.name == "img"
    assert image.x.long_name == "left-right"
    assert image.c.long_name == "features"

    assert si.SpatialImageDataClasses[("c", "x")] is si.SpatialImageCXDataClass


def test_SpatialImageTCXDataClass():
    array = np.random.random((3, 4, 2))

    image = si.SpatialImageTCXDataClass.new(array)
    assert np.array_equal(image.data, array)
    assert np.array_equal(image.coords["t"].data, np.arange(3))
    assert np.array_equal(image.coords["x"].data, np.arange(2, dtype=np.float64))
    assert np.array_equal(image.coords["c"].data, np.arange(4))
    assert image.name == "image"
    assert image.x.long_name == "x"
    assert image.t.long_name == "t"
    assert image.c.long_name == "c"
    assert image.x.units == ""
    assert image.t.units == ""
    assert image.c.units == ""

    image = si.SpatialImageTCXDataClass.new(
        array,
        scale={"x": 2.0},
        translation={"x": 3.5},
        name="img",
        axis_names={"x": "left-right", "t": "time"},
        axis_units={"x": "millimeters", "t": "seconds"},
        t_coords=["ta", "tb", "tc"],
        c_coords=["fa", "fb", "fc", "fd"],
    )
    assert np.array_equal(image.data, array)
    assert np.array_equal(
        image.coords["x"].data, np.arange(2, dtype=np.float64) * 2.0 + 3.5
    )
    assert np.array_equal(image.coords["t"].data, ["ta", "tb", "tc"])
    assert np.array_equal(image.coords["c"].data, ["fa", "fb", "fc", "fd"])
    assert image.name == "img"
    assert image.x.long_name == "left-right"
    assert image.t.long_name == "time"
    assert image.x.units == "millimeters"
    assert image.t.units == "seconds"

    assert si.SpatialImageDataClasses[("t", "c", "x")] is si.SpatialImageTCXDataClass


def test_SpatialImageCYXDataClass():
    array = np.random.random((1, 3, 2))

    image = si.SpatialImageCYXDataClass.new(array)
    assert np.array_equal(image.data, array)
    assert np.array_equal(image.coords["y"].data, np.arange(3, dtype=np.float64))
    assert np.array_equal(image.coords["x"].data, np.arange(2, dtype=np.float64))
    assert np.array_equal(image.coords["c"].data, np.arange(1))
    assert image.name == "image"
    assert image.x.long_name == "x"
    assert image.y.long_name == "y"
    assert image.c.long_name == "c"
    assert image.x.units == ""
    assert image.y.units == ""
    assert image.c.units == ""

    image = si.SpatialImageCYXDataClass.new(
        array,
        scale={"y": 3.4, "x": 2.0},
        translation={"y": 1.2, "x": 3.5},
        name="img",
        axis_names={"x": "left-right", "y": "anterior-posterior"},
        axis_units={"x": "millimeters", "y": "micrometers"},
        c_coords=[
            40,
        ],
    )
    assert np.array_equal(image.data, array)
    assert np.array_equal(
        image.coords["y"].data, np.arange(3, dtype=np.float64) * 3.4 + 1.2
    )
    assert np.array_equal(
        image.coords["x"].data, np.arange(2, dtype=np.float64) * 2.0 + 3.5
    )
    assert np.array_equal(
        image.coords["c"].data,
        [
            40,
        ],
    )
    assert image.name == "img"
    assert image.x.long_name == "left-right"
    assert image.y.long_name == "anterior-posterior"
    assert image.x.units == "millimeters"
    assert image.y.units == "micrometers"

    assert si.SpatialImageDataClasses[("c", "y", "x")] is si.SpatialImageCYXDataClass


def test_SpatialImageTCYXDataClass():
    array = np.random.random((2, 1, 3, 2))

    image = si.SpatialImageTCYXDataClass.new(array)
    assert np.array_equal(image.data, array)
    assert np.array_equal(image.coords["y"].data, np.arange(3, dtype=np.float64))
    assert np.array_equal(image.coords["x"].data, np.arange(2, dtype=np.float64))
    assert np.array_equal(image.coords["t"].data, np.arange(2))
    assert np.array_equal(image.coords["c"].data, np.arange(1))
    assert image.name == "image"
    assert image.x.long_name == "x"
    assert image.y.long_name == "y"
    assert image.t.long_name == "t"
    assert image.c.long_name == "c"
    assert image.x.units == ""
    assert image.y.units == ""
    assert image.t.units == ""
    assert image.c.units == ""

    image = si.SpatialImageTCYXDataClass.new(
        array,
        scale={"y": 3.4, "x": 2.0},
        translation={"y": 1.2, "x": 3.5},
        name="img",
        axis_names={"x": "left-right", "y": "anterior-posterior"},
        axis_units={"x": "millimeters", "y": "micrometers"},
        t_coords=[
            20,
            40,
        ],
        c_coords=[
            3,
        ],
    )
    assert np.array_equal(image.data, array)
    assert np.array_equal(
        image.coords["y"].data, np.arange(3, dtype=np.float64) * 3.4 + 1.2
    )
    assert np.array_equal(
        image.coords["x"].data, np.arange(2, dtype=np.float64) * 2.0 + 3.5
    )
    assert np.array_equal(
        image.coords["t"].data,
        [
            20,
            40,
        ],
    )
    assert np.array_equal(
        image.coords["c"].data,
        [
            3,
        ],
    )
    assert image.name == "img"
    assert image.x.long_name == "left-right"
    assert image.y.long_name == "anterior-posterior"
    assert image.x.units == "millimeters"
    assert image.y.units == "micrometers"

    assert (
        si.SpatialImageDataClasses[("t", "c", "y", "x")] is si.SpatialImageTCYXDataClass
    )


def test_SpatialImageTCZYXDataClass():
    array = np.random.random((1, 1, 2, 3, 2))

    image = si.SpatialImageTCZYXDataClass.new(array)
    assert np.array_equal(image.data, array)
    assert np.array_equal(image.coords["z"].data, np.arange(2, dtype=np.float64))
    assert np.array_equal(image.coords["y"].data, np.arange(3, dtype=np.float64))
    assert np.array_equal(image.coords["x"].data, np.arange(2, dtype=np.float64))
    assert np.array_equal(image.coords["t"].data, np.arange(1))
    assert np.array_equal(image.coords["t"].data, np.arange(1))
    assert image.name == "image"
    assert image.x.long_name == "x"
    assert image.y.long_name == "y"
    assert image.z.long_name == "z"
    assert image.t.long_name == "t"
    assert image.c.long_name == "c"
    assert image.x.units == ""
    assert image.y.units == ""
    assert image.z.units == ""
    assert image.t.units == ""
    assert image.c.units == ""

    image = si.SpatialImageTCZYXDataClass.new(
        array,
        scale={"z": 1.8, "y": 3.4, "x": 2.0},
        translation={"z": 0.9, "y": 1.2, "x": 3.5},
        name="img",
        axis_names={
            "z": "inferior-superior",
            "x": "left-right",
            "y": "anterior-posterior",
        },
        axis_units={"z": "millimeters", "x": "millimeters", "y": "micrometers"},
        t_coords=[
            20,
        ],
        c_coords=[
            4,
        ],
    )
    assert np.array_equal(image.data, array)
    assert np.array_equal(
        image.coords["z"].data, np.arange(2, dtype=np.float64) * 1.8 + 0.9
    )
    assert np.array_equal(
        image.coords["y"].data, np.arange(3, dtype=np.float64) * 3.4 + 1.2
    )
    assert np.array_equal(
        image.coords["x"].data, np.arange(2, dtype=np.float64) * 2.0 + 3.5
    )
    assert np.array_equal(
        image.coords["t"].data,
        [
            20,
        ],
    )
    assert np.array_equal(
        image.coords["c"].data,
        [
            4,
        ],
    )
    assert image.name == "img"
    assert image.x.long_name == "left-right"
    assert image.y.long_name == "anterior-posterior"
    assert image.z.long_name == "inferior-superior"
    assert image.x.units == "millimeters"
    assert image.y.units == "micrometers"
    assert image.z.units == "millimeters"

    assert (
        si.SpatialImageDataClasses[("t", "c", "z", "y", "x")]
        is si.SpatialImageTCZYXDataClass
    )
