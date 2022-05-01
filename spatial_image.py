"""spatial-image

A multi-dimensional spatial image data structure for Python."""

__version__ = "0.2.1"

from typing import Union, Optional, Sequence, Hashable, Tuple, Mapping, Any, Literal
from dataclasses import dataclass

import xarray as xr
import numpy as np

from xarray_dataclasses.dataarray import AsDataArray
from xarray_dataclasses.typing import Attr, Coordof, Data, Name
from xarray_dataclasses.dataoptions import DataOptions


_supported_dims = {"c", "x", "y", "z", "t"}
_spatial_dims = {"x", "y", "z"}

SupportedDims = Union[
    Literal["c"], Literal["x"], Literal["y"], Literal["z"], Literal["t"]
]
SpatialDims = Union[Literal["x"], Literal["y"], Literal["z"]]
AllInteger = Union[
    np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64
]
AllFloat = Union[np.float32, np.float64]

# Channel or component
C = Literal["c"]
X = Literal["x"]
Y = Literal["y"]
Z = Literal["z"]
# Time
T = Literal["t"]


@dataclass
class TAxis:
    # data: Data[T, Union[AllInteger, AllFloat, np.datetime64]]
    data: Data[T, Any]
    long_name: Attr[str] = "t"
    units: Attr[str] = ""


@dataclass
class XAxis:
    data: Data[X, np.float64]
    long_name: Attr[str] = "x"
    units: Attr[str] = ""


@dataclass
class YAxis:
    data: Data[Y, np.float64]
    long_name: Attr[str] = "y"
    units: Attr[str] = ""


@dataclass
class ZAxis:
    data: Data[Z, np.float64]
    long_name: Attr[str] = "z"
    units: Attr[str] = ""


@dataclass
class CAxis:
    # data: Data[C, Union[AllInteger, str]]
    data: Data[C, Any]
    long_name: Attr[str] = "c"
    units: Attr[str] = ""


default_name = "image"


class SpatialImage(xr.DataArray):
    """Spatial-image xarray DataArray datastructure.

    data:
        Multi-dimensional array that provides the image pixel values.

    dims:
        Values should drawn from: {'c', 'x', 'y', 'z', 't'} for channel or
        component, first spatial direction, second spatial direction, third
        spatial dimension, and time, respectively.

    coords: sequence or dict of array_like objects, optional
        For each {'x', 'y', 'z'} dim, 1-D np.float64 array specifing the
        pixel location in the image's local coordinate system. The distance
        between subsequent coords elements must be uniform. The 'c' coords
        are a sequence of integers by default but can be strings describing the
        channels, e.g. ['r', 'g', 'b']. The 't' coords can have int, float,
        or datetime64 type. A *long_name* coord attr can provide a more descriptive
        name for the axis, e.g. an anatomical identifier. A *units* coord attr can
        identify the units associated with the axis."""

    __slots__ = ()


@dataclass(init=False)
class SpatialImageDataClass(AsDataArray):
    """An xarray.DataArray dataclass for a spatial image."""

    __dataoptions__ = DataOptions(SpatialImage)

    name: Name[str]

    def __init__(self, name: str = default_name):
        self.name = name


@dataclass(init=False)
class SpatialImageXDataClass(SpatialImageDataClass):
    """A 1D spatial image."""

    data: Data[Tuple[X], Any]
    x: Coordof[XAxis]

    def __init__(
        self,
        data,
        scale: Optional[Union[Mapping[Hashable, float]]] = None,
        translation: Optional[Union[Mapping[Hashable, float]]] = None,
        name: str = default_name,
        axis_names: Optional[Union[Mapping[Hashable, str]]] = None,
        axis_units: Optional[Union[Mapping[Hashable, str]]] = None,
    ):
        super().__init__(name)
        self.data = data

        if scale is None:
            scale = {"x": 1.0}

        if translation is None:
            translation = {"x": 0.0}

        x_axis_name = "x"
        if axis_names and "x" in axis_names:
            x_axis_name = axis_names["x"]
        x_axis_units = ""
        if axis_units and "x" in axis_units:
            x_axis_units = axis_units["x"]
        self.x = XAxis(
            np.arange(data.shape[0], dtype=np.float64) * scale["x"] + translation["x"],
            long_name=x_axis_name,
            units=x_axis_units,
        )


@dataclass(init=False)
class SpatialImageXCDataClass(SpatialImageXDataClass):
    """A 1D spatial image with channels."""

    data: Data[Tuple[X, C], Any]
    c: Coordof[CAxis]

    def __init__(
        self,
        data,
        scale: Optional[Union[Mapping[Hashable, float]]] = None,
        translation: Optional[Union[Mapping[Hashable, float]]] = None,
        name: str = default_name,
        axis_names: Optional[Union[Mapping[Hashable, str]]] = None,
        axis_units: Optional[Union[Mapping[Hashable, str]]] = None,
        c_coords: Optional[Sequence[Union[AllInteger, str]]] = None,
    ):
        super().__init__(data, scale, translation, name, axis_names, axis_units)
        if c_coords is None:
            c_coords = np.arange(data.shape[1])

        c_axis_name = "c"
        if axis_names is not None and "c" in axis_names:
            c_axis_name = axis_names["c"]
        c_axis_units = ""
        if axis_units and "c" in axis_units:
            c_axis_units["c"] = ""

        self.c = CAxis(c_coords, c_axis_name, c_axis_units)


@dataclass(init=False)
class SpatialImageTXDataClass(SpatialImageDataClass):
    """A 1D spatial image with a time dimension."""

    data: Data[Tuple[T, X], Any]
    t: Coordof[TAxis]
    x: Coordof[XAxis]

    def __init__(
        self,
        data,
        scale: Optional[Union[Mapping[Hashable, float]]] = None,
        translation: Optional[Union[Mapping[Hashable, float]]] = None,
        name: str = default_name,
        axis_names: Optional[Union[Mapping[Hashable, str]]] = None,
        axis_units: Optional[Union[Mapping[Hashable, str]]] = None,
        t_coords: Optional[Sequence[Union[AllInteger, AllFloat, np.datetime64]]] = None,
    ):
        super().__init__(name)
        self.data = data

        if scale is None:
            scale = {"x": 1.0}

        if translation is None:
            translation = {"x": 0.0}

        x_axis_name = "x"
        if axis_names and "x" in axis_names:
            x_axis_name = axis_names["x"]
        x_axis_units = ""
        if axis_units and "x" in axis_units:
            x_axis_units = axis_units["x"]
        self.x = XAxis(
            np.arange(data.shape[1], dtype=np.float64) * scale["x"] + translation["x"],
            long_name=x_axis_name,
            units=x_axis_units,
        )

        t_axis_name = "t"
        if axis_names and "t" in axis_names:
            t_axis_name = axis_names["t"]
        t_axis_units = ""
        if axis_units and "t" in axis_units:
            t_axis_units = axis_units["t"]
        if t_coords is None:
            t_coords = np.arange(data.shape[0])
        self.t = TAxis(t_coords, t_axis_name, t_axis_units)


@dataclass(init=False)
class SpatialImageTXCDataClass(SpatialImageTXDataClass):
    """A 1D spatial image with a time dimension and channels."""

    data: Data[Tuple[T, X, C], Any]
    c: Coordof[CAxis]

    def __init__(
        self,
        data,
        scale: Optional[Union[Mapping[Hashable, float]]] = None,
        translation: Optional[Union[Mapping[Hashable, float]]] = None,
        name: str = default_name,
        axis_names: Optional[Union[Mapping[Hashable, str]]] = None,
        axis_units: Optional[Union[Mapping[Hashable, str]]] = None,
        t_coords: Optional[Sequence[Union[AllInteger, AllFloat, np.datetime64]]] = None,
        c_coords: Optional[Sequence[Union[AllInteger, str]]] = None,
    ):
        super().__init__(
            data, scale, translation, name, axis_names, axis_units, t_coords
        )
        if c_coords is None:
            c_coords = np.arange(data.shape[2])

        c_axis_name = "c"
        if axis_names is not None and "c" in axis_names:
            c_axis_name = axis_names["c"]

        c_axis_units = ""
        if axis_units is not None and "c" in axis_units:
            c_axis_units = axis_units["c"]

        self.c = CAxis(c_coords, c_axis_name, c_axis_units)


@dataclass(init=False)
class SpatialImageYXDataClass(SpatialImageDataClass):
    """A 2D spatial image."""

    data: Data[Tuple[Y, X], Any]
    y: Coordof[YAxis]
    x: Coordof[XAxis]

    def __init__(
        self,
        data,
        scale: Optional[Union[Mapping[Hashable, float]]] = None,
        translation: Optional[Union[Mapping[Hashable, float]]] = None,
        name: str = default_name,
        axis_names: Optional[Union[Mapping[Hashable, str]]] = None,
        axis_units: Optional[Union[Mapping[Hashable, str]]] = None,
    ):
        super().__init__(name)
        self.data = data

        if scale is None:
            scale = {"y": 1.0, "x": 1.0}

        if translation is None:
            translation = {"y": 0.0, "x": 0.0}

        y_axis_name = "y"
        if axis_names and "y" in axis_names:
            y_axis_name = axis_names["y"]
        y_axis_units = ""
        if axis_units and "y" in axis_units:
            y_axis_units = axis_units["y"]
        self.y = YAxis(
            np.arange(data.shape[0], dtype=np.float64) * scale["y"] + translation["y"],
            long_name=y_axis_name,
            units=y_axis_units,
        )

        x_axis_name = "x"
        if axis_names and "x" in axis_names:
            x_axis_name = axis_names["x"]
        x_axis_units = ""
        if axis_units and "x" in axis_units:
            x_axis_units = axis_units["x"]
        self.x = XAxis(
            np.arange(data.shape[1], dtype=np.float64) * scale["x"] + translation["x"],
            long_name=x_axis_name,
            units=x_axis_units,
        )


@dataclass(init=False)
class SpatialImageYXCDataClass(SpatialImageYXDataClass):
    """A 2D spatial image with channels."""

    data: Data[Tuple[Y, X, C], Any]
    c: Coordof[CAxis]

    def __init__(
        self,
        data,
        scale: Optional[Union[Mapping[Hashable, float]]] = None,
        translation: Optional[Union[Mapping[Hashable, float]]] = None,
        name: str = default_name,
        axis_names: Optional[Union[Mapping[Hashable, str]]] = None,
        axis_units: Optional[Union[Mapping[Hashable, str]]] = None,
        c_coords: Optional[Sequence[Union[AllInteger, str]]] = None,
    ):
        super().__init__(data, scale, translation, name, axis_names, axis_units)
        if c_coords is None:
            c_coords = np.arange(data.shape[2])

        c_axis_name = "c"
        if axis_names and "c" in axis_names:
            c_axis_name = axis_names["c"]
        c_axis_units = ""
        if axis_units and "c" in axis_units:
            c_axis_units["c"] = ""
        self.c = CAxis(c_coords, c_axis_name, c_axis_units)


@dataclass(init=False)
class SpatialImageTYXDataClass(SpatialImageDataClass):
    """A 2D spatial image with a time dimension."""

    data: Data[Tuple[T, Y, X], Any]
    t: Coordof[TAxis]
    y: Coordof[YAxis]
    x: Coordof[XAxis]

    def __init__(
        self,
        data,
        scale: Optional[Union[Mapping[Hashable, float]]] = None,
        translation: Optional[Union[Mapping[Hashable, float]]] = None,
        name: str = default_name,
        axis_names: Optional[Union[Mapping[Hashable, str]]] = None,
        axis_units: Optional[Union[Mapping[Hashable, str]]] = None,
        t_coords: Optional[Sequence[Union[AllInteger, AllFloat, np.datetime64]]] = None,
    ):
        super().__init__(name)
        self.data = data

        if scale is None:
            scale = {"y": 1.0, "x": 1.0}

        if translation is None:
            translation = {"y": 0.0, "x": 0.0}

        y_axis_name = "y"
        if axis_names and "y" in axis_names:
            y_axis_name = axis_names["y"]
        y_axis_units = ""
        if axis_units and "y" in axis_units:
            y_axis_units = axis_units["y"]
        self.y = YAxis(
            np.arange(data.shape[1], dtype=np.float64) * scale["y"] + translation["y"],
            long_name=y_axis_name,
            units=y_axis_units,
        )

        x_axis_name = "x"
        if axis_names and "x" in axis_names:
            x_axis_name = axis_names["x"]
        x_axis_units = ""
        if axis_units and "x" in axis_units:
            x_axis_units = axis_units["x"]
        self.x = XAxis(
            np.arange(data.shape[2], dtype=np.float64) * scale["x"] + translation["x"],
            long_name=x_axis_name,
            units=x_axis_units,
        )

        t_axis_name = "t"
        if axis_names and "t" in axis_names:
            t_axis_name = axis_names["t"]
        t_axis_units = ""
        if axis_units and "t" in axis_units:
            t_axis_units = axis_units["t"]
        if t_coords is None:
            t_coords = np.arange(data.shape[0])
        self.t = TAxis(t_coords, t_axis_name, t_axis_units)


@dataclass(init=False)
class SpatialImageTYXCDataClass(SpatialImageTYXDataClass):
    """A 2D spatial image with a time dimension and channels."""

    data: Data[Tuple[T, Y, X, C], Any]
    c: Coordof[CAxis]

    def __init__(
        self,
        data,
        scale: Optional[Union[Mapping[Hashable, float]]] = None,
        translation: Optional[Union[Mapping[Hashable, float]]] = None,
        name: str = default_name,
        axis_names: Optional[Union[Mapping[Hashable, str]]] = None,
        axis_units: Optional[Union[Mapping[Hashable, str]]] = None,
        t_coords: Optional[Sequence[Union[AllInteger, AllFloat, np.datetime64]]] = None,
        c_coords: Optional[Sequence[Union[AllInteger, str]]] = None,
    ):
        super().__init__(
            data, scale, translation, name, axis_names, axis_units, t_coords
        )
        if c_coords is None:
            c_coords = np.arange(data.shape[3])

        c_axis_name = "c"
        if axis_names is not None and "c" in axis_names:
            c_axis_name = axis_names["c"]

        c_axis_units = ""
        if axis_units is not None and "c" in axis_units:
            c_axis_units = axis_units["c"]

        self.c = CAxis(c_coords, c_axis_name, c_axis_units)


@dataclass(init=False)
class SpatialImageZYXDataClass(SpatialImageDataClass):
    """A 3D spatial image."""

    data: Data[Tuple[Z, Y, X], Any]
    z: Coordof[ZAxis]
    y: Coordof[YAxis]
    x: Coordof[XAxis]

    def __init__(
        self,
        data,
        scale: Optional[Union[Mapping[Hashable, float]]] = None,
        translation: Optional[Union[Mapping[Hashable, float]]] = None,
        name: str = default_name,
        axis_names: Optional[Union[Mapping[Hashable, str]]] = None,
        axis_units: Optional[Union[Mapping[Hashable, str]]] = None,
    ):
        super().__init__(name)
        self.data = data

        if scale is None:
            scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        if translation is None:
            translation = {"z": 0.0, "y": 0.0, "x": 0.0}

        z_axis_name = "z"
        if axis_names and "z" in axis_names:
            z_axis_name = axis_names["z"]
        z_axis_units = ""
        if axis_units and "z" in axis_units:
            z_axis_units = axis_units["z"]
        self.z = ZAxis(
            np.arange(data.shape[0], dtype=np.float64) * scale["z"] + translation["z"],
            long_name=z_axis_name,
            units=z_axis_units,
        )

        y_axis_name = "y"
        if axis_names and "y" in axis_names:
            y_axis_name = axis_names["y"]
        y_axis_units = ""
        if axis_units and "y" in axis_units:
            y_axis_units = axis_units["y"]
        self.y = YAxis(
            np.arange(data.shape[1], dtype=np.float64) * scale["y"] + translation["y"],
            long_name=y_axis_name,
            units=y_axis_units,
        )

        x_axis_name = "x"
        if axis_names and "x" in axis_names:
            x_axis_name = axis_names["x"]
        x_axis_units = ""
        if axis_units and "x" in axis_units:
            x_axis_units = axis_units["x"]
        self.x = XAxis(
            np.arange(data.shape[2], dtype=np.float64) * scale["x"] + translation["x"],
            long_name=x_axis_name,
            units=x_axis_units,
        )


@dataclass(init=False)
class SpatialImageZYXCDataClass(SpatialImageZYXDataClass):
    """A 3D spatial image with channels."""

    data: Data[Tuple[Z, Y, X, C], Any]
    c: Coordof[CAxis]

    def __init__(
        self,
        data,
        scale: Optional[Union[Mapping[Hashable, float]]] = None,
        translation: Optional[Union[Mapping[Hashable, float]]] = None,
        name: str = default_name,
        axis_names: Optional[Union[Mapping[Hashable, str]]] = None,
        axis_units: Optional[Union[Mapping[Hashable, str]]] = None,
        c_coords: Optional[Sequence[Union[AllInteger, str]]] = None,
    ):
        super().__init__(data, scale, translation, name, axis_names, axis_units)
        if c_coords is None:
            c_coords = np.arange(data.shape[3])

        c_axis_name = "c"
        if axis_names and "c" in axis_names:
            c_axis_name = axis_names["c"]
        c_axis_units = ""
        if axis_units and "c" in axis_units:
            c_axis_units["c"] = ""
        self.c = CAxis(c_coords, c_axis_name, c_axis_units)


@dataclass(init=False)
class SpatialImageTZYXDataClass(SpatialImageDataClass):
    """A 3D spatial image with a time dimension."""

    data: Data[Tuple[T, Z, Y, X], Any]
    t: Coordof[TAxis]
    z: Coordof[ZAxis]
    y: Coordof[YAxis]
    x: Coordof[XAxis]

    def __init__(
        self,
        data,
        scale: Optional[Union[Mapping[Hashable, float]]] = None,
        translation: Optional[Union[Mapping[Hashable, float]]] = None,
        name: str = default_name,
        axis_names: Optional[Union[Mapping[Hashable, str]]] = None,
        axis_units: Optional[Union[Mapping[Hashable, str]]] = None,
        t_coords: Optional[Sequence[Union[AllInteger, AllFloat, np.datetime64]]] = None,
    ):
        super().__init__(name)
        self.data = data

        if scale is None:
            scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        if translation is None:
            translation = {"z": 0.0, "y": 0.0, "x": 0.0}

        z_axis_name = "z"
        if axis_names and "z" in axis_names:
            z_axis_name = axis_names["z"]
        z_axis_units = ""
        if axis_units and "z" in axis_units:
            z_axis_units = axis_units["z"]
        self.z = ZAxis(
            np.arange(data.shape[1], dtype=np.float64) * scale["z"] + translation["z"],
            long_name=z_axis_name,
            units=z_axis_units,
        )

        y_axis_name = "y"
        if axis_names and "y" in axis_names:
            y_axis_name = axis_names["y"]
        y_axis_units = ""
        if axis_units and "y" in axis_units:
            y_axis_units = axis_units["y"]
        self.y = YAxis(
            np.arange(data.shape[2], dtype=np.float64) * scale["y"] + translation["y"],
            long_name=y_axis_name,
            units=y_axis_units,
        )

        x_axis_name = "x"
        if axis_names and "x" in axis_names:
            x_axis_name = axis_names["x"]
        x_axis_units = ""
        if axis_units and "x" in axis_units:
            x_axis_units = axis_units["x"]
        self.x = XAxis(
            np.arange(data.shape[3], dtype=np.float64) * scale["x"] + translation["x"],
            long_name=x_axis_name,
            units=x_axis_units,
        )

        t_axis_name = "t"
        if axis_names and "t" in axis_names:
            t_axis_name = axis_names["t"]
        t_axis_units = ""
        if axis_units and "t" in axis_units:
            t_axis_units = axis_units["t"]
        if t_coords is None:
            t_coords = np.arange(data.shape[0])
        self.t = TAxis(t_coords, t_axis_name, t_axis_units)


@dataclass(init=False)
class SpatialImageTZYXCDataClass(SpatialImageTZYXDataClass):
    """A 3D spatial image with a time dimension and channels."""

    data: Data[Tuple[T, Z, Y, X, C], Any]
    c: Coordof[CAxis]

    def __init__(
        self,
        data,
        scale: Optional[Union[Mapping[Hashable, float]]] = None,
        translation: Optional[Union[Mapping[Hashable, float]]] = None,
        name: str = default_name,
        axis_names: Optional[Union[Mapping[Hashable, str]]] = None,
        axis_units: Optional[Union[Mapping[Hashable, str]]] = None,
        t_coords: Optional[Sequence[Union[AllInteger, AllFloat, np.datetime64]]] = None,
        c_coords: Optional[Sequence[Union[AllInteger, str]]] = None,
    ):
        super().__init__(
            data, scale, translation, name, axis_names, axis_units, t_coords
        )
        if c_coords is None:
            c_coords = np.arange(data.shape[4])

        c_axis_name = "c"
        if axis_names is not None and "c" in axis_names:
            c_axis_name = axis_names["c"]

        c_axis_units = ""
        if axis_units is not None and "c" in axis_units:
            c_axis_units = axis_units["c"]

        self.c = CAxis(c_coords, c_axis_name, c_axis_units)


@dataclass(init=False)
class SpatialImageCXDataClass(SpatialImageDataClass):
    """A 1D spatial image with channels, channel first."""

    data: Data[Tuple[C, X], Any]
    c: Coordof[CAxis]
    x: Coordof[XAxis]

    def __init__(
        self,
        data,
        scale: Optional[Union[Mapping[Hashable, float]]] = None,
        translation: Optional[Union[Mapping[Hashable, float]]] = None,
        name: str = default_name,
        axis_names: Optional[Union[Mapping[Hashable, str]]] = None,
        axis_units: Optional[Union[Mapping[Hashable, str]]] = None,
        c_coords: Optional[Sequence[Union[AllInteger, str]]] = None,
    ):
        super().__init__(name)
        self.data = data

        if scale is None:
            scale = {"x": 1.0}

        if translation is None:
            translation = {"x": 0.0}

        x_axis_name = "x"
        if axis_names and "x" in axis_names:
            x_axis_name = axis_names["x"]
        x_axis_units = ""
        if axis_units and "x" in axis_units:
            x_axis_units = axis_units["x"]
        self.x = XAxis(
            np.arange(data.shape[1], dtype=np.float64) * scale["x"] + translation["x"],
            long_name=x_axis_name,
            units=x_axis_units,
        )

        if c_coords is None:
            c_coords = np.arange(data.shape[0])

        c_axis_name = "c"
        if axis_names is not None and "c" in axis_names:
            c_axis_name = axis_names["c"]
        c_axis_units = ""
        if axis_units and "c" in axis_units:
            c_axis_units["c"] = ""

        self.c = CAxis(c_coords, c_axis_name, c_axis_units)


@dataclass(init=False)
class SpatialImageTCXDataClass(SpatialImageDataClass):
    """A 1D spatial image with a time dimension and channels, channel first."""

    data: Data[Tuple[T, C, X], Any]
    t: Coordof[TAxis]
    c: Coordof[CAxis]
    x: Coordof[XAxis]

    def __init__(
        self,
        data,
        scale: Optional[Union[Mapping[Hashable, float]]] = None,
        translation: Optional[Union[Mapping[Hashable, float]]] = None,
        name: str = default_name,
        axis_names: Optional[Union[Mapping[Hashable, str]]] = None,
        axis_units: Optional[Union[Mapping[Hashable, str]]] = None,
        t_coords: Optional[Sequence[Union[AllInteger, AllFloat, np.datetime64]]] = None,
        c_coords: Optional[Sequence[Union[AllInteger, str]]] = None,
    ):
        super().__init__(name)
        self.data = data

        if scale is None:
            scale = {"x": 1.0}

        if translation is None:
            translation = {"x": 0.0}

        t_axis_name = "t"
        if axis_names and "t" in axis_names:
            t_axis_name = axis_names["t"]
        t_axis_units = ""
        if axis_units and "t" in axis_units:
            t_axis_units = axis_units["t"]
        if t_coords is None:
            t_coords = np.arange(data.shape[0])

        self.t = TAxis(t_coords, t_axis_name, t_axis_units)
        x_axis_name = "x"
        if axis_names and "x" in axis_names:
            x_axis_name = axis_names["x"]
        x_axis_units = ""
        if axis_units and "x" in axis_units:
            x_axis_units = axis_units["x"]
        self.x = XAxis(
            np.arange(data.shape[2], dtype=np.float64) * scale["x"] + translation["x"],
            long_name=x_axis_name,
            units=x_axis_units,
        )

        if c_coords is None:
            c_coords = np.arange(data.shape[1])

        c_axis_name = "c"
        if axis_names is not None and "c" in axis_names:
            c_axis_name = axis_names["c"]
        c_axis_units = ""
        if axis_units and "c" in axis_units:
            c_axis_units["c"] = ""

        self.c = CAxis(c_coords, c_axis_name, c_axis_units)


@dataclass(init=False)
class SpatialImageCYXDataClass(SpatialImageDataClass):
    """A 2D spatial image with a time dimension and channels, channel first."""

    data: Data[Tuple[C, Y, X], Any]
    c: Coordof[CAxis]
    y: Coordof[YAxis]
    x: Coordof[XAxis]

    def __init__(
        self,
        data,
        scale: Optional[Union[Mapping[Hashable, float]]] = None,
        translation: Optional[Union[Mapping[Hashable, float]]] = None,
        name: str = default_name,
        axis_names: Optional[Union[Mapping[Hashable, str]]] = None,
        axis_units: Optional[Union[Mapping[Hashable, str]]] = None,
        c_coords: Optional[Sequence[Union[AllInteger, str]]] = None,
    ):
        super().__init__(name)
        self.data = data

        if scale is None:
            scale = {"y": 1.0, "x": 1.0}

        if translation is None:
            translation = {"y": 0.0, "x": 0.0}

        x_axis_name = "x"
        if axis_names and "x" in axis_names:
            x_axis_name = axis_names["x"]
        x_axis_units = ""
        if axis_units and "x" in axis_units:
            x_axis_units = axis_units["x"]
        self.x = XAxis(
            np.arange(data.shape[2], dtype=np.float64) * scale["x"] + translation["x"],
            long_name=x_axis_name,
            units=x_axis_units,
        )

        y_axis_name = "y"
        if axis_names and "y" in axis_names:
            y_axis_name = axis_names["y"]
        y_axis_units = ""
        if axis_units and "y" in axis_units:
            y_axis_units = axis_units["y"]
        self.y = YAxis(
            np.arange(data.shape[1], dtype=np.float64) * scale["y"] + translation["y"],
            long_name=y_axis_name,
            units=y_axis_units,
        )

        if c_coords is None:
            c_coords = np.arange(data.shape[0])

        c_axis_name = "c"
        if axis_names is not None and "c" in axis_names:
            c_axis_name = axis_names["c"]
        c_axis_units = ""
        if axis_units and "c" in axis_units:
            c_axis_units["c"] = ""

        self.c = CAxis(c_coords, c_axis_name, c_axis_units)


@dataclass(init=False)
class SpatialImageTCYXDataClass(SpatialImageDataClass):
    """A 2D spatial image with a time dimension and channels, channel first."""

    data: Data[Tuple[T, C, Y, X], Any]
    t: Coordof[TAxis]
    c: Coordof[CAxis]
    y: Coordof[YAxis]
    x: Coordof[XAxis]

    def __init__(
        self,
        data,
        scale: Optional[Union[Mapping[Hashable, float]]] = None,
        translation: Optional[Union[Mapping[Hashable, float]]] = None,
        name: str = default_name,
        axis_names: Optional[Union[Mapping[Hashable, str]]] = None,
        axis_units: Optional[Union[Mapping[Hashable, str]]] = None,
        t_coords: Optional[Sequence[Union[AllInteger, AllFloat, np.datetime64]]] = None,
        c_coords: Optional[Sequence[Union[AllInteger, str]]] = None,
    ):
        super().__init__(name)
        self.data = data

        t_axis_name = "t"
        if axis_names and "t" in axis_names:
            t_axis_name = axis_names["t"]
        t_axis_units = ""
        if axis_units and "t" in axis_units:
            t_axis_units = axis_units["t"]
        if t_coords is None:
            t_coords = np.arange(data.shape[0])

        self.t = TAxis(t_coords, t_axis_name, t_axis_units)
        if scale is None:
            scale = {"y": 1.0, "x": 1.0}

        if translation is None:
            translation = {"y": 0.0, "x": 0.0}

        x_axis_name = "x"
        if axis_names and "x" in axis_names:
            x_axis_name = axis_names["x"]
        x_axis_units = ""
        if axis_units and "x" in axis_units:
            x_axis_units = axis_units["x"]
        self.x = XAxis(
            np.arange(data.shape[3], dtype=np.float64) * scale["x"] + translation["x"],
            long_name=x_axis_name,
            units=x_axis_units,
        )

        y_axis_name = "y"
        if axis_names and "y" in axis_names:
            y_axis_name = axis_names["y"]
        y_axis_units = ""
        if axis_units and "y" in axis_units:
            y_axis_units = axis_units["y"]
        self.y = YAxis(
            np.arange(data.shape[2], dtype=np.float64) * scale["y"] + translation["y"],
            long_name=y_axis_name,
            units=y_axis_units,
        )

        if c_coords is None:
            c_coords = np.arange(data.shape[1])

        c_axis_name = "c"
        if axis_names is not None and "c" in axis_names:
            c_axis_name = axis_names["c"]
        c_axis_units = ""
        if axis_units and "c" in axis_units:
            c_axis_units["c"] = ""

        self.c = CAxis(c_coords, c_axis_name, c_axis_units)


@dataclass(init=False)
class SpatialImageTCZYXDataClass(SpatialImageDataClass):
    """A 2D spatial image with a time dimension and channels, channel first."""

    data: Data[Tuple[T, C, Z, Y, X], Any]
    t: Coordof[TAxis]
    c: Coordof[CAxis]
    z: Coordof[ZAxis]
    y: Coordof[YAxis]
    x: Coordof[XAxis]

    def __init__(
        self,
        data,
        scale: Optional[Union[Mapping[Hashable, float]]] = None,
        translation: Optional[Union[Mapping[Hashable, float]]] = None,
        name: str = default_name,
        axis_names: Optional[Union[Mapping[Hashable, str]]] = None,
        axis_units: Optional[Union[Mapping[Hashable, str]]] = None,
        t_coords: Optional[Sequence[Union[AllInteger, AllFloat, np.datetime64]]] = None,
        c_coords: Optional[Sequence[Union[AllInteger, str]]] = None,
    ):
        super().__init__(name)
        self.data = data

        t_axis_name = "t"
        if axis_names and "t" in axis_names:
            t_axis_name = axis_names["t"]
        t_axis_units = ""
        if axis_units and "t" in axis_units:
            t_axis_units = axis_units["t"]
        if t_coords is None:
            t_coords = np.arange(data.shape[0])

        self.t = TAxis(t_coords, t_axis_name, t_axis_units)
        if scale is None:
            scale = {"z": 1.0, "y": 1.0, "x": 1.0}

        if translation is None:
            translation = {"z": 0.0, "y": 0.0, "x": 0.0}

        z_axis_name = "z"
        if axis_names and "z" in axis_names:
            z_axis_name = axis_names["z"]
        z_axis_units = ""
        if axis_units and "z" in axis_units:
            z_axis_units = axis_units["z"]
        self.z = ZAxis(
            np.arange(data.shape[2], dtype=np.float64) * scale["z"] + translation["z"],
            long_name=z_axis_name,
            units=z_axis_units,
        )

        x_axis_name = "x"
        if axis_names and "x" in axis_names:
            x_axis_name = axis_names["x"]
        x_axis_units = ""
        if axis_units and "x" in axis_units:
            x_axis_units = axis_units["x"]
        self.x = XAxis(
            np.arange(data.shape[4], dtype=np.float64) * scale["x"] + translation["x"],
            long_name=x_axis_name,
            units=x_axis_units,
        )

        y_axis_name = "y"
        if axis_names and "y" in axis_names:
            y_axis_name = axis_names["y"]
        y_axis_units = ""
        if axis_units and "y" in axis_units:
            y_axis_units = axis_units["y"]
        self.y = YAxis(
            np.arange(data.shape[3], dtype=np.float64) * scale["y"] + translation["y"],
            long_name=y_axis_name,
            units=y_axis_units,
        )

        if c_coords is None:
            c_coords = np.arange(data.shape[1])

        c_axis_name = "c"
        if axis_names is not None and "c" in axis_names:
            c_axis_name = axis_names["c"]
        c_axis_units = ""
        if axis_units and "c" in axis_units:
            c_axis_units["c"] = ""

        self.c = CAxis(c_coords, c_axis_name, c_axis_units)


def is_spatial_image(image: Any) -> bool:
    """Verify that the image 'quacks like a spatial-image'.

    Parameters
    ----------

    image: spatial_image
        The spatial image to verify.

    Returns
    -------

    bool
        Verification tests result.
    """
    if not isinstance(image, xr.DataArray):
        return False

    if not set(image.dims).issubset(_supported_dims):
        return False

    for dim in _spatial_dims.intersection(image.dims):
        if not image.coords[dim].dtype == np.float64:
            return False

        diff = np.diff(image.coords[dim])
        if not np.allclose(diff, diff[0]):
            return False

    if "t" in image.dims:
        t_coord = image.coords["t"]
        if (
            t_coord.dtype.char not in np.typecodes["AllInteger"]
            and t_coord.dtype.char not in np.typecodes["AllFloat"]
            and t_coord.dtype.char not in np.typecodes["Datetime"]
        ):
            return False

    return True


SpatialImageDataClasses = {
    ("x",): SpatialImageXDataClass,
    ("x", "c"): SpatialImageXCDataClass,
    ("t", "x"): SpatialImageTXDataClass,
    ("t", "x", "c"): SpatialImageTXCDataClass,
    ("y", "x"): SpatialImageYXDataClass,
    ("y", "x", "c"): SpatialImageYXCDataClass,
    ("t", "y", "x"): SpatialImageTYXDataClass,
    ("t", "y", "x", "c"): SpatialImageTYXCDataClass,
    ("z", "y", "x"): SpatialImageZYXDataClass,
    ("z", "y", "x", "c"): SpatialImageZYXCDataClass,
    ("t", "z", "y", "x"): SpatialImageTZYXDataClass,
    ("t", "z", "y", "x", "c"): SpatialImageTZYXCDataClass,
    ("c", "x"): SpatialImageCXDataClass,
    ("t", "c", "x"): SpatialImageTCXDataClass,
    ("c", "y", "x"): SpatialImageCYXDataClass,
    ("t", "c", "y", "x"): SpatialImageTCYXDataClass,
    ("t", "c", "z", "y", "x"): SpatialImageTCZYXDataClass,
}


def to_spatial_image(
    array_like: Any,
    dims: Optional[Sequence[Union["t", "z", "y", "x", "c"]]] = None,
    scale: Optional[Union[Mapping[Hashable, float]]] = None,
    translation: Optional[Union[Mapping[Hashable, float]]] = None,
    name: str = default_name,
    axis_names: Optional[Union[Mapping[Hashable, str]]] = None,
    axis_units: Optional[Union[Mapping[Hashable, str]]] = None,
    t_coords: Optional[Sequence[Union[AllInteger, AllFloat, np.datetime64]]] = None,
    c_coords: Optional[Sequence[Union[AllInteger, str]]] = None,
) -> SpatialImage:
    """Convert the array-like to a spatial-image.

    Parameters
    ----------

    array_like:
        Multi-dimensional array that provides the image pixel values.

    dims: sequence of hashable, optional
        Tuple specifying the data dimensions.
        Values should drawn from: {'t', 'z', 'y', 'x', 'c'} for time, third spatial direction
        second spatial direction, first spatial dimension, and channel or
        component, respectively spatial dimension, and time, respectively.

    scale: dict of floats, optional
        Pixel spacing for the spatial dims

    translation: dict of floats, optional
        Origin or offset of the center of the first pixel.

    name: str, optional
        Name of the resulting xarray DataArray

    axis_names: dict of str, optional
        Long names for the dim axes, e.g. {'x': 'x-axis'} or {'x': 'anterior-posterior'}

    axis_units: dict of str, optional
        Units names for the dim axes, e.g. {'x': 'millimeters', 't': 'seconds'}

    c_coords: sequence integers or strings, optional
        If there is a 'c' dim, the coordiantes for this channel/component dimension.
        A sequence of integers by default but can be strings describing the
        channels, e.g. ['r', 'g', 'b'].

    t_coords: sequence of integers, strings or datetime64, optional
        The 't' time coords can have int, float, or datetime64 type.

    Returns
    -------

    spatial_image
        SpatialImage corresponding to the array and provided metadata.
    """

    ndim = array_like.ndim
    if dims is None:
        if ndim < 4:
            dims = ("z", "y", "x")[-ndim:]
        elif ndim < 5:
            dims = ("z", "y", "x", "c")
        elif ndim < 6:
            dims = ("t", "z", "y", "x", "c")
        else:
            raise ValueError("Unsupported dimension: " + str(ndim))
    else:
        if not set(dims).issubset(_supported_dims):
            raise ValueError("dims not valid for a SpatialImage")

    dims = tuple(dims)
    if dims not in SpatialImageDataClasses:
        raise ValueError("The dims provided are not supported yet")

    SIDataClass = SpatialImageDataClasses[dims]
    si_kwargs = {
        "scale": scale,
        "translation": translation,
        "name": name,
        "axis_names": axis_names,
        "axis_units": axis_units,
    }
    if "c" in dims:
        si_kwargs["c_coords"] = c_coords
    if "t" in dims:
        si_kwargs["t_coords"] = t_coords

    image = SIDataClass.new(array_like, **si_kwargs)

    return image
