"""spatial-image

A multi-dimensional spatial image data structure for Python."""

__version__ = "0.0.2"

import xarray as xr
import numpy as np

_supported_dims = {"c", "x", "y", "z", "t"}
_spatial_dims = {"x", "y", "z"}


def is_spatial_image(image):
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


def to_spatial_image(array_like, dims=None, coords=None):
    """Convert the array-like to a spatial-image.

    Parameters
    ----------

    array_like:
        Multi-dimensional array that provides the image pixel values.

    dims: hashable or sequence of hashable, optional
        Tuple specifying the data dimensions.
        Values should drawn from: {'c', 'x', 'y', 'z', 't'} for channel or
        component, first spatial direction, second spatial direction, third
        spatial dimension, and time, respectively.

    coords: sequence or dict of array_like objects, optional
        For each {'x', 'y', 'z'} dim, 1-D np.float64 array specifing the
        pixel location in the image's local coordinate system. The distance
        between subsequent coords elements should be uniform. The 'c' coords
        are a sequence of integers by default but can be strings describing the
        channels, e.g. ['r', 'g', 'b']. The 't' coords can have int, float,
        or datetime64 type.

    Returns
    -------

    spatial_image
        Spatial image corresponding to the array and provided metadata.
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
            raise ValueError("dims not valid for a spatial image")

    if coords is None:
        coords = {}
        for index, dim in enumerate(dims):
            if dim in ("c", "t"):
                coords[dim] = np.arange(array_like.shape[index])
            else:
                coords[dim] = np.arange(array_like.shape[index], dtype=np.float64)

    image = xr.DataArray(array_like, coords=coords, dims=dims)

    return image
