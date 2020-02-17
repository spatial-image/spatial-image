"""spatial-image

A multi-dimensional spatial image data structure for Python."""

__version__ = "0.0.1"

import xarray as xr

_supported_dims = {"c", "x", "y", "z", "t"}


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

    return True


def to_spatial_image(array_like, dims=None):
    """Convert the array-like to a spatial-image.

    Parameters
    ----------

    array_like:
        Multi-dimensional array that provides the image pixel values.

    dims: sequence, optional
        Tuple specifying the data dimensions.
        Values should drawn from: {'c', 'x', 'y', 'z', 't'}.

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

    image = xr.DataArray(array_like, dims=dims)

    return image
