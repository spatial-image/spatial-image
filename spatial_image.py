"""spatial-image

A multi-dimensional spatial image data structure for Python."""

__version__ = '0.0.1'

import xarray as xr

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

    return True

def to_spatial_image(array_like):
    """Convert the array-like to a spatial-image.

    Parameters
    ----------

    array_like:
        Multi-dimensional array that provides the image pixel values.

    Returns
    -------

    spatial_image
        Spatial image corresponding to the array and provided metadata.
    """

    image = xr.DataArray(array_like)

    return image
