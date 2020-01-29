# spatial-image

A multi-dimensional spatial image data structure for scientific Python.

To facilitate:

- Multi-scale processing and analysis
- Registration
- Resampling

on scientific images, which are typically multi-dimensional with anisotropic
sampling, this package provides a spatial-image data structure.

Spatial image metadata is defined, a function, `is_spatial_image`, verifies
the expected behavior of a spatial image instance, and a reference function,
`to_spatial_image` converts an array-like, e.g. a [NumPy
ndarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html)
or a [Dask array](https://docs.dask.org/en/latest/array.html), to a spatial
image.
