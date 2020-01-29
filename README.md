# spatial-image

A multi-dimensional spatial image data structure for scientific Python.

To facilitate:

- Multi-scale processing and analysis
- Registration
- Resampling

on scientific images, which are typically multi-dimensional with anisotropic
sampling, this package provides a spatial-image data structure. In addition to
an N-dimensional array of pixel values, spatial metadata defines the location
of the pixel sampling grid in space time. We also label the array dimensions.
This metadata is easily utilized and elegantly carried through image
processing pipelines.

This package defines spatial image metadata, provides a function,
`is_spatial_image`, to verify the expected behavior of a spatial image
instance, and provides a reference function, `to_spatial_image` to convert an
array-like, e.g. a [NumPy
ndarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html)
or a [Dask array](https://docs.dask.org/en/latest/array.html), to a spatial
image.

The spatial-image data structure is implemented with
[xarray](https://xarray.pydata.org/en/stable/), a library for N-D labeled
arrays and datasets in Python. The xarray library is well-tested, relatively
mature, and integrates well with scientific Python ecosystem tooling. The
xarray library leverages [NumPy](https://numpy.org/) and
[pandas](https://pandas.pydata.org/) for labeled array indexing, integrates
well with machine-learning libraries utilizing the
[scikit-learn](https://scikit-learn.org/) interface, integrates with
[Dask](https://dask.org) for distributed computing, and
[zarr](https://zarr.readthedocs.io/) for serialization.

In essence, a spatial image is an
[`xarray.DataArray`](https://xarray.pydata.org/en/stable/data-structures.html#dataarray) with a defined set of
[`dims`](https://xarray.pydata.org/en/stable/terminology.html) labels,
`('c', 'x', 'y', 'z', 't')`,
constraints on the
[`coords`](https://xarray.pydata.org/en/stable/terminology.html), to
enforce uniform spacing in a given direction, and defined set of
additional metadata [`attrs`](https://xarray.pydata.org/en/stable/data-structures.html).

## Installation

```
pip install spatial-image
```

## Development

Contributions are welcome and appreciated.

To run the test suite:

```
git clone https://github.com/spatial-image/spatial-image
cd spatial-image
pip install -r requirements.txt -r requirements-dev.txt
pytest
```
