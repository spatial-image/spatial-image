# spatial-image

[![Test](https://github.com/spatial-image/spatial-image/actions/workflows/test.yml/badge.svg)](https://github.com/spatial-image/spatial-image/actions/workflows/test.yml)
[![DOI](https://zenodo.org/badge/234798632.svg)](https://zenodo.org/badge/latestdoi/234798632)

A multi-dimensional spatial image data structure for scientific Python.

To facilitate:

- Multi-scale processing and analysis
- Registration
- Resampling
- Subregion parallel processing
- Coupling with meshes, point sets, and annotations

with scientific images, which are typically multi-dimensional with anisotropic
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

The spatial-image data structure is implemented with [Xarray], a library for
N-D labeled arrays and datasets in Python. The Xarray library is well-tested,
relatively mature, and integrates well with scientific Python ecosystem
tooling. The Xarray library leverages [NumPy](https://numpy.org/) and
[pandas](https://pandas.pydata.org/) for labeled array indexing, integrates
well with machine-learning libraries utilizing the
[scikit-learn](https://scikit-learn.org/) interface, integrates with
[Dask](https://dask.org) for distributed computing, and
[zarr](https://zarr.readthedocs.io/) for serialization.

In essence, a spatial image is an
[`xarray.DataArray`](https://xarray.pydata.org/en/stable/data-structures.html#dataarray)
with a defined set of [`dims`] labels, `{'c', 'x', 'y', 'z', 't'}`,
constraints on the [`coords`], to enforce uniform spacing in a given
direction, and defined set of additional metadata [`attrs`].

## Installation

```
pip install spatial-image
```

## Definitions

### Data Dimensions

A spatial image's xarray [`dims`] belong to the set: `{'c', 'x', 'y', 'z', 't'}`. These dimensions are:

<dl>
  <dt>c</dt>
  <dd>Component / channel dimension.</dd>
  <dt>x</dt>
  <dd>First spatial dimension.</dd>
  <dt>y</dt>
  <dd>Second spatial dimension.</dd>
  <dt>z</dt>
  <dd>Third spatial dimension.</dd>
  <dt>t</dt>
  <dd>Time dimension.</dd>
</dl>

### Axis attributes

Each `dim` has an axis with additional attributes to describe the dimension.

<dl>
  <dt>long_name</dt>
  <dd>A descriptive name for the axis, e.g. <i>anterior-posterior</i> or <i>x-axis</i>. Defaults to the dim name.</dd>
  <dt>units</dt>
  <dd>Units for the axis, e.g. <i>millimeters</i>. Defaults to the empty string.</dd>
</dl>

### Coordinates

A spatial image's Xarray [`coords`] specify the spatial location of pixels in
the image for the `'x'`, `'y'`, and `'z'` data dimensions.  For the `'c'` and
`'t'` data dimensions, component identities and timestamps can optionally
be provided.

Spatial coordinates define the position *in the coordinate reference frame of
the image*. In general, the image's coordinate reference frame may be
different from the world coordinate reference frame.

Pixels are sampled on a uniform, possibly anisotropic, spatial grid.  Spatial
coordinates have a 64-bit float type. The difference between adjacent
coordinates, i.e. the pixel *spacing*, for a dimension must be uniform. The
first coordinate value defines the *origin* or *offset* of an image.

The component or channel dimension coordinates defaults to a sequence of
integer identifiers but can be strings describing the channels, e.g. ['r',
'g', 'b'].

The time coordinates can have integer, float, or [`datetime64`] type.

### Motivational Notes

* Image-axis-aligned Cartesian coordinate reference frames enable Pythonic subscripting in processing pipelines on `xarray.DataArray`'s. When indexing with slices, the same slices are applied to the multi-dimensional pixel array as the 1-D coordinate arrays, and the result is valid.

* Regular coordinate spacing enables processing optimizations, both algorithmically and computationally.


## Development

Contributions are welcome and appreciated.

To run the test suite:

```
git clone https://github.com/spatial-image/spatial-image
cd spatial-image
pip install -e ".[test]"
pytest
```

[Xarray]: https://xarray.pydata.org/en/stable/
[`dims`]: https://xarray.pydata.org/en/stable/terminology.html
[`coords`]: https://xarray.pydata.org/en/stable/terminology.html
[`attrs`]: https://xarray.pydata.org/en/stable/data-structures.html
[`datetime64`]: https://docs.scipy.org/doc/numpy/reference/arrays.datetime.html
