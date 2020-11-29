.. region-grow documentation master file, created by
   sphinx-quickstart on Sat Nov 28 18:25:27 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Region Grow!
=======================================

This library allows you to create a polygon
using a set of points from a region of interest by grouping pixels
whose spectral reflectance is similar. The polygons are created using
a satellite image in GeoTIFF format. In this project several algorithms
are implemented to build this figure. Among them are: Selection by similarity threshold (%),
Euclidean distance and selection by confidence interval. 
The generated polygon is exported in ESRI Shapefile format.

Installation
--------------------------

You can easily install this package from PyPi

.. code:: ipython3

    pip install region-grow


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   example
   contribution

.. toctree::
   :maxdepth: 2
   :caption: API:

   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
