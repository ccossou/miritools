Welcome to MIRITools's documentation!
===================================

The **MIRITools** package contains convenience functions usefull to display or characterize MIRI data on board JWST. Some are more generic and could be usefull in general, but all of them have been created with MIRI in mind in some way or another.

MIRITools start at v4.0.0. This is for historical reasons because it is based on the miricap package and I kept the version number for convenience.

MIRITools is split into several sub-packages:

* **coord**: Everything related to sky coordinates or pixel coordinates
* **flux**: Everything related to flux conversions, magnitude, Jansky or photon
* **imager**: detector image manipulation, pixel coordinates from FULL array to subarray, subarray intersection, radial profiles
* **mask**: Everything related to Data Quality and how to manipulate theses complex informations
* **plot**: Usefull plots to quickly display information (e.g. :ref:`single_image` is the equivalent to ZScale in DS9)
* **read**: Reading functions for the various MIRI data products
* **utils**: Various utility functions, the ones I advise you to check are :ref:`optimum_nbins` and :ref:`mast_reorder`
* **write**: Functions to write fits files or thumbnails quickly

Check :ref:`installation` the project.

Contents
--------

.. toctree::

   intro
   sub-package
   api

