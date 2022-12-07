
.. _installation:

How to install
------------------
The latest version of this software can be retrieved from the repository hosted at:
https://github.com/ccossou/miritools

This package is a simple Python module, and all default commands
works, such as:

..  code-block:: console

    pip install miritools

How to use
-----------------
* First import the package:

..  code-block:: python

    import miritools
    # or
    from miritools import mask, read, plot, flux  #...

* To get help on a given function, either look at the documentation or use the built-in *help* function:

..  code-block:: python

    from miritools import plot
    help(plot.single_image)

Description
-----------

MIRITools is split into several sub-packages:

* **coord**: Everything related to sky coordinates or pixel coordinates
* **flux**: Everything related to flux conversions, magnitude, Jansky or photon
* **imager**: detector image manipulation, pixel coordinates from FULL array to subarray, subarray intersection, radial profiles
* **mask**: Everything related to Data Quality and how to manipulate theses complex informations
* **plot**: Usefull plots to quickly display information (e.g. :ref:`single_image` is the equivalent to ZScale in DS9)
* **read**: Reading functions for the various MIRI data products
* **utils**: Various utility functions, the ones I advise you to check are :ref:`optimum_nbins` and :ref:`mast_reorder`
* **write**: Functions to write fits files or thumbnails quickly
