
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

