======================
Setup and Installation
======================

Dependencies
############
This software is dependent on open source programs that can be installed using OS-specific package management systems,
`conda <https://anaconda.org/conda-forge/repo>`_ or from source:

- `GDAL <https://gdal.org/index.html>`_
- `GEOS <https://trac.osgeo.org/geos>`_
- `PROJ <https://proj.org/>`_
- `HDF5 <https://www.hdfgroup.org>`_
- `netCDF <https://www.unidata.ucar.edu/software/netcdf/>`_

The version of GDAL used within ``pointAdvection`` will match the version of the installed C program.
The path to the C program that will be used with ``pointAdvection`` is given by:

.. code-block:: bash

    gdal-config --datadir

The ``pointAdvection`` installation uses the ``gdal-config`` routines to set the GDAL package version.

Installation
############

Presently ``pointAdvection`` is only available for use as a
`GitHub repository <https://github.com/tsutterley/pointAdvection>`_.
The contents of the repository can be downloaded as a
`zipped file <https://github.com/tsutterley/pointAdvection/archive/main.zip>`_  or cloned.
To use this repository, please fork into your own account and then clone onto your system.

.. code-block:: bash

    git clone https://github.com/tsutterley/pointAdvection.git

Can then install using ``pip``:

.. code-block:: bash

    python3 -m pip install --user .

To include all optional dependencies:

.. code-block:: bash

   python3 -m pip install --user .[all]

Alternatively can install the utilities directly from GitHub with ``pip``:

.. code-block:: bash

    python3 -m pip install --user git+https://github.com/tsutterley/pointAdvection.git
