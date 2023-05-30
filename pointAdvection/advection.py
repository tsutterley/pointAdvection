#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
advection.py
Written by Tyler Sutterley and Ben Smith (05/2023)
Routines for advecting ice parcels using velocity estimates

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    netCDF4: Python interface to the netCDF C library
         https://unidata.github.io/netcdf4-python/netCDF4/index.html
    gdal: Pythonic interface to the Geospatial Data Abstraction Library (GDAL)
        https://pypi.python.org/pypi/GDAL/
    matplotlib: Python 2D plotting library
        http://matplotlib.org/
        https://github.com/matplotlib/matplotlib
    pointCollection: Utilities for organizing and manipulating point data
        https://github.com/SmithB/pointCollection

UPDATE HISTORY:
    Updated 05/2023: add fill gaps function and xy0 interpolator
        add option to advect parcels to set the number of steps directly
        using pathlib to define and expand paths
    Updated 03/2023: added function for extracting from a dictionary
        verify input times are float64 arrays
        set default interpolator to linear regular grid
        add case for using regular grid interpolation with 2d velocities
    Updated 10/2022: added option to plot divergence of velocity field
        added streaklines based on a velocity field
    Updated 08/2022: verify datatype of imported velocity fields
        add interpolation and plot routines for unstructured meshes
        place some imports within try/except statements
    Updated 06/2022: added velocity and streamline plot routine
        using numpy nan_to_num function to convert NaN values
    Updated 05/2022: verify that input spatial coordinates are doubles
    Updated 04/2022: updated docstrings to numpy documentation format
    Updated 02/2022: converted to a python class using pointCollection
        added spline and regular grid interpolators
    Updated 02/2018: added catch for points outside image extents
        added adaptive Runge-Kutta-Fehlberg method
    Written 01/2018
"""
from __future__ import annotations

import os
import io
import re
import copy
import logging
import pathlib
import warnings
import numpy as np
import scipy.interpolate
import scipy.spatial
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
# attempt imports
try:
    import pointCollection as pc
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("pointCollection not available", ImportWarning)
# ignore warnings
warnings.filterwarnings("ignore")

class advection():
    """
    Data class for advecting ice parcels using velocity estimates

    Attributes
    ----------
    x: np.ndarray
        x-coordinates
    y: np.ndarray
        y-coordinates
    t: np.ndarray
        time coordinates
    x0: np.ndarray or NoneType, default None
        Final x-coordinate after advection
    y0: np.ndarray or NoneType, default None
        Final y-coordinate after advection
    t0: np.ndarray or float, default 0.0
        Ending time for advection
    velocity: obj
        ``pointCollection`` object of velocity fields

        Can be type ``grid`` or ``mesh``
    streak: dict
        path traversed during advection (set kwarg 'streak' to true
                        in translate methods to enable)
    filename: str
        input filename of velocity file
    integrator: str
        Advection function

            - ``'euler'``
            - ``'RK4'``
            - ``'RKF45'``
    method: str, default 'linear'
        Interpolation method for velocities

            - ``'bilinear'``: quick bilinear interpolation
            - ``'spline'``: scipy bivariate spline interpolation
            - ``'linear'``, ``'nearest'``: scipy regular grid interpolations
            - ``'linearND'``, ``'nearestND'``: scipy unstructured N-dimensional interpolations
    interpolant: obj
        Interpolation function for velocity fields
    fill_value: float or NoneType, default np.nan
        invalid value for output data
    """
    np.seterr(invalid='ignore')
    def __init__(self, **kwargs):
        # set default keyword arguments
        kwargs.setdefault('x', None)
        kwargs.setdefault('y', None)
        kwargs.setdefault('t', None)
        kwargs.setdefault('t0', 0.0)
        kwargs.setdefault('integrator', 'RK4')
        kwargs.setdefault('method', 'linear')
        kwargs.setdefault('fill_value', np.nan)
        # set default class attributes
        self.x=np.atleast_1d(kwargs['x']).astype('f8')
        self.y=np.atleast_1d(kwargs['y']).astype('f8')
        self.t=np.array(kwargs['t'], dtype='f8')
        self.x0=None
        self.y0=None
        self.t0=np.array(kwargs['t0'], dtype='f8')
        self.velocity=None
        self.streak={'x':[], 'y':[],'t':[]}
        self.filename=None
        self.integrator=copy.copy(kwargs['integrator'])
        self.method=copy.copy(kwargs['method'])
        self.interpolant={}
        self.xy0_interpolator=None
        self.fill_value=kwargs['fill_value']

    def __update_streak__(self, t, **kwargs):
        """
        Update the streakline during flow tracing.
        """
        if kwargs['streak']:
            self.streak['x'] += [self.x0.copy()]
            self.streak['y'] += [self.y0.copy()]
            self.streak['t'] += [t.copy()]

    def case_insensitive_filename(self, filename: str | io.BytesIO):
        """
        Searches a directory for a filename without case dependence

        Parameters
        ----------
        filename: str
            input filename
        """
        # check if filename is open file object
        if isinstance(filename, io.IOBase):
            self.filename = copy.copy(filename)
        else:
            # tilde-expand input filename
            self.filename = pathlib.Path(filename).expanduser().absolute()
            # check if file presently exists with input case
            if not self.filename.exists():
                # search for filename without case dependence
                f = [f.name for f in self.filename.parent.iterdir() if
                    re.match(self.filename.name, f.name, re.I)]
                if not f:
                    errmsg = f'{filename} not found in file system'
                    raise FileNotFoundError(errmsg)
                self.filename = self.filename.with_name(f.pop())
        # print filename
        logging.debug(self.filename)
        return self

    # PURPOSE: read geotiff velocity file and extract x and y velocities
    def from_geotiff(self,
            filename: str | io.IOBase,
            bounds: list | np.ndarray | None = None,
            buffer: float | None = 5e4,
            scale: float = 1.0/31557600.0
        ):
        """
        Read geotiff velocity file and extract x and y velocities

        Parameters
        ----------
        filename: str
            geotiff velocity file
        bounds: list or NoneType, default None
            boundaries to read: ``[[xmin, xmax], [ymin, ymax]]``

            If not specified, can read around input points
        buffer: float or NoneType, default 5e4
            Buffer around input points for extracting velocity fields
        scale: float, default 1.0/31557600.0
            scaling factor for converting velocities to m/s

            defaults to converting from m/yr
        """
        # find input geotiff velocity file
        self.case_insensitive_filename(filename)
        # x and y limits (buffered maximum and minimum)
        if (bounds is None) and (buffer is not None):
            bounds = self.buffered_bounds(buffer)
        # read input velocity file from geotiff
        UV = pc.grid.data().from_geotif(str(self.filename),
            bands=[1,2], bounds=bounds)
        # return the input velocity field as new point collection object
        # use scale to convert from m/yr to m/s
        self.velocity = pc.grid.data().from_dict(dict(x=UV.x, y=UV.y,
            U=UV.z[:,:,0]*scale, V=UV.z[:,:,1]*scale))
        # copy projection from geotiff to output pc object
        self.velocity.projection = copy.copy(UV.projection)
        # set the spacing and dimensions of grid
        dx = self.velocity.x[1] - self.velocity.x[0]
        dy = self.velocity.y[1] - self.velocity.y[0]
        setattr(self.velocity, 'spacing', (dx, dy))
        setattr(self.velocity, 'type', 'grid')
        # attempt to set dimensions if not current attribute
        try:
            setattr(self.velocity, 'ndim', self.velocity.U.ndim)
        except AttributeError as exc:
            pass
        return self

    # PURPOSE: read netCDF4 velocity file and extract x and y velocities
    def from_nc(self,
            filename: str | io.IOBase,
            field_mapping: dict = dict(U='VX', V='VY'),
            group: str or None = None,
            bounds: list | np.ndarray | None = None,
            buffer: float | None = 5e4,
            scale: float = 1.0/31557600.0
        ):
        """
        Read netCDF4 velocity file and extract x and y velocities

        Parameters
        ----------
        filename: str
            netCDF4 velocity file
        field_mapping: dict, default {'U':'VX', 'V':'VY'}
            mapping between netCDF4 and output field variables
        group: str or NoneType, default None
            netCDF4 group to extract variables
        bounds: list or NoneType, default None
            boundaries to read: ``[[xmin, xmax], [ymin, ymax]]``

            If not specified, can read around input points
        buffer: float or NoneType, default 5e4
            Buffer around input points for extracting velocity fields
        scale: float, default 1.0/31557600.0
            scaling factor for converting velocities to m/s

            defaults to converting from m/yr
        """
        # find input netCDF4 velocity file
        self.case_insensitive_filename(filename)
        # x and y limits (buffered maximum and minimum)
        if (bounds is None) and (buffer is not None):
            bounds = self.buffered_bounds(buffer)
        # read input velocity file from netCDF4
        self.velocity = pc.grid.data().from_nc(self.filename,
            field_mapping=field_mapping, group=group,
            bounds=bounds)
        # attempt to set dimensions if not current attribute
        try:
            setattr(self.velocity, 'ndim', self.velocity.U.ndim)
        except AttributeError as exc:
            pass
        # swap orientation of axes
        if (self.velocity.t_axis == 0) and (self.velocity.ndim == 3):
            self.velocity.U = np.transpose(self.velocity.U, axes=(1,2,0))
            self.velocity.V = np.transpose(self.velocity.V, axes=(1,2,0))
            # check if velocity has error components
            if hasattr(self.velocity, 'eU'):
                self.velocity.eU = np.transpose(self.velocity.eU, axes=(1,2,0))
            if hasattr(self.velocity, 'eV'):
                self.velocity.eV = np.transpose(self.velocity.eV, axes=(1,2,0))
            # update time dimension axis
            self.velocity.t_axis = 2
        # create mask for invalid velocity points
        mask = ((self.velocity.U.data == self.velocity.fill_value) & \
            (self.velocity.V.data == self.velocity.fill_value))
        # check if any grid values are nan
        mask |= np.isnan(self.velocity.U.data) | np.isnan(self.velocity.V.data)
        # use scale to convert from m/yr to m/s
        self.velocity.U = scale*np.array(self.velocity.U, dtype=float)
        self.velocity.V = scale*np.array(self.velocity.V, dtype=float)
        if hasattr(self.velocity, 'eU'):
            self.velocity.eU = scale*np.array(self.velocity.eU, dtype=float)
        if hasattr(self.velocity, 'eV'):
            self.velocity.eV = scale*np.array(self.velocity.eV, dtype=float)
        # update fill values in velocity grids
        self.velocity.U[mask] = self.fill_value
        self.velocity.V[mask] = self.fill_value
        if hasattr(self.velocity, 'eU'):
            self.velocity.eU[mask] = self.fill_value
        if hasattr(self.velocity, 'eV'):
            self.velocity.eV[mask] = self.fill_value
        # set the spacing of grid
        dx = self.velocity.x[1] - self.velocity.x[0]
        dy = self.velocity.y[1] - self.velocity.y[0]
        setattr(self.velocity, 'spacing', (dx, dy))
        setattr(self.velocity, 'type', 'grid')
        return self

    # PURPOSE: create an advection object from an input dictionary
    def from_dict(self,
            D_dict: dict,
            bounds: list | np.ndarray | None = None,
            buffer: float | None = None,
            scale: float = 1.0/31557600.0,
            t_axis: int = 0
        ):
        """
        Create an advection object from an input dictionary

        Parameters
        ----------
        D_dict: dict
            Dictionary of advection grid variables
        bounds: list or NoneType, default None
            boundaries to read: ``[[xmin, xmax], [ymin, ymax]]``

            If not specified, can read around input points
        buffer: float or NoneType, default None
            Buffer around input points for extracting velocity fields
        scale: float, default 1.0/31557600.0
            scaling factor for converting velocities to m/s

            defaults to converting from m/yr
        """
        # read input velocity file from netCDF4
        self.velocity = pc.grid.data(t_axis=t_axis).from_dict(D_dict)
        # x and y limits (buffered maximum and minimum)
        if (bounds is None) and (buffer is not None):
            # x and y limits (buffered maximum and minimum)
            bounds = self.buffered_bounds(buffer)
        # crop grid data to bounds
        if bounds is not None:
            self.velocity.crop(bounds[0], bounds[1])
        # attempt to set dimensions if not current attribute
        try:
            setattr(self.velocity, 'ndim', self.velocity.U.ndim)
        except AttributeError as exc:
            pass
        # swap orientation of axes
        if (self.velocity.t_axis == 0) and (self.velocity.ndim == 3):
            self.velocity.U = np.transpose(self.velocity.U, axes=(1,2,0))
            self.velocity.V = np.transpose(self.velocity.V, axes=(1,2,0))
            # check if velocity has error components
            if hasattr(self.velocity, 'eU'):
                self.velocity.eU = np.transpose(self.velocity.eU, axes=(1,2,0))
            if hasattr(self.velocity, 'eV'):
                self.velocity.eV = np.transpose(self.velocity.eV, axes=(1,2,0))
            # update time dimension axis
            self.velocity.t_axis = 2
        # create mask for invalid velocity points
        mask = ((self.velocity.U.data == self.velocity.fill_value) & \
            (self.velocity.V.data == self.velocity.fill_value))
        # check if any grid values are nan
        mask |= np.isnan(self.velocity.U.data) | np.isnan(self.velocity.V.data)
        # use scale to convert from m/yr to m/s
        self.velocity.U = scale*np.array(self.velocity.U, dtype=float)
        self.velocity.V = scale*np.array(self.velocity.V, dtype=float)
        if hasattr(self.velocity, 'eU'):
            self.velocity.eU = scale*np.array(self.velocity.eU, dtype=float)
        if hasattr(self.velocity, 'eV'):
            self.velocity.eV = scale*np.array(self.velocity.eV, dtype=float)
        # update fill values in velocity grids
        self.velocity.U[mask] = self.fill_value
        self.velocity.V[mask] = self.fill_value
        if hasattr(self.velocity, 'eU'):
            self.velocity.eU[mask] = self.fill_value
        if hasattr(self.velocity, 'eV'):
            self.velocity.eV[mask] = self.fill_value
        # set the spacing of grid
        dx = self.velocity.x[1] - self.velocity.x[0]
        dy = self.velocity.y[1] - self.velocity.y[0]
        setattr(self.velocity, 'spacing', (dx, dy))
        setattr(self.velocity, 'type', 'grid')
        return self

    # PURPOSE: build a data object from a list of other data objects
    def from_list(self,
            D_list: list,
            sort: bool = False,
            bounds: list | np.ndarray | None = None,
            buffer: float | None = None,
            scale: float = 1.0/31557600.0,
        ):
        """
        Build a data object from a list of other data objects

        Parameters
        ----------
        D_list: list
            ``pointCollection`` grid objects
        sort: bool, default False
            sort the list of data objects before merging
        bounds: list or NoneType, default None
            boundaries to read: ``[[xmin, xmax], [ymin, ymax]]``

            If not specified, can read around input points
        buffer: float or NoneType, default None
            Buffer around input points for extracting velocity fields
        scale: float, default 1.0/31557600.0
            scaling factor for converting velocities to m/s

            defaults to converting from m/yr
        """
        # x and y limits (buffered maximum and minimum)
        if (bounds is None) and (buffer is not None):
            # x and y limits (buffered maximum and minimum)
            bounds = self.buffered_bounds(buffer)
        # read input velocity data from grid objects
        self.velocity = pc.grid.data().from_list(D_list, sort=sort)
        # crop grid data to bounds
        if bounds is not None:
            self.velocity.crop(bounds[0], bounds[1])
        # attempt to set dimensions if not current attribute
        try:
            setattr(self.velocity, 'ndim', self.velocity.U.ndim)
        except AttributeError as exc:
            pass
        # swap orientation of axes
        if (self.velocity.t_axis == 0) and (self.velocity.ndim == 3):
            self.velocity.U = np.transpose(self.velocity.U, axes=(1,2,0))
            self.velocity.V = np.transpose(self.velocity.V, axes=(1,2,0))
            # check if velocity has error components
            if hasattr(self.velocity, 'eU'):
                self.velocity.eU = np.transpose(self.velocity.eU, axes=(1,2,0))
            if hasattr(self.velocity, 'eV'):
                self.velocity.eV = np.transpose(self.velocity.eV, axes=(1,2,0))
            # update time dimension axis
            self.velocity.t_axis = 2
        # use scale to convert from m/yr to m/s
        self.velocity.U *= scale
        self.velocity.V *= scale
        if hasattr(self.velocity, 'eU'):
            self.velocity.eU = scale*np.array(self.velocity.eU, dtype=float)
        if hasattr(self.velocity, 'eV'):
            self.velocity.eV = scale*np.array(self.velocity.eV, dtype=float)
        # update fill values in velocity grids
        mask = np.isnan(self.velocity.U) | np.isnan(self.velocity.V)
        self.velocity.U[mask] = self.fill_value
        self.velocity.V[mask] = self.fill_value
        if hasattr(self.velocity, 'eU'):
            self.velocity.eU[mask] = self.fill_value
        if hasattr(self.velocity, 'eV'):
            self.velocity.eV[mask] = self.fill_value
        # set the spacing of grid
        dx = self.velocity.x[1] - self.velocity.x[0]
        dy = self.velocity.y[1] - self.velocity.y[0]
        setattr(self.velocity, 'spacing', (dx, dy))
        setattr(self.velocity, 'type', 'grid')
        return self

    # PURPOSE: fill in holes in velocity maps
    def fill_velocity_gaps(self):
        """
        Fill in gaps in the velocity field.

        First, check if the velocity at a point and time can be filled from the
        velocity in a previous or subsequent year, or from the average of the
        previous and subsequent years.  If not, fill the velocity with the mean
        velocity field.

        Returns
        -------
        None.

        """
        # calculate an error-weighted average of the velocities
        v=self.velocity.copy()
        wU=1/v.eU**2/np.nansum(1/v.eU**2, axis=2)[:,:,None]
        wV=1/v.eV**2/np.nansum(1/v.eV**2, axis=2)[:,:,None]
        vbar=pc.grid.data().from_dict({'x':np.array(v.x),
                               'y':np.array(v.y),
                               'U':np.nansum(wU*v.U, axis=2),
                               'V':np.nansum(wV*v.V, axis=2)})

        # attempt to fill in gaps in each velocity field with the average of
        # the velocity from one year prior and that from one year later.
        # if one or the other of these is missing, use valid values from the
        # slice that is present.
        v_filled=self.velocity.copy()

        delta_year=float(24*3600*365)
        delta_tol = 24*3600*365/8

        for ii in range(v.U.shape[2]-1):
            this_U=v.U[:,:,ii].copy()
            this_V=v.V[:,:,ii].copy()
            u_temp = np.zeros_like(this_U)
            v_temp = np.zeros_like(this_U)
            w_temp = np.zeros_like(this_U)
            for dt in [-delta_tol, delta_tol]:
                other_year = np.argmin(np.abs(v.time[ii]+delta_year-v.time))
                if np.abs((v.time[other_year]-v.time[ii]).astype(float)) > delta_tol:
                    continue
                good = np.isfinite(v.U[:,:,other_year])
                u_temp[good] += v.U[:,:,other_year][good]
                v_temp[good] += v.V[:,:,other_year][good]
                w_temp[good] += 1
            u_temp[w_temp > 0] /= w_temp[w_temp>0]
            v_temp[w_temp > 0] /= w_temp[w_temp>0]
            to_replace = ((~np.isfinite(this_U)) & (w_temp>0)).ravel()
            if np.any(to_replace):
                this_U.ravel()[to_replace] = u_temp.ravel()[to_replace]
                this_V.ravel()[to_replace] = v_temp.ravel()[to_replace]
            v_filled.U[:,:,ii]=this_U
            v_filled.V[:,:,ii]=this_V

        # fill in the remaining gaps using the mean velocity field
        for ii in range(v.U.shape[2]):
            this_U=v.U[:,:,ii].copy()
            this_V=v.V[:,:,ii].copy()
            to_replace = (~np.isfinite(this_U)).ravel()
            if np.any(to_replace):
                this_U.ravel()[to_replace] = vbar.U.ravel()[to_replace]
                this_V.ravel()[to_replace] = vbar.V.ravel()[to_replace]
            v_filled.U[:,:,ii]=this_U
            v_filled.V[:,:,ii]=this_V

        self.velocity.V = v_filled.V
        self.velocity.U = v_filled.U

    #PURPOSE: make an interpolation object to allow fast interpolation of the velocity field
    def xy0_interpolator(self, bounds=None, t_range=None, t_step=None):
        """
        Build interpolation objects from initial to final positions

        Parameters
        ----------
        bounds : iterable, optional
            bounds of the interpolation domain.  Can be specified as two iterables
            (x, y) or three (x, y, t). The default is None.
        t_range : iterable, optional
            time range for the interpolation, in velocity time units (seconds relative
                to J2000).  If not specified, the time range of self.velocity
                will be used.  The default is None.
        t_step : float, optional
            The time step for the interpolation, in seconds.  If not specified,
            the time values in the velocity object will be used. The default is None.

        Returns
        -------
        interp_dict : dict
            dictionary containing x and y interpolator objects giving the final
            location of a parcel as a function of time.  Each should be called
            with coordinates (y, x, time).
        """
        if t_range is None:
            t_range = [np.nanmin(self.velocity.time), np.nanmax(self.velocity.time)]
        if bounds is None:
            bounds=self.velocity.bounds() + [t_range]
        else:
            if len(bounds)==2:
                bounds += [t_range]
        if t_step is None:
            # use the times on the velocity object
            ti = self.velocity.t
        else:
            ti = np.arange(bounds[2][0], bounds[2][1]+t_step, t_step)
        dx=self.velocity.x[1]-self.velocity.x[0]

        x0, y0 = [np.arange(bb[0]-dx, bb[1]+dx, dx) for bb in bounds]
        yg, xg, tg = np.meshgrid(x0, y0, ti, indexing='ij')
        shp=xg.shape
        self.x, self.y, self.t = [item.ravel() for item in (xg, yg, tg)]
        self.translate_parcel()

        interp_dict = {'x':scipy.interpolate.RegularGridInterpolator((y0, x0, ti), self.x0.reshape(shp), bounds_error=False),
              'y':scipy.interpolate.RegularGridInterpolator((y0, x0, ti), self.y0.reshape(shp), bounds_error=False)}

        return interp_dict

    # PURPOSE: translate a parcel between two times using an advection function
    def translate_parcel(self, **kwargs):
        """
        Translates a parcel between two times using an advection function

        Parameters
        ----------
        integrator: str
            Advection function

                - ``'euler'``
                - ``'RK4'``
                - ``'RKF45'``
        method: str
            Interpolation method for velocities

                - ``'bilinear'``: quick bilinear interpolation
                - ``'spline'``: scipy bivariate spline interpolation
                - ``'linear'``, ``'nearest'``: scipy regular grid interpolations
                - ``'linearND'``, ``'nearestND'``: scipy unstructured N-dimensional interpolations
        step: int or float, default 1
            Temporal step size in days
        N: int or NoneType, default None
            Number of integration steps

            Default is determined based on the temporal step size
        t0: float, default 0.0
            Ending time for advection
        """
        # set default keyword arguments
        kwargs.setdefault('integrator', self.integrator)
        kwargs.setdefault('method', self.method)
        kwargs.setdefault('step', 1)
        kwargs.setdefault('N', None)
        kwargs.setdefault('t0', self.t0)
        kwargs.setdefault('streak', False)
        # check that there are points within the velocity file
        if not self.inside_polygon(self.x,self.y).any():
            raise ValueError('No points within ice velocity image')
        # update advection class attributes
        if (kwargs['integrator'] != self.integrator):
            self.integrator = copy.copy(kwargs['integrator'])
        if (kwargs['method'] != self.method):
            self.method = copy.copy(kwargs['method'])
        if (kwargs['t0'] != self.t0):
            self.t0 = np.copy(kwargs['t0'])
        # advect the parcel every step days
        # (using closest number of iterations)
        seconds = (kwargs['step']*86400.0)
        # set or calculate the number of steps to advect the dataset
        if kwargs['N'] is not None:
            n_steps = np.copy(kwargs['N'])
        elif (np.min(self.t0) < np.min(self.t)):
            # maximum number of steps to advect backwards in time
            n_steps = np.abs(np.max(self.t) - np.min(self.t0))/seconds
        elif (np.max(self.t0) > np.max(self.t)):
            # maximum number of steps to advect forward in time
            n_steps = np.abs(np.max(self.t0) - np.min(self.t))/seconds
        elif (np.ndim(self.t0) == 0) or (np.ndim(self.t) == 0):
            # maximum number of steps between the two datasets
            n_steps = np.max(np.abs(self.t0 - self.t))/seconds
        else:
            # average number of steps between the two datasets
            n_steps = np.abs(np.mean(self.t0) - np.mean(self.t))/seconds
        # check input advection functions
        kwargs.update({'N':np.int64(n_steps)})
        if (self.integrator == 'euler'):
            # euler: Explicit Euler method
            return self.euler(**kwargs)
        elif (self.integrator == 'RK4'):
            # RK4: Fourth-order Runge-Kutta method
            return self.RK4(**kwargs)
        elif (self.integrator == 'RKF45'):
            # RKF45: adaptive Runge-Kutta-Fehlberg 4(5) method
            return self.RKF45(**kwargs)
        else:
            raise ValueError('Invalid advection function')

    # PURPOSE: Advects parcels using an Explicit Euler integration
    def euler(self, **kwargs):
        """
        Advects parcels using an Explicit Euler integration

        Parameters
        ----------
        N: int, default 1
            Number of integration steps
        """
        # set default keyword options
        kwargs.setdefault('N', 1)
        # translate parcel from time 1 to time 2 at time step
        dt = (self.t0 - self.t)/np.float64(kwargs['N'])
        self.x0 = np.copy(self.x)
        self.y0 = np.copy(self.y)
        # keep track of time for 3-dimensional interpolations
        t = np.copy(self.t)
        self.__update_streak__(t, **kwargs)
        for i in range(kwargs['N']):
            u1, v1 = self.interpolate(x=self.x0, y=self.y0, t=t)
            self.x0 += u1*dt
            self.y0 += v1*dt
            # add to time
            t += dt
            self.__update_streak__(t, **kwargs)

        # return the translated coordinates
        return self

    # PURPOSE: Advects parcels using a fourth-order Runge-Kutta integration
    def RK4(self, **kwargs):
        """
        Advects parcels using a fourth-order Runge-Kutta integration

        Parameters
        ----------
        N: int, default 1
            Number of integration steps
        """
        # set default keyword options
        kwargs.setdefault('N', 1)
        # translate parcel from time 1 to time 2 at time step
        dt = (self.t0 - self.t)/np.float64(kwargs['N'])
        self.x0 = np.copy(self.x)
        self.y0 = np.copy(self.y)
        self.__update_streak__(self.t, **kwargs)
        # keep track of time for 3-dimensional interpolations
        t = np.copy(self.t)
        for i in range(kwargs['N']):
            u1, v1 = self.interpolate(x=self.x0, y=self.y0, t=t)
            x2, y2 = (self.x0 + 0.5*u1*dt, self.y0 + 0.5*v1*dt)
            u2, v2 = self.interpolate(x=x2, y=y2, t=t)
            x3, y3 = (self.x0 + 0.5*u2*dt, self.y0 + 0.5*v2*dt)
            u3, v3 = self.interpolate(x=x3, y=y3, t=t)
            x4, y4 = (self.x0 + u3*dt, self.y0 + v3*dt)
            u4, v4 = self.interpolate(x=x4, y=y4, t=t)
            self.x0 += dt*(u1 + 2.0*u2 + 2.0*u3 + u4)/6.0
            self.y0 += dt*(v1 + 2.0*v2 + 2.0*v3 + v4)/6.0
            # add to time
            t += dt
            self.__update_streak__(t, **kwargs)

        # return the translated coordinates
        return self

    # PURPOSE: Advects parcels using a Runge-Kutta-Fehlberg integration
    def RKF45(self, **kwargs):
        """
        Advects parcels using a Runge-Kutta-Fehlberg 4(5) integration

        Parameters
        ----------
        N: int, default 1
            Number of integration steps
        """
        # set default keyword options
        kwargs.setdefault('N', 1)
        # coefficients in Butcher tableau for Runge-Kutta-Fehlberg 4(5) method
        b4 = [25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4104.0, -1.0/5.0, 0.0]
        b5 = [16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0]
        # using an adaptive step size:
        # iterate solution until the difference is less than the tolerance
        # difference between the 4th and 5th order solutions
        sigma = np.inf
        # tolerance for solutions
        tolerance = 5e-2
        # multiply scale by factors of 2 until iteration reaches tolerance level
        scale = 1
        self.x0 = np.copy(self.x)
        self.y0 = np.copy(self.y)
        self.__update_streak__(self.t, **kwargs)

        # while the difference (sigma) is greater than the tolerance
        while (sigma > tolerance) or np.isnan(sigma):
            # translate parcel from time 1 to time 2 at time step
            dt = (self.t0 - self.t)/np.float64(scale*kwargs['N'])
            X4OA, Y4OA = (np.copy(self.x), np.copy(self.y))
            X5OA, Y5OA = (np.copy(self.x), np.copy(self.y))
            # keep track of time for 3-dimensional interpolations
            t = np.copy(self.t)
            for i in range(scale*kwargs['N']):
                # calculate fourth order accurate solutions
                u4, v4 = self.RFK45_velocities(X4OA, Y4OA, dt, t=t)
                X4OA += dt*np.dot(b4, u4)
                Y4OA += dt*np.dot(b4, v4)
                # calculate fifth order accurate solutions
                u5, v5 = self.RFK45_velocities(X5OA, Y5OA, dt, t=t)
                X5OA += dt*np.dot(b5, u5)
                Y5OA += dt*np.dot(b5, v5)
                # add to time
                t += dt
            # calculate difference between 4th and 5th order accurate solutions
            i, = np.nonzero(np.isfinite(X4OA) & np.isfinite(Y4OA))
            num = np.count_nonzero(np.isfinite(X4OA) & np.isfinite(Y4OA))
            sigma = np.sqrt(np.sum((X5OA[i]-X4OA[i])**2 + (Y5OA[i]-Y4OA[i])**2)/num)
            # if sigma is less than the tolerance: save xi and yi coordinates
            # else: multiply scale by factors of 2 and re-run iteration
            if (sigma <= tolerance) or np.isnan(sigma):
                self.x0,self.y0 = (np.copy(X4OA), np.copy(Y4OA))
                self.__update_streak__(t, **kwargs)
            else:
                scale *= 2
        # return the translated coordinates
        return self

    # PURPOSE: calculates X and Y velocities for Runge-Kutta-Fehlberg 4(5) method
    def RFK45_velocities(self,
            xi: np.ndarray,
            yi: np.ndarray,
            dt: np.ndarray,
            **kwargs
        ):
        """
        Calculates X and Y velocities for Runge-Kutta-Fehlberg 4(5) method

        Parameters
        ----------
        xi: np.ndarray
            x-coordinates
        yi: np.ndarray
            y-coordinates
        dt: np.ndarray
            integration time step size
        t: np.ndarray or NoneType, default None
            time coordinates
        """
        kwargs.setdefault('t', None)
        # Butcher tableau for Runge-Kutta-Fehlberg 4(5) method
        A = np.array([[1.0/4.0, 0.0, 0.0, 0.0, 0.0],
            [3.0/32.0, 9.0/32.0, 0.0, 0.0, 0.0],
            [1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0, 0.0, 0.0],
            [439.0/216.0, -8.0, 3680.0/513.0, -845.0/4104.0, 0.0],
            [-8.0/27.0, 2.0, -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0]])
        # calculate velocities and parameters for iteration
        u1, v1 = self.interpolate(x=xi, y=yi, t=kwargs['t'])
        x2 = xi + A[0,0]*u1*dt
        y2 = yi + A[0,0]*v1*dt
        u2, v2 = self.interpolate(x=x2, y=y2, t=kwargs['t'])
        x3 = xi + (A[1,0]*u1 + A[1,1]*u2)*dt
        y3 = yi + (A[1,0]*v1 + A[1,1]*v2)*dt
        u3, v3 = self.interpolate(x=x3, y=y3, t=kwargs['t'])
        x4 = xi + (A[2,0]*u1 + A[2,1]*u2 + A[2,2]*u3)*dt
        y4 = yi + (A[2,0]*v1 + A[2,1]*v2 + A[2,2]*v3)*dt
        u4, v4 = self.interpolate(x=x4, y=y4, t=kwargs['t'])
        x5 = xi + (A[3,0]*u1 + A[3,1]*u2 + A[3,2]*u3 + A[3,3]*u4)*dt
        y5 = yi + (A[3,0]*v1 + A[3,1]*v2 + A[3,2]*v3 + A[3,3]*v4)*dt
        u5, v5 = self.interpolate(x=x5, y=y5, t=kwargs['t'])
        x6 = xi + (A[4,0]*u1 + A[4,1]*u2 + A[4,2]*u3 + A[4,3]*u4 + A[4,4]*u5)*dt
        y6 = yi + (A[4,0]*v1 + A[4,1]*v2 + A[4,2]*v3 + A[4,3]*v4 + A[4,4]*v5)*dt
        u6, v6 = self.interpolate(x=x6, y=y6, t=kwargs['t'])
        return (np.array([u1,u2,u3,u4,u5,u6]), np.array([v1,v2,v3,v4,v5,v6]))

    def buffered_bounds(self, buffer=5e4):
        """
        Calculates the bounding box including a buffer distance
        """
        xmin = np.floor(self.x.min()) - buffer
        xmax = np.ceil(self.x.max()) + buffer
        ymin = np.floor(self.y.min()) - buffer
        ymax = np.ceil(self.y.max()) + buffer
        return [[xmin,xmax], [ymin,ymax]]

    @property
    def distance(self):
        """
        Calculates displacement between the start and end coordinates

        Returns
        -------
        dist: np.ndarray
            Eulerian distance between start and end points
        """
        try:
            dist = np.sqrt((self.x0 - self.x)**2 + (self.y0 - self.y)**2)
        except Exception as exc:
            return None
        else:
            return dist

    # PURPOSE: check a specified 2D point is inside a specified 2D polygon
    def inside_polygon(self,
            x: np.ndarray,
            y: np.ndarray,
            threshold: float = 0.01
        ):
        """
        Indicates whether a specified 2D point is inside a specified 2D polygon

        Parameters
        ----------
        x: np.ndarray
            x-coordinates to query
        y: np.ndarray
            y-coordinates to query
        threshold: float, default 0.01
            Minimum angle for checking if inside polygon

        Returns
        -------
        flag: bool
            Flag specifying if point is within polygon

                - ``True`` for points within polygon,
                - ``False`` for points outside polygon
        """
        # create numpy arrays for 2D points
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        nn = len(x)
        # get points for polygon
        if (self.velocity.type == 'grid'):
            # create polygon with the extents of the image
            xmin,xmax,ymin,ymax = self.velocity.extent
            xpts = np.array([xmin, xmax, xmax, xmin, xmin])
            ypts = np.array([ymin, ymin, ymax, ymax, ymin])
        elif (self.velocity.type == 'mesh'):
            # create polygon with convex hull of points
            points = np.c_[self.velocity.x, self.velocity.y]
            hull = scipy.spatial.ConvexHull(points)
            xpts = points[hull.vertices, 0]
            ypts = points[hull.vertices, 1]
        # check dimensions of polygon points
        if (xpts.ndim != 1):
            raise ValueError('X coordinates of polygon not a vector.')
        if (ypts.ndim != 1):
            raise ValueError('Y coordinates of polygon not a vector.')
        if (len(xpts) != len(ypts)):
            raise ValueError('Incompatable vector dimensions.')
        # maximum possible number of vertices in polygon
        N = len(xpts)
        # Close the polygon if not already closed
        if not np.isclose(xpts[-1],xpts[0]) and not np.isclose(ypts[-1],ypts[0]):
            xpts = np.append(xpts,xpts[0])
            ypts = np.append(ypts,ypts[0])
        else:
            # remove 1 from number of vertices
            N -= 1
        # Calculate dot and cross products of points to neighboring polygon points
        i = np.arange(N)
        X1 = np.dot(xpts[i][:,np.newaxis],np.ones((1,nn))) - \
            np.dot(np.ones((N,1)),x[np.newaxis,:])
        Y1 = np.dot(ypts[i][:,np.newaxis],np.ones((1,nn))) - \
            np.dot(np.ones((N,1)),y[np.newaxis,:])
        X2 = np.dot(xpts[i+1][:,np.newaxis],np.ones((1,nn))) - \
            np.dot(np.ones((N,1)),x[np.newaxis,:])
        Y2 = np.dot(ypts[i+1][:,np.newaxis],np.ones((1,nn))) - \
            np.dot(np.ones((N,1)),y[np.newaxis,:])
        # Dot-product
        dp = X1*X2 + Y1*Y2
        # Cross-product
        cp = X1*Y2 - Y1*X2
        # Calculate tangent of the angle between the two nearest adjacent points
        theta = np.arctan2(cp,dp)
        # If point is outside polygon then summation over all possible
        # angles will equal a small number (e.g. 0.01)
        flag = np.where(np.abs(np.sum(theta,axis=0)) > threshold, True, False)
        # Make a scalar value if there was only one input value
        if (nn == 1):
            return flag[0]
        else:
            return flag

    # PURPOSE: wrapper function for calling interpolation functions
    def interpolate(self, **kwargs):
        """
        Wrapper function for calling interpolation functions

        Returns
        -------
        U: np.ndarray
            Velocity in x-direction
        V: np.ndarray
            Velocity in y-direction
        """
        if (self.method == 'bilinear'):
            return self.bilinear_interpolation(**kwargs)
        elif (self.method == 'spline'):
            return self.spline_interpolation(**kwargs)
        elif (self.method in ('nearest','linear')):
            return self.regular_grid_interpolation(**kwargs)
        elif (self.method in ('nearestND','linearND')):
            return self.unstructured_interpolation(**kwargs)
        else:
            raise ValueError('Invalid interpolation function')

    # PURPOSE: use bilinear interpolation of velocities to coordinates
    def bilinear_interpolation(self, **kwargs):
        """
        Bilinearly interpolate U and V velocities to coordinates

        Parameters
        ----------
        x: np.ndarray or NoneType, default None
            x-coordinates
        y: np.ndarray or NoneType, default None
            y-coordinates
        fill_value: float, default np.nan
            Invalid value

        Returns
        -------
        U: np.ndarray
            Velocity in x-direction
        V: np.ndarray
            Velocity in y-direction
        """
        # set default keyword options
        kwargs.setdefault('x', None)
        kwargs.setdefault('y', None)
        kwargs.setdefault('fill_value', self.fill_value)
        # output bilinear interpolated data
        U = np.full_like(kwargs['x'], np.nan)
        V = np.full_like(kwargs['x'], np.nan)
        # only run interpolation if coordinates are finite
        # and within the extents of the input velocity map
        v, = np.nonzero(np.isfinite(kwargs['x']) & np.isfinite(kwargs['y']) &
            self.inside_polygon(kwargs['x'],kwargs['y']))
        # check that there are indice values (0 is falsy)
        if not np.any(v) and not np.any(v == 0):
            return (U, V)
        # calculating indices for original grid
        xmin,xmax,ymin,ymax = self.velocity.extent
        ii = np.floor((kwargs['x'][v]-xmin)/self.velocity.spacing[0]).astype('i')
        jj = np.floor((kwargs['y'][v]-ymin)/self.velocity.spacing[1]).astype('i')
        # weight arrays
        Wa = ((kwargs['x'][v]-self.velocity.x[ii])*(kwargs['y'][v]-self.velocity.y[jj]))
        Wb = ((self.velocity.x[ii+1]-kwargs['x'][v])*(kwargs['y'][v]-self.velocity.y[jj]))
        Wc = ((kwargs['x'][v]-self.velocity.x[ii])*(self.velocity.y[jj+1]-kwargs['y'][v]))
        Wd = ((self.velocity.x[ii+1]-kwargs['x'][v])*(self.velocity.y[jj+1]-kwargs['y'][v]))
        W = ((self.velocity.x[ii+1]-self.velocity.x[ii])*(self.velocity.y[jj+1]-self.velocity.y[jj]))
        # data at indices
        Ua,Va = (self.velocity.U[jj,ii], self.velocity.V[jj,ii])
        Ub,Vb = (self.velocity.U[jj+1,ii], self.velocity.V[jj+1,ii])
        Uc,Vc = (self.velocity.U[jj,ii+1], self.velocity.V[jj,ii+1])
        Ud,Vd = (self.velocity.U[jj+1,ii+1],self.velocity.V[jj+1,ii+1])
        # calculate bilinear interpolated data
        U[v] = (Ua*Wa + Ub*Wb + Uc*Wc + Ud*Wd)/W
        V[v] = (Va*Wa + Vb*Wb + Vc*Wc + Vd*Wd)/W
        # replace invalid values with fill value
        U = np.nan_to_num(U, nan=kwargs['fill_value'])
        V = np.nan_to_num(V, nan=kwargs['fill_value'])
        return (U, V)

    # PURPOSE: use biharmonic splines to interpolate velocities to coordinates
    def spline_interpolation(self, **kwargs):
        """
        Interpolate U and V velocities to coordinates using biharmonic splines

        Parameters
        ----------
        x: np.ndarray or NoneType, default None
            x-coordinates
        y: np.ndarray or NoneType, default None
            y-coordinates
        kx: int, default 1
            degrees of the bivariate spline in x-direction
        ky: int, default 1
            degrees of the bivariate spline in y-direction
        fill_value: float, default np.nan
            Invalid value

        Returns
        -------
        U: np.ndarray
            Velocity in x-direction
        V: np.ndarray
            Velocity in y-direction
        """
        # set default keyword options
        kwargs.setdefault('x', None)
        kwargs.setdefault('y', None)
        kwargs.setdefault('kx', 1)
        kwargs.setdefault('ky', 1)
        kwargs.setdefault('fill_value', self.fill_value)
        # output interpolated data
        U = np.full_like(kwargs['x'], np.nan)
        V = np.full_like(kwargs['x'], np.nan)
        # only run interpolation if coordinates are finite
        # and within the extents of the input velocity map
        v, = np.nonzero(np.isfinite(kwargs['x']) & np.isfinite(kwargs['y']) &
            self.inside_polygon(kwargs['x'],kwargs['y']))
        # check that there are indice values (0 is falsy)
        if not np.any(v) and not np.any(v == 0):
            return (U, V)
        # create mask for invalid values
        indy,indx = np.nonzero((self.velocity.U == self.fill_value) |
            (self.velocity.V == self.fill_value) |
            (np.isnan(self.velocity.U)) | (np.isnan(self.velocity.V)))
        self.velocity.U[indy,indx] = 0.0
        self.velocity.V[indy,indx] = 0.0
        mask = np.zeros((self.velocity.shape))
        mask[indy,indx] = 1.0
        # build spline interpolants for input grid
        if not self.interpolant:
            # use scipy bivariate splines to interpolate values
            self.interpolant['U'] = scipy.interpolate.RectBivariateSpline(
                self.velocity.x, self.velocity.y, self.velocity.U.T,
                kx=kwargs['kx'], ky=kwargs['ky'])
            self.interpolant['V'] = scipy.interpolate.RectBivariateSpline(
                self.velocity.x, self.velocity.y, self.velocity.V.T,
                kx=kwargs['kx'], ky=kwargs['ky'])
            self.interpolant['mask'] = scipy.interpolate.RectBivariateSpline(
                self.velocity.x, self.velocity.y, mask.T,
                kx=kwargs['kx'], ky=kwargs['ky'])
        # create mask for invalid values
        invalid = np.ceil(self.interpolant['mask'].ev(
            kwargs['x'][v],kwargs['y'][v])).astype(bool)
        masked_values = np.where(invalid, np.nan, 0.0)
        # calculate interpolated data
        U[v] = self.interpolant['U'].ev(
            kwargs['x'][v],kwargs['y'][v]) + masked_values
        V[v] = self.interpolant['V'].ev(
            kwargs['x'][v],kwargs['y'][v]) + masked_values
        # replace invalid values with fill value
        U = np.nan_to_num(U, nan=kwargs['fill_value'])
        V = np.nan_to_num(V, nan=kwargs['fill_value'])
        return (U, V)

    # PURPOSE: use regular grid interpolation of velocities to coordinates
    def regular_grid_interpolation(self, **kwargs):
        """
        Interpolate U and V velocities to coordinates using
            regular grid interpolation
        Can use time-variable ``U`` and ``V`` velocities

        Parameters
        ----------
        x: np.ndarray or NoneType, default None
            x-coordinates
        y: np.ndarray or NoneType, default None
            y-coordinates
        t: np.ndarray or NoneType, default None
            time coordinates
        method: str
            Method of regular grid interpolation

                - ``'nearest'``
                - ``'linear'``
        fill_value: float, default np.nan
            Invalid value

        Returns
        -------
        U: np.ndarray
            Velocity in x-direction
        V: np.ndarray
            Velocity in y-direction
        """
        # set default keyword options
        kwargs.setdefault('x', None)
        kwargs.setdefault('y', None)
        kwargs.setdefault('t', None)
        kwargs.setdefault('method', self.method)
        kwargs.setdefault('fill_value', self.fill_value)
        # output interpolated data
        U = np.full_like(kwargs['x'], np.nan)
        V = np.full_like(kwargs['x'], np.nan)
        # only run interpolation if coordinates are finite
        # and within the extents of the input velocity map
        v, = np.nonzero(np.isfinite(kwargs['x']) & np.isfinite(kwargs['y']) &
            self.inside_polygon(kwargs['x'],kwargs['y']))
        # check that there are indice values (0 is falsy)
        if not np.any(v) and not np.any(v == 0):
            return (U, V)
        # build regular grid interpolants for input grid
        if not self.interpolant and (self.velocity.ndim == 3):
            # use scipy regular grid to interpolate values for a given method
            # will extrapolate velocities forward in time if outside range
            self.interpolant['U'] = scipy.interpolate.RegularGridInterpolator(
                (self.velocity.y, self.velocity.x, self.velocity.time),
                self.velocity.U, method=kwargs['method'], bounds_error=False,
                fill_value=None)
            self.interpolant['V'] = scipy.interpolate.RegularGridInterpolator(
                (self.velocity.y, self.velocity.x, self.velocity.time),
                self.velocity.V, method=kwargs['method'], bounds_error=False,
                fill_value=None)
        elif not self.interpolant and (self.velocity.ndim == 2):
            # use scipy regular grid to interpolate values for a given method
            self.interpolant['U'] = scipy.interpolate.RegularGridInterpolator(
                (self.velocity.y, self.velocity.x), self.velocity.U,
                method=kwargs['method'], bounds_error=False, fill_value=None)
            self.interpolant['V'] = scipy.interpolate.RegularGridInterpolator(
                (self.velocity.y, self.velocity.x), self.velocity.V,
                method=kwargs['method'], bounds_error=False, fill_value=None)
        # calculate interpolated data
        if (self.velocity.ndim == 3):
            U[v] = self.interpolant['U'].__call__(
                np.c_[kwargs['y'][v],kwargs['x'][v],kwargs['t'][v]])
            V[v] = self.interpolant['V'].__call__(
                np.c_[kwargs['y'][v],kwargs['x'][v],kwargs['t'][v]])
        elif (self.velocity.ndim == 2):
            U[v] = self.interpolant['U'].__call__(
                np.c_[kwargs['y'][v],kwargs['x'][v]])
            V[v] = self.interpolant['V'].__call__(
                np.c_[kwargs['y'][v],kwargs['x'][v]])
        # replace invalid values with fill value
        U = np.nan_to_num(U, nan=kwargs['fill_value'])
        V = np.nan_to_num(V, nan=kwargs['fill_value'])
        return (U, V)

    # PURPOSE: use unstructured interpolation of velocities to coordinates
    def unstructured_interpolation(self, **kwargs):
        """
        Interpolate unstructured U and V velocities to coordinates using
            N-dimensional interpolation functions

        Parameters
        ----------
        x: np.ndarray or NoneType, default None
            x-coordinates
        y: np.ndarray or NoneType, default None
            y-coordinates
        t: np.ndarray or NoneType, default None
            time coordinates
        method: str
            Method of unstructured interpolation

                - ``'nearestND'``
                - ``'linearND'``
        rescale: bool, default False
            Rescale points to unit cube before performing interpolation
        tree_options: dict or NoneType, default None
            Options passed to underlying KDTree for ``nearestND``
        fill_value: float, default np.nan
            Invalid value

        Returns
        -------
        U: np.ndarray
            Velocity in x-direction
        V: np.ndarray
            Velocity in y-direction
        """
        # set default keyword options
        kwargs.setdefault('x', None)
        kwargs.setdefault('y', None)
        kwargs.setdefault('t', None)
        kwargs.setdefault('method', None)
        kwargs.setdefault('rescale', False)
        kwargs.setdefault('tree_options', None)
        kwargs.setdefault('fill_value', self.fill_value)
        # output interpolated data
        U = np.full_like(kwargs['x'], np.nan)
        V = np.full_like(kwargs['x'], np.nan)
        # only run interpolation if coordinates are finite
        # and within the extents of the input velocity mesh
        valid = np.ones_like(kwargs['x'], dtype=bool)
        valid &= np.isfinite(kwargs['x'])
        valid &= np.isfinite(kwargs['y'])
        valid &= self.inside_polygon(kwargs['x'], kwargs['y'])
        # check that there are indice values
        if not np.any(valid):
            return (U, V)
        # build delaunay triangulations for input mesh coordinates
        if not hasattr(self.velocity, 'mesh'):
            # attempt to build delaunay triangulation
            _, self.velocity.mesh = self.find_valid_triangulation(
                self.velocity.x, self.velocity.y)
        # reduce to points within the convex hull of the triangulation
        valid &= self.inside_simplex(kwargs['x'], kwargs['y'])
        v, = np.nonzero(valid)
        # interpolator and keyword arguments for selected method
        if (kwargs['method'] == 'nearestND'):
            Interpolator = scipy.interpolate.NearestNDInterpolator
            kwds = dict(rescale=kwargs['rescale'],
                tree_options=kwargs['tree_options'])
        elif (kwargs['method'] == 'linearND'):
            Interpolator = scipy.interpolate.LinearNDInterpolator
            kwds = dict(rescale=kwargs['rescale'], fill_value=np.nan)
        else:
            raise ValueError('Invalid interpolation function')
        # build interpolants for input velocity meshes
        if not self.interpolant:
            self.interpolant['U'] = Interpolator(
                self.velocity.mesh, self.velocity.U, **kwds)
            self.interpolant['V'] = Interpolator(
                self.velocity.mesh, self.velocity.V, **kwds)
        if not self.interpolant and (self.velocity.ndim > 1):
            # build interpolants for time-variable velocities
            nt = len(self.velocity.time)
            # create 1-d interpolant through time
            self.interpolant['time'] = scipy.interpolate.interp1d(
                self.velocity.time, np.arange(nt), kind='linear')
        # evaluate at points
        if (self.velocity.ndim == 1):
            # evalulate using invariant velocities
            U[v] = self.interpolant['U'].__call__(
                np.c_[kwargs['x'][v], kwargs['y'][v]])
            V[v] = self.interpolant['V'].__call__(
                np.c_[kwargs['x'][v], kwargs['y'][v]])
        else:
            # evalulate using time-variable velocities
            UT = self.interpolant['U'].__call__(
                np.c_[kwargs['x'][v], kwargs['y'][v]])
            VT = self.interpolant['V'].__call__(
                np.c_[kwargs['x'][v], kwargs['y'][v]])
            # linearly interpolate in time
            times = np.ones_like(kwargs['x'])*kwargs['t']
            indices = self.interpolant['time'].__call__(times[v]).astype(int)
            # clip indices to valid range for temporal interpolation
            indices = np.clip(indices, 0, nt-1)
            for tstep in np.unique(indices):
                # find valid points
                vi, = np.nonzero((indices == tstep) & valid)
                # check that there are indice values (0 is falsy)
                if not np.any(vi) and not np.any(vi == 0):
                    continue
                # calculate weights for linearly interpolating in time
                weight = (times[vi] - self.velocity.time[tstep]) / \
                    (self.velocity.time[tstep+1] - self.velocity.time[tstep])
                # linearly interpolate in time
                U[vi] = (1.0 - weight)*UT[vi,tstep] + weight*UT[vi,tstep+1]
                V[vi] = (1.0 - weight)*VT[vi,tstep] + weight*VT[vi,tstep+1]
        # replace invalid values with fill value
        U = np.nan_to_num(U, nan=kwargs['fill_value'])
        V = np.nan_to_num(V, nan=kwargs['fill_value'])
        return (U, V)

    # PURPOSE: find a valid Delaunay triangulation for coordinates x0 and y0
    # http://www.qhull.org/html/qhull.htm#options
    # Attempt 1: standard qhull options Qt Qbb Qc Qz
    # Attempt 2: rescale and center the inputs with option QbB
    # Attempt 3: joggle the inputs to find a triangulation with option QJ
    # if no passing triangulations: exit with empty list
    def find_valid_triangulation(self,
            x: np.ndarray,
            y: np.ndarray
        ):
        """
        Attempt to find a valid Delaunay triangulation for coordinates

        - Attempt 1: ``Qt Qbb Qc Qz``
        - Attempt 2: ``Qt Qc QbB``
        - Attempt 3: ``QJ QbB``

        Parameters
        ----------
        x: np.ndarray
            x-coordinates for mesh
        y: np.ndarray
            y-coordinates for mesh
        """
        # Attempt 1: try with standard options Qt Qbb Qc Qz
        # Qt: triangulated output, all facets will be simplicial
        # Qbb: scale last coordinate to [0,m] for Delaunay triangulations
        # Qc: keep coplanar points with nearest facet
        # Qz: add point-at-infinity to Delaunay triangulation

        # Attempt 2 in case of qhull error from Attempt 1 try Qt Qc QbB
        # Qt: triangulated output, all facets will be simplicial
        # Qc: keep coplanar points with nearest facet
        # QbB: scale input to unit cube centered at the origin

        # Attempt 3 in case of qhull error from Attempt 2 try QJ QbB
        # QJ: joggle input instead of merging facets
        # QbB: scale input to unit cube centered at the origin

        # try each set of qhull_options
        for i,qhull_option in enumerate(['Qt Qbb Qc Qz','Qt Qc QbB','QJ QbB']):
            try:
                triangle = scipy.spatial.Delaunay(np.c_[x, y],
                    qhull_options=qhull_option)
            except scipy.spatial.qhull.QhullError:
                pass
            else:
                return (i+1, triangle)
        # raise exception if still finding errors
        raise scipy.spatial.qhull.QhullError

    # PURPOSE: check a specified 2D point is inside the convex hull of a mesh
    def inside_simplex(self,
            x: np.ndarray,
            y: np.ndarray
        ):
        """
        Indicates whether a specified 2D point is inside the convex hull of a mesh

        Parameters
        ----------
        x: np.ndarray
            x-coordinates to query
        y: np.ndarray
            y-coordinates to query

        Returns
        -------
        flag: bool
            Flag specifying if point is within convex hull

                - ``True`` for points within convex hull
                - ``False`` for points outside convex hull
        """
        # only run if velocity mesh has a find simplex attribute
        if hasattr(self.velocity.mesh, 'find_simplex'):
            return (self.velocity.mesh.find_simplex(np.c_[x, y]) >= 0)
        else:
            raise AttributeError("Convex hull cannot be found for mesh")

    # PURPOSE: calculates the maximum angle within a triangle given the
    # coordinates of the triangles vertices A(x,y), B(x,y), C(x,y)
    def triangle_maximum_angle(self,
            Ax: np.ndarray,
            Ay: np.ndarray,
            Bx: np.ndarray,
            By: np.ndarray,
            Cx: np.ndarray,
            Cy: np.ndarray
        ):
        """
        Calculates the maximum angles within triangles with
        vertices A, B and C

        Parameters
        ----------
        Ax: np.ndarray
            x-coordinates of A vertices
        Ay: np.ndarray
            y-coordinates of A vertices
        Bx: np.ndarray
            x-coordinates of B vertices
        By: np.ndarray
            y-coordinates of B vertices
        Cx: np.ndarray
            x-coordinates of C vertices
        Cy: np.ndarray
            y-coordinates of C vertices
        """
        # calculate sides of triangle (opposite interior angle at vertex)
        a = np.atleast_1d(np.sqrt((Cx - Bx)**2 + (Cy - By)**2))
        b = np.atleast_1d(np.sqrt((Cx - Ax)**2 + (Cy - Ay)**2))
        c = np.atleast_1d(np.sqrt((Ax - Bx)**2 + (Ay - By)**2))
        # calculate interior angles and convert to degrees
        alpha = np.arccos((b**2 + c**2 - a**2)/(2.0*b*c))*180.0/np.pi
        beta = np.arccos((a**2 + c**2 - b**2)/(2.0*a*c))*180.0/np.pi
        gamma = np.arccos((a**2 + b**2 - c**2)/(2.0*a*b))*180.0/np.pi
        # return the largest angle within the triangle
        return np.max(np.c_[alpha, beta, gamma], axis=1)

    def imshow(self, band=None, ax=None, imtype='speed', xy_scale=1.0, **kwargs):
        """
        Create plot of velocity magnitude or divergence

        Parameters
        ----------
        band: int or NoneType, default None
            band of velocity grid to show
        ax: obj or NoneType, default None
            matplotlib figure axis
        imtype: str, default 'speed'
            image type to plot


                - ``'speed'``: velocity magnitude
                - ``'divergence'``: flow divergence
        xy_scale: float, default 1.0
            Scaling factor for converting horizontal coordinates
        **kwargs: dict
            Keyword arguments for ``plt.imshow``

        Returns
        -------
        im: obj
            matplotlib ``AxesImage`` object
        """
        kwargs['extent'] = np.array(self.velocity.extent)*xy_scale
        kwargs['origin'] = 'lower'
        if ax is None:
            ax = plt.gca()
        if band is None:
            U = getattr(self.velocity, 'U')
            V = getattr(self.velocity, 'V')
        elif (band is not None):
            U = getattr(self.velocity, 'U')[:,:,band]
            V = getattr(self.velocity, 'V')[:,:,band]
        # calculate ice speed or flow divergence
        if (imtype == 'speed'):
            # calculate speed
            zz = np.sqrt(U**2 + V**2)
        elif (imtype == 'divergence'):
            # calculate divergence
            dU = np.gradient(U, self.velocity.x, axis=1)
            dV = np.gradient(V, self.velocity.y, axis=0)
            zz = dU + dV
        # create image plot of velocity magnitude or divergence
        im = ax.imshow(zz, **kwargs)
        # return the image
        return im

    def triplot(self, ax=None, **kwargs):
        """
        Create plot of unstructured triangular mesh

        Parameters
        ----------
        ax: obj or NoneType, default None
            matplotlib figure axis
        **kwargs: dict
            Keyword arguments for ``plt.triplot``

        Returns
        -------
        im: obj
            matplotlib ``AxesImage`` object
        """
        if ax is None:
            ax = plt.gca()
        # get delaunay triangulation
        if not hasattr(self.velocity, 'mesh'):
            # attempt to build delaunay triangulation
            _, self.velocity.mesh = self.find_valid_triangulation(
                self.velocity.x, self.velocity.y)
        # build matplotlib triangulation object
        triangle = mtri.Triangulation(self.velocity.x, self.velocity.y,
            self.velocity.mesh.vertices)
        # create triangle plot of velocity magnitude
        tri = ax.triplot(triangle, **kwargs)
        # return the triangle plot
        return tri

    def tricontourf(self, band=None, ax=None, **kwargs):
        """
        Create plot of velocity magnitude for unstructured meshes

        Parameters
        ----------
        band: int or NoneType, default None
            band of velocity mesh to show
        ax: obj or NoneType, default None
            matplotlib figure axis
        **kwargs: dict
            Keyword arguments for ``plt.tricontourf``

        Returns
        -------
        im: obj
            matplotlib ``AxesImage`` object
        """
        if ax is None:
            ax = plt.gca()
        if band is None:
            U = getattr(self.velocity, 'U')
            V = getattr(self.velocity, 'V')
        elif (band is not None):
            U = getattr(self.velocity, 'U')[:,band]
            V = getattr(self.velocity, 'V')[:,band]
        # calculate speed
        zz = np.sqrt(U**2 + V**2)
        # get delaunay triangulation
        if not hasattr(self.velocity, 'mesh'):
            # attempt to build delaunay triangulation
            _, self.velocity.mesh = self.find_valid_triangulation(
                self.velocity.x, self.velocity.y)
        # build matplotlib triangulation object
        triangle = mtri.Triangulation(self.velocity.x, self.velocity.y,
            self.velocity.mesh.vertices)
        # create triangle plot of velocity magnitude
        tri = ax.tricontourf(triangle, zz, **kwargs)
        # return the triangle plot
        return tri

    def streamplot(self, band=None, ax=None, xy_scale=1.0, density=[0.5, 0.5], color='0.3', **kwargs):
        """
        Create stream plot of velocity vectors

        Parameters
        ----------
        band: int or NoneType, default None
            band of velocity grid to show
        ax: obj or NoneType, default None
            matplotlib figure axis
        xy_scale: float, default 1.0
            Scaling factor for converting horizontal coordinates
        density: float, default [0.5, 0.5]
            Closeness of streamlines
        color: str, default '0.3'
            Streamline color
        **kwargs: dict
            Keyword arguments for ``plt.streamplot``

        Returns
        -------
        sp: obj
            matplotlib ``StreamplotSet`` object
        """
        if ax is None:
            ax = plt.gca()
        if band is None:
            U = getattr(self.velocity, 'U')
            V = getattr(self.velocity, 'V')
        elif (band is not None):
            U = getattr(self.velocity, 'U')[:,:,band]
            V = getattr(self.velocity, 'V')[:,:,band]
        # add stream plot of velocity vectors
        gridx,gridy = np.meshgrid(self.velocity.x*xy_scale, self.velocity.y*xy_scale)
        sp = ax.streamplot(gridx, gridy, U, V, density=density, color=color, **kwargs)
        return sp
