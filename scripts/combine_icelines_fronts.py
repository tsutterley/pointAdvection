#!/usr/bin/env python
u"""
combine_icelines_fronts.py
Written by Tyler Sutterley (05/2023)
Combines ice front masks into mosaics

COMMAND LINE OPTIONS:
    --help: list the command line options
    -D X, --directory X: working data directory
    -R X, --region X: ice front regions
    --mask-file X: initial ice mask file
    -e X, --epoch X: Reference epoch of input mask
    -Y X, --year X: Years of ice front data to run
    -i X, --interval X: Time inverval of ice front data to run
    -V, --verbose: Verbose output of processing run
    -M X, --mode X: permissions mode of the output files

UPDATE HISTORY:
    Updated 05/2023: using pathlib to define and expand paths
        allow reading of different mask file types
    Written 02/2023
"""
import sys
import os
import re
import time
import logging
import pathlib
import argparse
import warnings
import traceback
import numpy as np
import pointAdvection

# attempt imports
try:
    import pointCollection as pc
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("pointCollection not available", ImportWarning)
try:
    import pyproj
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("pyproj not available", ImportWarning)
try:
    import xarray as xr
except (ImportError, ModuleNotFoundError) as exc:
    warnings.filterwarnings("module")
    warnings.warn("xarray not available", ImportWarning)
# ignore warnings
warnings.filterwarnings("ignore")

# PURPOSE: keep track of threads
def info(args):
    logging.info(pathlib.Path(sys.argv[0]).name)
    logging.info(args)
    logging.info(f'module name: {__name__}')
    if hasattr(os, 'getppid'):
        logging.info(f'parent process: {os.getppid():d}')
    logging.info(f'process id: {os.getpid():d}')

# PURPOSE: combines ice front masks into mosaics
def combine_icelines_fronts(base_dir, regions,
    mask_file=None,
    years=None,
    interval=None,
    band=0,
    mode=None):

    # directory setup
    base_dir = pathlib.Path(base_dir).expanduser().absolute()

    # read initial mask
    mask_file = pathlib.Path(mask_file).expanduser().absolute()
    if mask_file.suffix[1:] in ('tif','geotiff','tiff'):
        mask = pc.grid.data().from_geotif(str(mask_file))
    else:
        dinput = xr.open_dataset(mask_file).isel(phony_dim_0 = band)
        thedict = dict(x=dinput.x.values, y=dinput.y.values, z=dinput.z.values)
        mask = pc.grid.data().from_dict(thedict)
    # convert nan values to 0
    mask.z = np.nan_to_num(mask.z, nan=0).astype(np.uint8)
    temp = np.copy(mask.z)
    # size, extent and spacing of mask dataset
    ny, nx = mask.shape
    xmin, xmax, ymin, ymax = np.copy(mask.extent)
    dx, = np.abs(np.diff(mask.x[0:2]))
    dy, = np.abs(np.diff(mask.y[0:2]))
    logging.info(f'x-limits: {xmin:0.0f} {xmax:0.0f}')
    logging.info(f'y-limits: {ymin:0.0f} {ymax:0.0f}')
    logging.info(f'spacing: {dx:0.0f},{dy:0.0f}')
    # Climate and Forecast (CF) Metadata Conventions
    crs = pyproj.CRS.from_wkt(mask.projection)
    x_cf,y_cf = crs.cs_to_cf()
    # output attributes
    attributes = dict(ROOT={}, x={}, y={}, t={}, z={})
    attributes['ROOT']['Conventions'] = 'CF-1.6'
    today = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    attributes['ROOT']['date_created'] = today
    for att_name in ['long_name','standard_name','units']:
        attributes['x'][att_name] = x_cf[att_name]
        attributes['y'][att_name] = y_cf[att_name]

    # for each year to run
    for y in years:
        # number of time points for a given interval
        if (interval == 'daily'):
            dpm = pointAdvection.time.calendar_days(y)
            nt = np.sum(dpm)
            time_units = 'days'
        elif (interval == 'monthly'):
            nt = 12
            time_units = 'months'
        # output mask dataset for year
        output = pc.grid.mosaic()
        output.update_spacing(mask)
        output.update_bounds(mask)
        output.update_dimensions(mask)
        output.t = np.arange(nt)
        output.fill_value = 0
        # initialize output with initial mask
        z = np.zeros((ny, nx, nt), dtype=np.uint8)
        z[:] = np.broadcast_to(temp[:,:,np.newaxis], (ny, nx, nt))
        output.assign(dict(mask=z))
        # output time attributes
        attributes['t']['units'] = f'{time_units} since {y}-01-01T00:00:00'

        # for each region
        for region in regions:
            # regular expression pattern for finding files and
            # extracting information from file names
            regex_pattern = rf'({region})_({y:4d})\-(\d{{2}})\-(\d{{2}}).tif$'
            rx = re.compile(regex_pattern, re.VERBOSE | re.IGNORECASE)
            # directory for region masks
            directory = base_dir.joinpath(region)
            # find mask files for region
            mask_files = [f for f in directory.iterdir() if rx.match(f.name)]
            # for each mask file
            for region_file in sorted(mask_files):
                # read regional mask
                reg, YY, MM, DD = rx.findall(region_file.name).pop()
                region = pc.grid.data().from_geotif(str(region_file))
                region.z = np.nan_to_num(region.z, nan=0).astype(np.uint8)
                # get image coordinate of regional mask
                indy, indx = output.image_coordinates(region)
                # get temporal coordinate of regional mask
                if (interval == 'daily'):
                    indt = np.sum(dpm[:int(MM) - 1]) + int(DD) - 1
                elif (interval == 'monthly'):
                    indt = int(MM) - 1
                # update mask
                for t in range(indt, nt):
                    output.mask[indy, indx, t] = region.z[:,:]

        # output to netCDF4
        output_file = base_dir.joinpath(f'antarctic_icelines_mask_{y:4d}.nc')
        output.to_nc(str(output_file),
            replace=True, attributes=attributes,
            srs_wkt=crs.to_wkt())
        # change the permissions mode
        output_file.chmod(mode=mode)
        # save final mask for initializing the next iteration
        temp = np.copy(output.mask[:,:,-1])

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Combines ice front masks into mosaics
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = \
        pointAdvection.utilities.convert_arg_line_to_args
    # command line parameters
    # working data directory for location of ice fronts
    parser.add_argument('--directory','-D',
        type=pathlib.Path, default=pathlib.Path.cwd(),
        help='Working data directory')
    parser.add_argument('--region','-R',
        required=True, type=str, nargs='+',
        help='Ice front regions')
    parser.add_argument('--mask-file', required=True,
        type=pathlib.Path,
        help='Initial ice mask file')
    parser.add_argument('--band','-b',
        metavar='BAND', type=int,
        default=0,
        help='Time slice of mask file to use')
    # years of ice front data to run
    parser.add_argument('--year','-Y',
        type=int, nargs='+', default=[2021, 2022],
        help='Years of ice front data to run')
    # Time interval of ice front data to run
    parser.add_argument('--interval','-i',
        metavar='INTERVAL', type=str,
        choices=('daily','monthly'), default='daily',
        help='Time inverval of ice front data to run')
    # print information about processing run
    parser.add_argument('--verbose','-V',
        action='count', default=0,
        help='Verbose output of processing run')
    # permissions mode of the output files
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permissions mode of the output files')
    # return the parser
    return parser

# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    # create logger
    loglevels = [logging.CRITICAL, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=loglevels[args.verbose])

    # try to run the program with listed parameters
    try:
        info(args)
        # run program with parameters
        combine_icelines_fronts(args.directory, args.region,
            mask_file=args.mask_file,
            years=args.year,
            interval=args.interval,
            mode=args.mode)
    except Exception as exc:
        # if there has been an error exception
        # print the type, value, and stack trace of the
        # current exception being handled
        logging.critical(f'process id {os.getpid():d} failed')
        logging.error(traceback.format_exc())

# run main program
if __name__ == '__main__':
    main()
