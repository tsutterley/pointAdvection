#!/usr/bin/env python
u"""
mask_ice_fronts.py
Written by Tyler Sutterley (12/2022)
Creates time-variable ice front masks using data from
    the DLR Icelines Download Service
https://download.geoservice.dlr.de/icelines/files/

COMMAND LINE OPTIONS:
    --help: list the command line options
    -D X, --directory X: working data directory
    -R X, --region X: ice front regions
    --velocity-file X: ice sheet velocity file
    --mask-file X: initial ice mask file
    -e X, --epoch X: Reference epoch of input mask
    -Y X, --year X: Years of ice front data to run
    -B X, --buffer X: Distance in kilometers to buffer extents
    -I X, --interpolate X: Interpolation method
        spline
        linear
        nearest
        bilinear
    -V, --verbose: Verbose output of processing run
    -M X, --mode X: permissions mode of the output files

UPDATE HISTORY:
    Updated 12/2022: using virtual file systems to access files
    Written 08/2022
"""
import sys
import os
import re
import logging
import argparse
import datetime
import warnings
import posixpath
import traceback
import numpy as np
import pointAdvection

# attempt imports
try:
    import fiona
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("module")
    warnings.warn("fiona not available", ImportWarning)
try:
    import pointCollection as pc
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("module")
    warnings.warn("pointCollection not available", ImportWarning)
try:
    import pyproj
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("module")
    warnings.warn("pyproj not available", ImportWarning)
try:
    import shapely.geometry
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("module")
    warnings.warn("shapely not available", ImportWarning)
# ignore warnings
warnings.filterwarnings("ignore")

# PURPOSE: keep track of threads
def info(args):
    logging.info(os.path.basename(sys.argv[0]))
    logging.info(args)
    logging.info(f'module name: {__name__}')
    if hasattr(os, 'getppid'):
        logging.info(f'parent process: {os.getppid():d}')
    logging.info(f'process id: {os.getpid():d}')

# PURPOSE: create time-variable ice front masks
def mask_ice_fronts(base_dir, regions,
    velocity_file=None,
    mask_file=None,
    epoch=None,
    years=None,
    buffer=0,
    method=None,
    mode=None):

    # dictionary of files for dates
    ice_front_files = {}
    # total bounds
    minx, miny, maxx, maxy = np.inf, np.inf, -np.inf, -np.inf
    # regular expression pattern for finding files and
    # extracting information from file names
    regex_years = r'|'.join(f'{y:d}' for y in years) if years else r'\d+'
    regex_pattern = rf'(.*?)_({regex_years})(\d{{2}})(\d{{2}})_(.*?)-(.*?).gpkg$'
    rx = re.compile(regex_pattern, re.VERBOSE)

    # get all available regions from icelines service
    HOST = ['https://download.geoservice.dlr.de','icelines','files']
    if (regions is None):
        colnames,_ = pointAdvection.utilities.geoservice_list(HOST,
            pattern=r'[\w]\/', sort=True)
        regions = [r.replace(posixpath.sep,'') for r in colnames]

    # for each region to read
    for region in regions:
        # url for region
        region_url = [*HOST, region, 'daily', 'fronts']
        colnames,_ = pointAdvection.utilities.geoservice_list(region_url,
            pattern=regex_pattern, sort=True)
        # for each regional file
        for f in colnames:
            # extract information from file
            SAT, YY, MM, DD, ID, ICES = rx.findall(f).pop()
            # create list for day
            if f'{YY}-{MM}-{DD}' not in ice_front_files.keys():
                ice_front_files[f'{YY}-{MM}-{DD}'] = []
            # read file to extract bounds
            mmap_name = posixpath.join('/vsicurl',*region_url,f)
            ds = fiona.open(mmap_name)
            # coordinate reference system of file
            crs = pyproj.CRS.from_string(ds.crs['init'])
            # try to extract the bounds of the dataset
            try:
                # determine if bounds need to be extended
                if (ds.bounds[0] < minx):
                    minx = np.copy(ds.bounds[0])
                if (ds.bounds[1] < miny):
                    miny = np.copy(ds.bounds[1])
                if (ds.bounds[2] > maxx):
                    maxx = np.copy(ds.bounds[2])
                if (ds.bounds[3] > maxy):
                    maxy = np.copy(ds.bounds[3])
            except fiona.errors.DriverError:
                pass
            else:
                # append filename to list
                ice_front_files[f'{YY}-{MM}-{DD}'].append(mmap_name)

    # calculate buffered limits of x and y
    xlimits = (minx - 1e3*buffer, maxx + 1e3*buffer)
    ylimits = (miny - 1e3*buffer, maxy + 1e3*buffer)
    # scale for converting from m/yr to m/s
    scale = 1.0/31557600.0
    # time steps to calculate advection
    step = 86400.0
    # distance to densify geometry along path
    distance = 10

    # create advection object with interpolated velocities
    kwargs = dict(integrator='RK4', method=method)
    adv = pointAdvection.advection(**kwargs).from_nc(
        velocity_file, bounds=[xlimits,ylimits],
        field_mapping=dict(U='vx', V='vy'), scale=scale)

    # read initial mask
    mask = pc.grid.data().from_geotif(mask_file,
        bounds=[xlimits, ylimits])
    mask.z = np.nan_to_num(mask.z, nan=0)
    mask.z = mask.z.astype(bool)
    # extent of mask dataset
    xmin, xmax, ymin, ymax = np.copy(mask.extent)
    dx, = np.abs(np.diff(mask.x[0:2]))
    dy, = np.abs(np.diff(mask.y[0:2]))

    # start and end time for forward and backwards advection
    start_date = None
    end_date = None
    # for each date in the list of files
    ice_front_dates = sorted(ice_front_files.keys())
    for i,date in enumerate(ice_front_dates):
        # only run if date with all regions
        if len(ice_front_files[date]) < len(regions):
            continue
        # log date
        logging.info(date)
        # extract information
        YY1, MM1, DD1 = np.array(date.split('-'), dtype='f')
        # get dates in J2000 seconds
        J2000 = pointAdvection.time.convert_calendar_dates(
            YY1, MM1, DD1, epoch=(2000,1,1,0,0,0), scale=86400.0)
        # initial start and end date to run to create mask
        if start_date is None:
            start_date = pointAdvection.time.convert_calendar_dates(
                *epoch, epoch=(2000,1,1,0,0,0), scale=86400.0)
        if end_date is None:
            now = datetime.datetime.now()
            end_date = pointAdvection.time.convert_calendar_dates(
                now.year, now.month, now.day,
                epoch=(2000,1,1,0,0,0), scale=86400.0)

        # convert polylines to points
        x = []
        y = []
        for f in sorted(ice_front_files[date]):
            logging.info(os.path.basename(f))
            # read geopackage url and extract coordinates
            ds = fiona.open(f)
            # iterate over features
            for key, val in ds.items():
                # iterate over geometries and convert to linestrings
                for geo in val['geometry']['coordinates']:
                    line = shapely.geometry.LineString(geo)
                    # calculate the distances along the line string at spacing
                    distances = np.arange(0, line.length, distance)
                    # interpolate along path to densify the geometry
                    points = [line.interpolate(d) for d in distances] + \
                        [line.boundary[1]]
                    # extract each point in the densified geometry
                    for p in points:
                        # try to extract the point
                        try:
                            x.append(p.x)
                            y.append(p.y)
                        except IndexError:
                            pass

        # total number of points
        logging.info(f'Total points: {len(x):d}')
        # set original x and y coordinates
        adv.x = np.copy(x)
        adv.y = np.copy(y)
        # run advection for each time step
        # run forward in time to find masked points
        for t in np.arange(J2000, end_date, step):
            # update times
            adv.t = np.copy(t)
            adv.t0 = np.copy(t + step)
            # run advection
            adv.translate_parcel()
            # update x and y coordinates
            adv.x = np.copy(adv.x0)
            adv.y = np.copy(adv.y0)
            # verify that the advected points are within domain
            valid = np.nonzero((adv.x0 >= xmin) & (adv.x0 <= xmax) &
                (adv.y0 >= ymin) & (adv.y0 <= ymax))
            # rasterize advected points
            indx = ((adv.x0[valid] - xmin)//dx).astype(int)
            indy = ((adv.y0[valid] - ymin)//dy).astype(int)
            # set mask
            mask.z[indy, indx] = False

        # reset original x and y coordinates
        adv.x = np.copy(x)
        adv.y = np.copy(y)
        # run advection for each time step
        # run backwards in time to find valid points
        for t in np.arange(J2000, start_date, -step):
            # update times
            adv.t = np.copy(t)
            adv.t0 = np.copy(t - step)
            # run advection
            adv.translate_parcel()
            # update x and y coordinates
            adv.x = np.copy(adv.x0)
            adv.y = np.copy(adv.y0)
            # verify that the advected points are within domain
            valid = np.nonzero((adv.x0 >= xmin) & (adv.x0 <= xmax) &
                (adv.y0 >= ymin) & (adv.y0 <= ymax))
            # rasterize advected points
            indx = ((adv.x0[valid] - xmin)//dx).astype(int)
            indy = ((adv.y0[valid] - ymin)//dy).astype(int)
            # set mask
            mask.z[indy, indx] = True

        # write mask to file
        # use GDT_Byte as output data type
        output_file = os.path.join(base_dir, f'icefront_{date}.tif')
        mask.to_geotif(output_file, dtype=1, srs_wkt=crs.to_wkt())
        logging.info(output_file)
        # change the permissions mode of the output file
        os.chmod(output_file, mode=mode)
        # # update start of advection to improve computational times
        # start_date = np.copy(J2000)

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Creates time-variable ice front masks
            using data from the DLR Icelines Download Service
            """,
        fromfile_prefix_chars="@"
    )
    # command line parameters
    # working data directory for location of ice fronts
    parser.add_argument('--directory','-D',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help='Working data directory')
    parser.add_argument('--region','-R',
        required=True, type=str, nargs='+',
        help='Ice front regions')
    parser.add_argument('--velocity-file', required=True,
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        help='Ice sheet velocity file')
    parser.add_argument('--mask-file', required=True,
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        help='Initial ice mask file')
    # reference time epoch for input mask file
    parser.add_argument('--epoch','-e',
        type=int, default=(2014, 1, 1), nargs=3,
        metavar=('YEAR','MONTH','DAY'),
        help='Reference epoch of input mask')
    # years of ice front data to run
    parser.add_argument('--year','-Y',
        type=int, nargs='+',
        help='Years of ice front data to run')
    # extent buffer
    parser.add_argument('--buffer','-B',
        type=float, default=5.0,
        help='Distance in kilometers to buffer extents')
    # interpolation method
    parser.add_argument('--interpolate','-I',
        metavar='METHOD', type=str, default='spline',
        choices=('spline','linear','nearest','bilinear'),
        help='Spatial interpolation method')
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

    # check internet connection with icelines file service
    HOST = 'https://download.geoservice.dlr.de/icelines/files/'

    # try to run the program with listed parameters
    try:
        info(args)
        pointAdvection.utilities.check_connection(HOST)
        # run program with parameters
        mask_ice_fronts(args.directory, args.region,
            velocity_file=args.velocity_file,
            mask_file=args.mask_file,
            epoch=args.epoch,
            years=args.year,
            buffer=args.buffer,
            method=args.interpolate,
            mode=args.mode)
    except Exception as e:
        # if there has been an error exception
        # print the type, value, and stack trace of the
        # current exception being handled
        logging.critical(f'process id {os.getpid():d} failed')
        logging.error(traceback.format_exc())

# run main program
if __name__ == '__main__':
    main()
