import os
import sys
import logging
import subprocess
from setuptools import setup, find_packages

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger()

# package description and keywords
description = 'Utilities for advecting point data for use in a Lagrangian reference frame'
keywords = 'ICESat-2 laser altimetry, ATLAS, surface elevation and change, Lagrangian reference frame'
# get long_description from README.rst
with open("README.rst", mode="r", encoding='utf8') as fh:
    long_description = fh.read()
long_description_content_type = "text/x-rst"

# install requirements and dependencies
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    install_requires = []
    dependency_links = []
else:
    # get install requirements
    with open('requirements.txt', encoding='utf8') as fh:
        install_requires = [line.split().pop(0) for line in fh.read().splitlines()]
    # dependency links
    dependency_links = ['https://github.com/smithb/pointCollection/tarball/tarball/main']

# get version
with open('version.txt', encoding='utf8') as fh:
    version = fh.read()

# list of all scripts to be included with package
scripts=[os.path.join('scripts',f) for f in os.listdir('scripts') if f.endswith('.py')]

# run cmd from the command line
def check_output(cmd):
    return subprocess.check_output(cmd).decode('utf')

# check if GDAL is installed
gdal_output = [None] * 4
try:
    for i, flag in enumerate(("--cflags", "--libs", "--datadir", "--version")):
        gdal_output[i] = check_output(['gdal-config', flag]).strip()
except Exception as exc:
    log.warning('Failed to get options via gdal-config')
else:
    log.info(f"GDAL version from via gdal-config: {gdal_output[3]}")
# if setting GDAL version from via gdal-config
if gdal_output[3] and ('gdal' in install_requires):
    # add version information to gdal in install_requires
    gdal_index = install_requires.index('gdal')
    install_requires[gdal_index] = f'gdal=={gdal_output[3]}'
elif any(install_requires) and ('gdal' in install_requires):
    # gdal version not found
    gdal_index = install_requires.index('gdal')
    install_requires.pop(gdal_index)

setup(
    name='pointAdvection',
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    url='https://github.com/tsutterley/pointAdvection',
    author='Tyler Sutterley',
    author_email='tsutterl@uw.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords=keywords,
    packages=find_packages(),
    install_requires=install_requires,
    dependency_links=dependency_links,
    scripts=scripts
)
