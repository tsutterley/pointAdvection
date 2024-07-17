#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pointAdvection
==============

pointAdvection contains Python tools for advecting point data
for use in a Lagrangian reference frame

Documentation is available at https://pointAdvection.readthedocs.io
"""
# base modules
import pointAdvection.tools
import pointAdvection.utilities
from pointAdvection.advection import advection
import pointAdvection.version
from pointAdvection.velocity import velocity
# get version number
__version__ = pointAdvection.version.version
