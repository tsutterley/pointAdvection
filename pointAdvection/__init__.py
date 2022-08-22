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
import pointAdvection.time
import pointAdvection.tools
from pointAdvection.advection import advection
import pointAdvection.version
# get version number
__version__ = pointAdvection.version.version
