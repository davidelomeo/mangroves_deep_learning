#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script tests the cloud masking functions
#
# Author: Davide Lomeo
# Email: davide.lomeo20@imperial.ac.uk
# GitHub: https://github.com/acse-2020/acse2020-acse9-finalreport-acse-dl1420-3
# Date: 20 July 2021
# Version: 1.0

import ee
from eeCustomTools import mask_sentinel_clouds, mask_landsat_clouds

# Initialising Earth Engine for testing purposes
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()


def test_mask_sentinel_clouds():
    "Testing the mask_sentinel_clouds() function"

    # Loading a Sentinel-2 ImageCollection for testing purposes
    image_collection = ee.ImageCollection('COPERNICUS/S2') \
                         .filterDate('2017-01-01', '2017-01-31')

    function_output_1 = mask_sentinel_clouds(image_collection)
    function_output_2 = mask_sentinel_clouds(image_collection.first())

    assert function_output_1 is None
    assert function_output_2.name() == 'Image'

    return


def test_mask_landsat_clouds():
    "Testing the mask_sentinel_clouds() function"

    # Loading a Landsat 7 ImageCollection for testing purposes
    image_collection = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR') \
                         .filterDate('2017-01-01', '2017-01-31')

    function_output_1 = mask_landsat_clouds(image_collection)
    function_output_2 = mask_landsat_clouds(image_collection.first())

    assert function_output_1 is None
    assert function_output_2.name() == 'Image'

    return
