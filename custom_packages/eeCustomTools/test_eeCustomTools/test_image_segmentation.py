#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script tests the function that perform segmentation
#
# Author: Davide Lomeo
# Email: davide.lomeo20@imperial.ac.uk
# GitHub: https://github.com/acse-2020/acse2020-acse9-finalreport-acse-dl1420-3
# Date: 20 July 2021
# Version: 1.0

import ee
from eeCustomTools import segment_image

# Initialising Earth Engine for testing purposes
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()


def test_segment_image():
    "Testing the mask_sentinel_clouds() function"

    # Loading a Sentinel-2 ImageCollection for testing purposes
    image_collection = ee.ImageCollection('COPERNICUS/S2') \
                         .filterDate('2017-01-01', '2017-01-31')

    bands = ['B1', 'B2', 'B3', 'B4', 'B5']
    function_output_1 = segment_image(image_collection, bands)
    function_output_2 = segment_image(image_collection.first(), bands)

    assert function_output_1 is None
    assert function_output_2.name() == 'Image'

    return
