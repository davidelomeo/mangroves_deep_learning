#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script contains functions to mask out clouds from Sentinel-2 and
# Landsat 5-7-8 Images
#
# Author: Davide Lomeo,
# Email: davide.lomeo20@imperial.ac.uk
# GitHub: https://github.com/acse-2020/acse2020-acse9-finalreport-acse-dl1420-3
# Date: 19 July 2021
# Version: 0.1.0

import math

__all__ = ['mask_sentinel_clouds', 'mask_landsat_clouds']


def mask_sentinel_clouds(img):
    """
    Function that masks out clouds from the input Sentinel-2 image. The
    input image has to have the QA60 band in its bands list.

    Parameters
    ----------
    img : ee.image.Image
        Single Sentinel-2 Earth Engine Image that needs cloud-masking

    Returns
    -------
    ee.image.Image
        The cloud-masked image
    """

    try:
        # selecting the qa band
        qa = img.select('QA60')

        # Bits 10 belong to clouds, bits 11 belong to cirrus
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11

        # Setting flags to 0 to indicate no cloudy condtions
        mask = qa.bitwiseAnd(cloudBitMask).eq(0) \
            .And(qa.bitwiseAnd(cirrusBitMask).eq(0))

        return img.updateMask(mask).divide(10000)

    # The function will return an error message if the input is not
    # of type <class 'ee.image.Image'>
    except AttributeError:
        print("""
        Error: the input is {}. It needs to be a <class 'ee.image.Image'>
        """.format(str(type(img))))
        return None


def mask_landsat_clouds(img):
    """
    Function that masks out clouds and their shadows from the input
    Landsat image. The function works for any Landsat sensor as it
    only uses the pixel_qa band, common to all Landsat sensors. The
    input image has to have the pixel_qa band in its bands list.

    Parameters
    ----------
    img : ee.image.Image
        Single Landsat Earth Engine Image that needs cloud-masking

    Returns
    -------
    ee.image.Image
        The cloud-masked image
    """

    try:
        def get_qa_bits(img, start, end, new_name):

            # Computing the bits tha need to be extacted
            bits_etxracted = 0
            for i in range(start, end, 1):
                bits_etxracted += math.pow(2, i)

            # Returning a single band img of the extracted qa bits,
            # and giving it a new name
            return img.select([0], [new_name]) \
                      .bitwiseAnd(bits_etxracted) \
                      .rightShift(start)

        def compute_shadow(img):
            "Function to mask out cloud shadows pixels"

            # Select the pixel_qa band
            qa = img.select(['pixel_qa'])
            return get_qa_bits(qa, 3, 3, 'Cloud_shadows').eq(0)

        def compute_clouds(img):
            "Function to mask out cloudy pixels"

            # Select the pixel_qa band
            qa = img.select(['pixel_qa'])
            return get_qa_bits(qa, 5, 5, 'Cloud').eq(0)

        # Calling the function to mask both clouds and clouds shadows
        # and returning the maksed image
        clouds_shadow = compute_shadow(img)
        clouds = compute_clouds(img)
        img = img.updateMask(clouds_shadow)

        return img.updateMask(clouds)

    # The function will return an error message if the input is not
    # of type <class 'ee.image.Image'>
    except AttributeError:
        print("""
        Error: the input is {}. It needs to be a <class 'ee.image.Image'>
        """.format(str(type(img))))
        return None
