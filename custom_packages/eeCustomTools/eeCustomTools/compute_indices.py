#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script contains functions that compute:
# NDVI, NDWI, MNDWI, NDSI, NDMI, EVI, EVI2, GOSAVI, SAVI
# for Sentinel-2 and Landsats 5-6-7 sensors.
#
# The functions can be called on a single image or can be used
# with the .map() method on an ImageCollection.
#
# Author: Davide Lomeo,
# Email: davide.lomeo20@imperial.ac.uk
# GitHub: https://github.com/acse-2020/acse2020-acse9-finalreport-acse-dl1420-3
# Date: 19 July 2021
# Version: 0.1.0

__all__ = ['sentinel2_spectral_indices', 'get_landsat_indices',
           'landsat57_spectral_indices', 'landsat8_spectral_indices']


def sentinel2_spectral_indices(img):
    """
    Function that computes several spectral indices with particular
    focus in detecting vegetation phenology, and water and salt content
    in the soil. The indices are added to the input image as bands. The
    function can be used either when calling a single image or using the
    .map() method to an ImageCollection to compute the indices for each
    image in it.

    Parameters
    ----------
    img : ee.image.Image
        Single Earth Engine Image that needs adding of the spectral indices

    Returns
    -------
    ee.image.Image
        The image with the added indices as bands
    """
    try:
        # Computing the Normalised Difference Vegetation Index
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')

        # Computing the Normalised Difference Water Index
        ndwi = img.normalizedDifference(['B8', 'B3']).rename('NDWI')

        # Computing the Modified Normalised Difference Water Index
        mndwi = img.normalizedDifference(['B11', 'B3']).rename('MNDWI')

        # Computing the Normalised Difference Salinity Index
        ndsi = img.normalizedDifference(['B12', 'B11']).rename('NDSI')

        # Computing the Normalised Difference Moisture Index
        ndmi = img.normalizedDifference(['B11', 'B8']).rename('NDMI')

        # Computing the Enhanced Vegetation Index
        evi = img.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
                'NIR': img.select('B8'),
                'RED': img.select('B4'),
                'BLUE': img.select('B2')}).rename('EVI')

        # Computing the Enhanced Vegetation Index 2
        evi2 = img.expression(
            '2.4 * ((NIR - RED) / (NIR + RED + 1))', {
                'NIR': img.select('B8'),
                'RED': img.select('B4')}).rename('EVI2')

        # Computing the Green Optimized Soil Adjusted Vegetation Index
        gosavi = img.expression(
            '(NIR - G) / (NIR + G + 0.16) ', {
                'NIR': img.select('B8'),
                'G': img.select('B3')}).rename('GOSAVI')

        # Computing the Soil Adjusted Vegetation Index
        savi = img.expression(
            '1.5 * (NIR - RED) / (NIR + RED + 0.5) ', {
                'NIR': img.select('B8'),
                'RED': img.select('B4')}).rename('SAVI')

        return img.addBands([ndvi, ndwi, mndwi, ndsi,
                             ndmi, evi, evi2, gosavi, savi])

    # The function will return an error message if the input is not
    # of type <class 'ee.image.Image'>
    except AttributeError:
        print("""
        Error: the input is {}. It needs to be a <class 'ee.image.Image'>
        """.format(str(type(img))))
        return None


def get_landsat_indices(img, bands):
    """
    Helper function that computes several spectral indices
    for the input image using the specified bands dictionary
    """

    try:
        # Computing the Normalised Difference Vegetation Index
        ndvi = img.normalizedDifference(
            [bands['NIR'], bands['RED']]).rename('NDVI')

        # Computing the Normalised Difference Water Index
        ndwi = img.normalizedDifference(
            [bands['SWIR'], bands['NIR']]).rename('NDWI')

        # Computing the Modified Normalised Difference Water Index
        mndwi = img.normalizedDifference(
            [bands['SWIR'], bands['RED']]).rename('MNDWI')

        # Computing the Normalised Difference Salinity Index
        ndsi = img.normalizedDifference(
            [bands['MIR'], bands['YELLOW']]).rename('NDSI')

        # Computing the Normalised Difference Moisture Index
        ndmi = img.normalizedDifference(
            [bands['SWIR'], bands['NIR']]).rename('NDMI')

        # Computing the Enhanced Vegetation Index
        evi = img.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
                'NIR': img.select(bands['NIR']),
                'RED': img.select(bands['RED']),
                'BLUE': img.select(bands['BLUE'])}).rename('EVI')

        # Computing the Enhanced Vegetation Index 2
        evi2 = img.expression(
            '2.4 * ((NIR - RED) / (NIR + RED + 1))', {
                'NIR': img.select(bands['NIR']),
                'RED': img.select(bands['RED'])}).rename('EVI2')

        # Computing the Green Optimized Soil Adjusted Vegetation Index
        gosavi = img.expression(
            '(NIR - G) / (NIR + G + 0.16) ', {
                'NIR': img.select(bands['NIR']),
                'G': img.select(bands['YELLOW'])}).rename('GOSAVI')

        # Computing the Soil Adjusted Vegetation Index
        savi = img.expression(
            '1.5 * (NIR - RED) / (NIR + RED + 0.5) ', {
                'NIR': img.select(bands['NIR']),
                'RED': img.select(bands['RED'])}).rename('SAVI')

        return img.addBands([ndvi, ndwi, mndwi, ndsi,
                            ndmi, evi, evi2, gosavi, savi])

    # The function will return an error message if the input is not
    # of type <class 'ee.image.Image'>
    except AttributeError:
        print("""
        Error: the input is {}. It needs to be a <class 'ee.image.Image'>
        """.format(str(type(img))))
        return None


def landsat57_spectral_indices(img):
    """
    Function that computes several spectral indices for Landsat 5 and 7
    sensors. The indices specifically focus in detecting vegetation phenology,
    and water and salt content in the soil. The indices are added to the
    input image as bands. The function can be used either when calling a single
    image or using the .map() method to an ImageCollection to compute the
    indices for each images in it.

    Parameters
    ----------
    img : ee.image.Image
        Single Earth Engine L5-7 Image that needs adding the spectral indices

    Returns
    -------
    ee.image.Image
        The image with the added indices as bands
    """

    # Specifying the list of bands needed to compute the spectral indices
    bands = {
        'BLUE': 'B1',
        'YELLOW': 'B2',
        'RED': 'B3',
        'NIR': 'B4',
        'SWIR': 'B5',
        'MIR': 'B7'
    }

    # Calling the function taht computes the spectral inidces
    return get_landsat_indices(img, bands)


def landsat8_spectral_indices(img):
    """
    Function that computes several spectral indices for Landsat 8 sensor.
    The indices specifically focus in detecting vegetation phenology,
    and water and salt content in the soil. The indices are added to the
    input image as bands. The function can be used either when calling a single
    image or using the .map() method to an ImageCollection to compute the
    indices for each images in it.

    Parameters
    ----------
    img : ee.image.Image
        Single Earth Engine L8 Image that needs adding the spectral indices

    Returns
    -------
    ee.image.Image
        The image with the added indices as bands
    """

    # Specifying the list of bands needed to compute the spectral indices
    bands = {
        'BLUE': 'B2',
        'YELLOW': 'B3',
        'RED': 'B4',
        'NIR': 'B5',
        'SWIR': 'B7',
        'MIR': 'B8'
    }

    # Calling the function taht computes the spectral inidces
    return get_landsat_indices(img, bands)
