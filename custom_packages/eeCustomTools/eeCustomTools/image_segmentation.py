#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script contains a function to segment the input image using
# the input bands to help classifiers better distinguish between objects.
#
# Author: Davide Lomeo,
# Email: davide.lomeo20@imperial.ac.uk
# GitHub: https://github.com/acse-2020/acse2020-acse9-finalreport-acse-dl1420-3
# Date: 20 July 2021
# Version: 0.1.0

import ee
__all__ = ['segment_image']


def segment_image(img, bands, grid_size=100, grid_shape='square',
                  compactness=0.8, connectivity=8):
    """
    Function that generates a segmentation of the input image using
    the input bands.

    Parameters
    ----------
    img : ee.image.Image
        Single Earth Engine Image
    bands : list
        List of bands to inlcude in the segmentation
    grid_size : int, optional
        Seed location spacing in pixels.
    grid_shape : str, optional
        The shape of the grid. Choises: 'square' or 'helix'
    comp : float, optional
        Spatial distance weighting (0.0 to 1.0). High values = more compactness
    conn : int, optional
        Connectivity of the clusters. Choises: 4 or 8

    Returns
    -------
    ee.image.Image
        The segmented Earth Engine Image
    """

    try:
        # Setting the seed for image segmentation
        seeds_grid = ee.Algorithms.Image.Segmentation.seedGrid(
            grid_size, grid_shape)

        # Image segmentation
        segmented_image = ee.Algorithms.Image.Segmentation.SNIC(
            image=img,
            compactness=compactness,
            connectivity=connectivity,
            seeds=seeds_grid)

        # Changing the name of the bands
        segmented_band_names = [b + '_mean' for b in bands]

        # Setting the name of the bands of the segmented images
        segmented_image_names = segmented_image.select(
            segmented_band_names).bandNames()

        # Switching bands names from segmented back to the original names
        segmented_image = segmented_image.select(
            segmented_image_names, img.bandNames())

        return segmented_image

    # The function will return an error message if the input is not
    # of type <class 'ee.image.Image'>
    except AttributeError:
        print("""
        Error: the input is {}. It needs to be a <class 'ee.image.Image'>
        """.format(str(type(img))))
        return None
