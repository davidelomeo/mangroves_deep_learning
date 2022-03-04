#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script generates a dictionary of Fixed Lenght Features for each
# of the input bands. The dictionary is needed when later parsing each
# TFRecord to obtain batches dataset ready to be fed into Keras deep models
#
# Author: Davide Lomeo
# Email: davide.lomeo20@imperial.ac.uk
# GitHub: https://github.com/acse-2020/acse2020-acse9-finalreport-acse-dl1420-3
# Date: 22 July 2021
# Version: 0.1.0

import tensorflow as tf

__all__ = ['get_features_dict']


def get_features_dict(bands, class_label, bands_of_interest, dims):
    """
    Function that maps the names of the bands into Fixed Lenght
    Features. This step is necessary to tell TensorFlow how to read
    the TFRecords dataset into tensors and add channels corresponding
    to each band. The function expects that the input TFRecords dataset
    were previoulsy classified in Earth Engine, and therefore, contain a
    band for the classification. NOTE: This function is specifically
    designed to map pacthes of known dimensions (height and width).

    Parameters
    ----------
    bands : list
        The bands that will need to be mapped into Tensorflow Features
    class_label : str
        The name to give to the classification band
    bands_of_interest : list
        The bands that the user wants to inlcude in the output dictionary
    dims : list
        The dimensions of the patches. E.g. [256, 256]

    Returns
    -------
    dictionary
        A dictionary containing the bands as keys and the Features as values
    """

    if not isinstance(class_label, str):
        print('ERROR: ensure that the class_label is of type string')
        return None

    elif not isinstance(dims, list):
        print('ERROR: ensure that the dimensions are input as a list')
        return None
    try:
        # Generating a list of fixed-length features. By default, tensorflow
        # expects values in float32 format
        columns = [tf.io.FixedLenFeature(
          shape=dims, dtype=tf.float32) for k in bands]

        # Adding the classes band and create a feature in int64 format.
        bands += [class_label]
        bands_of_interest += [class_label]
        columns += [tf.io.FixedLenFeature(shape=dims, dtype=tf.int64)]

        # Dictionary with names as keys, features as values.
        features_dict = dict(zip(bands, columns))

        # looping through the features_dict and only pull out the
        # bands of interest
        features_of_interest = {}
        invalid_bands = []
        for i in bands_of_interest:
            if i in features_dict.keys():
                features_of_interest[i] = features_dict[i]
            else:
                invalid_bands.append(i)

        if invalid_bands != []:
            print('''WARNING: the bands {}
                  were not found in the input bands list\n'''.format(
                    invalid_bands))

        return features_of_interest

    except TypeError:
        print('''TypeError: One of the input is invalid. Please ensure that
        the types of each of the inputs follows the guidelines as dscrbed in
        the Parameters specifications''')
        return None
