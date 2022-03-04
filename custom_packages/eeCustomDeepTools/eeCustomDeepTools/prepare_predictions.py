#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script reads a list of TFRecords generated with the purpose of
# making predictions. Differently from the other functions in this
# package, the function does not inlcude the classification band
# in the records, and therefore, it does not split the records in
# record/ hot-encoded class tuples.
#
# Author: Davide Lomeo
# Email: davide.lomeo20@imperial.ac.uk
# GitHub: https://github.com/acse-2020/acse2020-acse9-finalreport-acse-dl1420-3
# Date: 24 July 2021
# Version: 0.1.0

import tensorflow as tf
from pprint import pprint

__all__ = ['prepare_prediction_dataset']


def prepare_prediction_dataset(file_list, dims, bands, verbose=True):
    """
    Function specifically designed to prepare a dataset destined
    for predictions. Given that this dataset does not need to be
    trained into a Keras deep model, there is much less work needed
    for its preparation, as the dataset does not need splitting and
    does not require labels. The output dataset is designed to be
    used to run predictions with a pre-trained Keras model as follow:
    -> model.predict(function_output_dataset).
    NOTE: This function is specifically designed to map pacthes of
    known dimensions (height and width).

    Parameters
    ----------
    file_list : list
        List of TFrecords file names
    dims : list
        List of 2 integers that defines the size of the patches
    bands : list
        List of bands names to inlcude in the predictions dataset
    verbose : bool, optional
        Flag to output the content of the dictionary of features

    Returns
    -------
    tf.data.TFRecordDataset
        Tensorflow dataset ready to for predictions using Keras models
    """

    if not isinstance(file_list, list):
        print('ERROR: ensure that the file_list is a list')
        return None
    elif not isinstance(dims, list):
        print('ERROR: ensure that the dimensions are input as a list')
        return None
    elif not isinstance(bands, list):
        print('ERROR: ensure that the bands are input as a list')
        return None

    # Reading the TFRecords from the input file_list paths
    dataset = tf.data.TFRecordDataset(file_list, compression_type='GZIP')

    # Generating a dictionary of features for each input band. This is
    # necessary to map and create multi-channel tensors
    features_dict = {x: tf.io.FixedLenFeature(
        dims, dtype=tf.float32) for x in bands}

    if verbose:
        pprint(features_dict)

    def parse_image(example_proto):
        "Function that parses each input feature to the feature_dict"
        parsed_features = tf.io.parse_single_example(
            example_proto, features_dict)
        return parsed_features

    def stack_images(features):
        "Function that stack all the input features"
        stacked_features = tf.transpose(
            tf.squeeze(
                tf.stack(list(features.values()))))
        return stacked_features

    # Parsing each TFrecords to the feature dictionary in order to
    # obtain multi-channel tensors and stacking the images to get
    # long tensors for each band and feature. This is required when
    # making prediction using TFRecords in Keras
    dataset = dataset \
        .map(parse_image, num_parallel_calls=5) \
        .map(stack_images, num_parallel_calls=5) \
        .batch(1)

    return dataset
