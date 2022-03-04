#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script reads a list of TFRecords generated with the purpose of
# cross-validating the results of the prediction with the classification
# done through a traditional classifier on Google Earth Engine.
#
# Author: Davide Lomeo
# Email: davide.lomeo20@imperial.ac.uk
# GitHub: https://github.com/acse-2020/acse2020-acse9-finalreport-acse-dl1420-3
# Date: 12 August 2021
# Version: 0.1.0

import tensorflow as tf
from pprint import pprint

__all__ = ['prepare_prediction_classes']


def prepare_prediction_classes(file_list, dims, bands, one_hot=False,
                               num_classes=None, verbose=True):
    """
    Function specifically designed to prepare a dataset containing
    the classification matrix obtained with the traditional classifier
    on Google Earth Engine. The resultant dataset will be used for
    cross-validation between the classifier and the predictions
    generated with a Keras model.
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
    one_hot : bool, optional
        Flag to request the classification as one-hot matrices
    num_classes : int, optional
        Number of classes of the classification. Only required if one_hot=True
    verbose : bool, optional
        Flag to output the content of the dictionary of features

    Returns
    -------
    tf.data.TFRecordDataset
        Tensorflow dataset containing the classification values
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
        dims, dtype=tf.int64) for x in bands}

    if verbose:
        pprint(features_dict)

    def parse_image(example_proto):
        "Function that parses each input feature to the feature_dict"
        parsed_features = tf.io.parse_single_example(
            example_proto, features_dict)

        # pulling the feature of the label
        labels = parsed_features.pop('classes')

        # returning the parsed record and it corresponding labels as a tuple
        return tf.cast(labels, tf.int64)

    def parse_one_hot(inputs):
        return (tf.one_hot(indices=inputs, depth=num_classes))

    # Parsing each TFrecords to the feature dictionary in order to
    # obtain a TensorFlow dataset with
    dataset = dataset.map(parse_image, num_parallel_calls=5)

    # producing num_class on-hot matrices if required
    if one_hot:
        dataset = dataset.map(parse_one_hot, num_parallel_calls=5)

    return dataset
