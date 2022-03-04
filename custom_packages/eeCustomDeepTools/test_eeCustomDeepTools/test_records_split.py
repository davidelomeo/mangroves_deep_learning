#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script tests the dataset split functions.
#
# As tensorflow.data do not provide much flexibility to check datatypes
# unless converting the data to numpy arrays, the test only checks if
# the function returns None when providing wrong parameters and returns
# 'not None' if the function returns an object. The issue with converting
# to numpy arrays is that the method is really time consuming, especially
# if using large datasets. It is therefore adviced to avoid it unless
# absolutely necessary.
#
# Author: Davide Lomeo
# Email: davide.lomeo20@imperial.ac.uk
# GitHub: https://github.com/acse-2020/acse2020-acse9-finalreport-acse-dl1420-3
# Date: 22 July 2021
# Version: 1.0

import tensorflow as tf
from eeCustomDeepTools import dataset_split


def test_dataset_split():
    "Testing the dataset_split() function"

    # Loading a sample dataset
    sample_dataset = tf.data.TFRecordDataset(
        'TFRecord_samples/record_256x256-.tfrecord.gz',
        compression_type='GZIP')

    function_output_1 = dataset_split(sample_dataset, 2, 0.8, 0.1, 0.1)
    function_output_2 = dataset_split(sample_dataset, 2, 0.8, 0.1, 0.2)
    function_output_3 = dataset_split(sample_dataset, 2, 0.8, 0.2)
    function_output_4 = dataset_split(sample_dataset, 2, 1, 0.2)
    function_output_5 = dataset_split(sample_dataset, -2, 0.8, 0.2)

    assert function_output_1[0] is not None
    assert function_output_1[1] is not None
    assert function_output_1[2] is not None

    assert function_output_2[0] is None
    assert function_output_2[1] is None
    assert function_output_2[2] is None

    assert function_output_3[0] is not None
    assert function_output_3[1] is not None

    assert function_output_4[0] is None
    assert function_output_4[1] is None

    assert function_output_5[0] is None
    assert function_output_5[1] is None

    return
