#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script tests the function that prepares the prediction dataset.
#
# The tests simply checks whether the function actually an object
# or returns None. This is because there is a preliminary testing phase
# inside the function itself, that runs checks to ensure that the input
# parameters are valid. If one or more input parameters are invalid, then
# the function will return None and ask the user to fix the issue.
#
# Author: Davide Lomeo
# Email: davide.lomeo20@imperial.ac.uk
# GitHub: https://github.com/acse-2020/acse2020-acse9-finalreport-acse-dl1420-3
# Date: 24 July 2021
# Version: 1.0

from eeCustomDeepTools import prepare_prediction_dataset


def test_prepare_prediction_dataset():
    "Testing the prepare_prediction_dataset() function"

    file_list = ['TFRecord_samples/record_256x256-.tfrecord.gz']
    dims = [256, 256]
    bands = ['B2', 'B3', 'B4']

    function_output_1 = prepare_prediction_dataset(file_list, dims, bands)
    function_output_2 = prepare_prediction_dataset(
        'record_256x256-.tfrecord.gz', dims, bands)
    function_output_3 = prepare_prediction_dataset(file_list, 256, bands)
    function_output_4 = prepare_prediction_dataset(file_list, dims, 'B1')

    assert function_output_1 is not None
    assert function_output_2 is None
    assert function_output_3 is None
    assert function_output_4 is None

    return
