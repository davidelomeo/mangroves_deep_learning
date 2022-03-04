#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script tests the function that prepares the classes dataset
# generated for cross-validation.
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
# Date: 12 Aug 2021
# Version: 1.0

from eeCustomDeepTools import prepare_prediction_classes


def test_prepare_prediction_classes():
    "Testing the prepare_prediction_classes() function"

    file_list = ['TFRecord_samples/record_256x256-.tfrecord.gz']
    dims = [256, 256]
    bands = ['classes']

    function_output_1 = prepare_prediction_classes(file_list, dims, bands)
    function_output_2 = prepare_prediction_classes(
        'record_256x256-.tfrecord.gz', dims, bands)
    function_output_3 = prepare_prediction_classes(file_list, 256, bands)
    function_output_4 = prepare_prediction_classes(file_list, dims, 'classes')

    assert function_output_1 is not None
    assert function_output_2 is None
    assert function_output_3 is None
    assert function_output_4 is None

    return
