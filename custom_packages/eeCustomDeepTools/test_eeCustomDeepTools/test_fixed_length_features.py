#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script tests the functions that generates a dictionary of Features.
#
# The tests simply checks whether the function returns a dictionary
# or returns None. This is because there is a preliminary testing phase
# inside the function itself, that runs checks to ensure that the input
# parameters are valid. If one or more input parameters are invalid, then
# the function will return None and ask the user to fix the issue.
#
# Author: Davide Lomeo
# Email: davide.lomeo20@imperial.ac.uk
# GitHub: https://github.com/acse-2020/acse2020-acse9-finalreport-acse-dl1420-3
# Date: 22 July 2021
# Version: 1.0

from eeCustomDeepTools import get_features_dict


def test_get_features_dict():
    "Testing the get_features_dict() function"

    function_output_1 = get_features_dict(['B2', 'B3', 'B4'], 'classes',
                                          ['B2', 'B3', 'B4'], [256, 256])
    function_output_2 = get_features_dict(['B1'], 'c',
                                          ['B2'], [256, 256])
    function_output_3 = get_features_dict(1, 'classes',
                                          ['B3', 'B4'], [256, 256])
    function_output_4 = get_features_dict('B3', 'classes',
                                          ['B3', 'B4'], [256, 256])
    function_output_5 = get_features_dict(['B2', 'B3', 'B4'], 'classes',
                                          ['B3', 'B4'], 256)

    assert isinstance(function_output_1, dict) is True
    assert isinstance(function_output_2, dict) is True
    assert function_output_3 is None
    assert function_output_4 is None
    assert function_output_5 is None

    return
