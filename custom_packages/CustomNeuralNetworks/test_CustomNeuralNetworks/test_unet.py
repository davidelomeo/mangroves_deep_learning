#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script tests the function that builds the Un-Net model. The test
# itself does not look for numerical values but checks if the model
# returns am object or not. This is because there are several tests
# within the UNet class that checks if the input parameters are valid
# and returns None if they are not. The test, therefor, simply checks
# if these preliminary tests work as intended.
#
# Author: Davide Lomeo
# Email: davide.lomeo20@imperial.ac.uk
# GitHub: https://github.com/acse-2020/acse2020-acse9-finalreport-acse-dl1420-3
# Date: 16 July 2021
# Version: 1.0

from CustomNeuralNetworks import unet


def test_UNet():
    "Testing the UNet class"

    u_net = unet.UNet(7)
    function_output_1 = u_net.build_model((256, 250, 12))
    function_output_2 = u_net.build_model((256, 256, -12))
    function_output_3 = u_net.build_model((300, 300, 12))
    function_output_4 = u_net.build_model((256, 256, 12))

    assert function_output_1 is None
    assert function_output_2 is None
    assert function_output_3 is None
    assert function_output_4 is not None

    return
