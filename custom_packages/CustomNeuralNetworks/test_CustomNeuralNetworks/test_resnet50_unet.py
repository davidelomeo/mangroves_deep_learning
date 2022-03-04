#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script tests the function that builds the Un-Net model combined
# with the ResNet50 model as an encoder. The test does not look for
# numerical values but checks if the model returns am object or not.
# This is because there are several tests within the UNet class that
# checks if the input parameters are valid and returns None if they are not.
# The test simply checks if these preliminary tests work as intended.
#
# Author: Davide Lomeo
# Email: davide.lomeo20@imperial.ac.uk
# GitHub: https://github.com/acse-2020/acse2020-acse9-finalreport-acse-dl1420-3
# Date: 1 August 2021
# Version: 1.0

from CustomNeuralNetworks import resnet50_unet


def test_ResNet50Unet():
    "Testing the ResNet50Unet class"

    resnet50unet = resnet50_unet.ResNet50Unet(7)
    function_output_1 = resnet50unet.build_model((256, 250, 3))
    function_output_2 = resnet50unet.build_model((256, 256, -3))
    function_output_3 = resnet50unet.build_model((300, 300, 3))
    function_output_4 = resnet50unet.build_model((256, 256, 3))

    assert function_output_1 is None
    assert function_output_2 is None
    assert function_output_3 is None
    assert function_output_4 is not None

    return
