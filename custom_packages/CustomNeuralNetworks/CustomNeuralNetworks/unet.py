#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script implements the U-Net architecture as originally desgined
# by Ronneberg et al. (2015): https://arxiv.org/pdf/1505.04597.pdf.
#
# Author: Davide Lomeo,
# Email: davide.lomeo20@imperial.ac.uk
# GitHub: https://github.com/acse-2020/acse2020-acse9-finalreport-acse-dl1420-3
# Date: 26 July 2021
# Version: 0.1.0

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

__all__ = ['UNet']


class UNet:
    """
    Class that implements the U-Net architecture as originally desgined
    by Ronneberg et al. (2015): https://arxiv.org/pdf/1505.04597.pdf.

    The main difference from the original U-Net is with the output of
    the last layer, as this applys a softmax probability on each pixel of the
    image. This is done to accomodate the multi-class classification.

    Parameters
    ----------
    n_classes : int
        number of final channels

    Methods
    -------
    build_model(input_shape)
        Function that builds the U-Net using the input image shape
    """
    def __init__(self, n_classes):
        "Class constructor"
        super().__init__()
        self.n_classes = n_classes

    def build_model(self, input_shape):
        """
        Function that implement the UNet model as designed by Ronneberg
        et al. (2015) (https://arxiv.org/pdf/1505.04597.pdf) using a
        softmax probability for pixels to belong to one of the input n classes.
        The input image height and width needs to be equal and multiple of 16
        for the architecture to build succesfully (e.g. 128, 256, 384, etc.)

        Parameters
        ----------
        input_shape : tuple
            Tuple containing the sizes of the input image (H, W, bands)

        Returns
        -------
        keras.model
            model ready to be used for training
        """

        if len(input_shape) != 3:
            print('''ERROR: The input shape is invalid. Ensure to provide height,
            width and number of channels as a 3 integers tuple''')
            return None
        elif input_shape[0] != input_shape[1]:
            print('''ERROR: the input width and height are different. This U-Net
            architecture is designed to only handle equal width and height''')
            return None
        elif input_shape[2] <= 0:
            print('ERROR: Invalid number of bands. Use a positive integer')
            return None
        elif input_shape[0] % 16 != 0:
            print('''ERROR: this U-Net can only take image width and height that
            are multiple of 16 (i.e., 64, 126, 256, 384, 416, etc.)  ''')
            return None

        # Adapting the first layer of the model to the input image's shape
        input_img = layers.Input(input_shape)

        # Encoding of the image
        s1, p1 = self.__encoder_block(input_img, 64)
        s2, p2 = self.__encoder_block(p1, 128)
        s3, p3 = self.__encoder_block(p2, 256)
        s4, p4 = self.__encoder_block(p3, 512)

        # Bridging the encoding part to the decoding part
        b1 = self.__conv_block(p4, 1024)

        # Decoding of the image
        d1 = self.__decoder_block(b1, s4, 512)
        d2 = self.__decoder_block(d1, s3, 256)
        d3 = self.__decoder_block(d2, s2, 128)
        d4 = self.__decoder_block(d3, s1, 64)

        # Defining the last layer of the model with a softmax
        # in order to get probabilities for each pixel to belong
        # to one of the n_classes expected classes
        output_img = layers.Conv2D(self.n_classes, (1, 1),
                                   activation=tf.nn.softmax)(d4)

        # Building the model using input and output layers
        model = Model(input_img, output_img, name='U-Net')

        return model

    def __conv_block(self, input_tensor, num_filters):
        "Function that implements image convolution"

        x = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    def __encoder_block(self, input_tensor, num_filters):
        "Function that downscale the input image size after"

        x = self.__conv_block(input_tensor, num_filters)
        p = layers.MaxPooling2D((2, 2))(x)

        return x, p

    def __decoder_block(self, input_tensor, skip_features, num_filters):
        "Function that upscales the input image size"

        x = layers.Conv2DTranspose(num_filters, (2, 2),
                                   strides=2, padding='same')(input_tensor)
        x = layers.Concatenate()([x, skip_features])
        x = self.__conv_block(x, num_filters)
        return x
