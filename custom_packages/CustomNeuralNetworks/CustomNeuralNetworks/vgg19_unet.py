#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script implements the U-Net architecture
# (https://arxiv.org/abs/1505.04597) with a pre-trained VGG19
# as encoder (https://arxiv.org/abs/1409.1556). The VGG19 is pre-trained on
# the Imagenet dataset (https://ieeexplore.ieee.org/abstract/document/5206848)
# and it is available directly from the tensorflow library.
#
# Author: Davide Lomeo,
# Email: davide.lomeo20@imperial.ac.uk
# GitHub: https://github.com/acse-2020/acse2020-acse9-finalreport-acse-dl1420-3
# Date: 28 July 2021
# Version: 0.1.0

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19

__all__ = ['VGG19Unet']


class VGG19Unet:
    """
    Class that implements the U-Net architecture
    (https://arxiv.org/abs/1505.04597) with a pre-trained VGG19
    as encoder (https://arxiv.org/abs/1409.1556). The VGG19 is
    pre-trained on the Imagenet dataset
    (https://ieeexplore.ieee.org/abstract/document/5206848) and it is available
    directly from the tensorflow library.

    The encoder is extrapolated from the VGG19 pre-trained on the Imagenet
    dataset and the last layer applies a softmax probability on each pixel
    of the image to accomodate a multi-class classification. NOTE: The input
    channels MUST be 3 if wanting to take advantage of the pre-training, as
    the model has been trained on 3 channels only.

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
        Function that implements the UNet architecture
        (https://arxiv.org/abs/1505.04597) with a pre-trained VGG19
        as encoder (https://arxiv.org/abs/1409.1556). The VGG19 is
        pre-trained on the Imagenet dataset
        (https://ieeexplore.ieee.org/abstract/document/5206848)
        and using a softmax probability for pixels to belong
        to one of the input n classes. The input image height and width
        needs to be equal and multiple of 16 for the architecture to build
        succesfully (e.g. 128, 256, 384, etc.). NOTE: The input channels
        MUST be 3 if wanting to take advantage of the pre-training, as
        the model has been trained on 3 channels only.

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

        # Only requesting the convolution layer and not the classifier. This is
        # because the model will use a custom classifier as last layer
        vgg19 = VGG19(include_top=False, weights='imagenet',
                      input_tensor=input_img)

        # Skip Conections
        s1 = vgg19.get_layer("block1_conv2").output
        s2 = vgg19.get_layer("block2_conv2").output
        s3 = vgg19.get_layer("block3_conv4").output
        s4 = vgg19.get_layer("block4_conv4").output

        # Bridging the encoding part to the decoding part
        b1 = vgg19.get_layer("block5_conv4").output

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
        model = Model(input_img, output_img, name='VGG19-UNet')

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

    def __decoder_block(self, input_tensor, skip_features, num_filters):
        "Function that upscales the input image size"

        x = layers.Conv2DTranspose(num_filters, (2, 2),
                                   strides=2, padding='same')(input_tensor)
        x = layers.Concatenate()([x, skip_features])
        x = self.__conv_block(x, num_filters)
        return x
