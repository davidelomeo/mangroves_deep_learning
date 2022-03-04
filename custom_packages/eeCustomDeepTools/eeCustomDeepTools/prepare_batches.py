#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script contains a class that prepares the input Tensorflow Records
# to be fed into Keras deep models.
#
# The TFRecords are first parsed using a dictionary of Features for each
# of the bands that need to be inlcluded in the model (the dictionary
# needs to be prepared beforehand - Look at the script in
# fixed_length_features.py for details). Once parsed, the classification
# band is separated from the rest of the bands for each of the tensors and
# hot encoded (this is beause keras expects a one-hot-encoded tensor when
# dealing with multi-class, pixel-wise, classification).
#
# Author: Davide Lomeo
# Email: davide.lomeo20@imperial.ac.uk
# GitHub: https://github.com/acse-2020/acse2020-acse9-finalreport-acse-dl1420-3
# Date: 22 July 2021
# Version: 0.1.0

import tensorflow as tf
__all__ = ['PrepareBatches']


class PrepareBatches:
    """
    Class that prepares the input datasets for training in Keras deep models.
    The class needs a dictionary with the features, the number of classes and
    the name of the band assigned to the labels. When the prepare_batches()
    function is called, it outputs tensorflow batch datasets.

    Parameters
    ----------
    features_dict : dict
        dictionary containing Fixed Lenght Features
    n_classes : int
        number of classes to output in the last layer of the deep model used
    class_label : str
        name of the label assigbed to the classification column (array)

    Functions
    ---------
    prepare_batches(train_batch_size, test_batch_size, train_batch,
                        test_batch, val_batch=None, val_batch_size=None)
        convert the input datases into tensorflow batches ready for training
    """

    def __init__(self, features_dict, n_classes, class_label):
        "Class constructor"

        super().__init__()
        self.features_dict = features_dict
        self.n_classes = n_classes
        self.class_label = class_label

    def prepare_batches(self,  train_batch_size, test_batch_size, train_batch,
                        test_batch, val_batch=None, val_batch_size=None):
        """
        Function that maps each TFRecord in the input datasets assigning them
        all the features in the features dictionary (i.e., expanding their
        dimensions, or channels, for how many bands have been inlcuded in the
        dictionary). Moreover, the feature containing the labels is separated
        and converted to a one-hot tensor, as needed by the keras deep models
        when feeding in multi-class pixel-wise classes. The records are then
        shuffled and split into batches as defined by the user. If the user
        passes a validation dataset to the function but does not specifcy its
        batch size, this will be assumed to be equal to the test batch size.
        NOTE: The function requires at least 2 datasets, and optionally a third
        if validation is used. The datasets need to have been pre-processed and
        converted in tf.data.TFRecordDataset. In the case of a split, the
        tensorflow datatype name may differ (i.e., TakeDataset), but the
        underlying structure is equivalent.

        Parameters
        ----------
        train_batch_size : int
            size of the batches of the training dataset
        test_batch_size, : int
            size of the batches of the test dataset
        train_batch : tensorflow dataset
            dataset of the training data
        test_batch : tensorflow dataset
            dataset of the test data
        val_batch : tensorflow dataset, optional
            dataset of the test data (default None)
        val_batch_size : int, optional
            size of the batches of the validation dataset (default None)

        Returns
        -------
        training BatchDataset, test BatchDataset, (optional valid BatchDataset)
            the BatchDatasets ready to be fed into deep models
        """

        # Mapping training and test datasets
        parsed_train = train_batch \
            .map(self.__parse_tfrecord, num_parallel_calls=5) \
            .map(self.__to_tuple) \
            .shuffle(10) \
            .batch(train_batch_size)

        parsed_test = test_batch \
            .map(self.__parse_tfrecord, num_parallel_calls=5) \
            .map(self.__to_tuple) \
            .shuffle(10) \
            .batch(test_batch_size)

        # Mapping the validation datasets
        if val_batch:
            parsed_valid = val_batch \
                .map(self.__parse_tfrecord, num_parallel_calls=5) \
                .map(self.__to_tuple) \
                .shuffle(10)

            # Checking if the user has provided a size for the valid dataset.
            # If not, this is assumed to be the same as the test dataset
            if val_batch_size:
                parsed_valid = parsed_valid.batch(val_batch_size)
            else:
                parsed_valid = parsed_valid.batch(test_batch_size)

            return parsed_train, parsed_test, parsed_valid

        return parsed_train, parsed_test

    def __parse_tfrecord(self, example_proto):
        """
        Parsing function that maps each record into the structure defined
        by the dictionary of features.

        Args
        ----
        example_proto
            the input tensorflow record

        Returns
        -------
        tuple
            A tuple of the predictors dictionary and the label in int64 format.
        """

        # parsing the input record to the feature dictionary
        parsed_features = tf.io.parse_single_example(
            example_proto, self.features_dict)

        # pulling the feature of the label
        labels = parsed_features.pop(self.class_label)

        # returning the parsed record and it corresponding labels as a tuple
        return parsed_features, tf.cast(labels, tf.int64)

    def __to_tuple(self, inputs, label):
        """
        Function that maps each feature and split it into a tensor of the
        expected patch size and expected number of channels (bands) and a
        one-hot tensor containing the labels for each of the features.
        These are retuend as a tuple. This process is key if performing
        multi-class, pixel-wise classification with Keras.

        Args
        ----
        inputs
            the input record features
        label
            the input label feature

        Returns
        -------
        tuple
            A tuple of the converted feature and label tensors.
        """
        return (tf.transpose(list(inputs.values())),
                tf.one_hot(indices=label, depth=self.n_classes))
