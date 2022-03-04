#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script allows to split an input tf.data.TFRecords dataset
# to training and test dataset using the input split proportions.
# Optionally, the user can request to get a validation dataset
# by passing a float value to the variable `valid_chunk`.
#
# Author: Davide Lomeo
# Email: davide.lomeo20@imperial.ac.uk
# GitHub: https://github.com/acse-2020/acse2020-acse9-finalreport-acse-dl1420-3
# Date: 22 July 2021
# Version: 0.1.0

__all__ = ['dataset_split']


def dataset_split(dataset, tot_patches, training_chunk,
                  test_chunk, valid_chunk=None):
    """
    Function that splits the input data into training and test
    datasets. The function expects at least two floating points
    between 0.0 and 1.0 to define the proportions of the split.
    Additionally, the user has the option to add a third float
    if wanting to obtain a validation dataset.

    Parameters
    ----------
    dataset : tf.data.TFRecordDataset
        The TFRecords loaded as a tensorflow dataset
    tot_patches : int
        The number of records in the input dataset
    training_chunk : float
        The proportion of records to use as training dataset
    test_chunk : float
        The proportion of records to use as test dataset
    valid_chunk : float, optional
        The proportion of records to use as validation dataset

    Returns
    -------
    training_dataset, test dataset, (optional, validation dataset)
        The datasets obtained with the split (speficically in this order)
    """

    # Series of check to ensure that all the parameters are valid
    error_messages = []
    if (tot_patches < 0) | (not isinstance(tot_patches, int)):
        error_messages.append(
            'ERROR: the number of patches needs to be a positive number')
    elif (training_chunk < 0.0) | (training_chunk > 1.0) | \
         (test_chunk < 0.0) | (test_chunk > 1.0):
        error_messages.append(
            'ERROR: the proportions need to be between 0.0 and 1.0')
    elif (training_chunk + test_chunk != 1.0) & (not valid_chunk):
        error_messages.append(
            'ERROR: the proportions need to add up to exactly 1.0')
    if valid_chunk:
        if (valid_chunk < 0.0) | (valid_chunk > 1.0):
            error_messages.append(
                'ERROR: the proportions need to be between 0.0 and 1.0')
        elif (training_chunk + test_chunk + valid_chunk) != 1.0:
            error_messages.append(
                'ERROR: the proportions need to add up to exactly 1.0')
    if error_messages != []:
        print(*error_messages, sep='\n')
        if valid_chunk:
            return None, None, None
        else:
            return None, None

    # shuffling the dataset to avoid biases
    full_dataset = dataset.shuffle(10)

    # Defining the size of both training and test datasets
    train_size = int(training_chunk * tot_patches)
    test_size = int(test_chunk * tot_patches)

    # Checking if a number for the validation split has been
    # provided and define its size
    if valid_chunk:
        valid_size = int(valid_chunk * tot_patches)

    # Assigning TFRecords to training and test datasets
    training_ds = full_dataset.take(train_size)
    test_ds = full_dataset.skip(train_size)

    if (test_chunk >= training_chunk):
        print('''WARNING: the script executed succesfully, but note that the
        test set was set to be larger than the training dataset.\n''')

    # Assigning TFRecords to validation dataset
    if valid_chunk:
        validation_ds = test_ds.skip(valid_size)
        test_ds = test_ds.take(test_size)

        if (valid_chunk >= training_chunk):
            print('''WARNING: the script executed succesfully, but note that the
            valid. set was set to be larger than the training dataset.\n''')

        print('Training size: {}\nTest size: {}\nValidation size: {}'. format(
          train_size, test_size, valid_size))

        return training_ds, test_ds, validation_ds

    print('Training size: {}\nTest size: {}'. format(
          train_size, test_size))

    return training_ds, test_ds
