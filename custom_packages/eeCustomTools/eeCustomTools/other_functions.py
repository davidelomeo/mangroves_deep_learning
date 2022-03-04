#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script contains functions that don't belong to a specific category.
# The buffer_size function computes a buffer around the input ee.Element
# whilst the get_metrics function returns the accuracy metrics of the input
# classifier and test set.
#
# Author: Davide Lomeo,
# Email: davide.lomeo20@imperial.ac.uk
# GitHub: https://github.com/acse-2020/acse2020-acse9-finalreport-acse-dl1420-3
# Date: 19 July 2021
# Version: 0.1.0

__all__ = ['buffer_size', 'get_metrics']


def buffer_size(size):
    """
    Function that uses the concept of currying to help the .map()
    method take more than one argument. The function uses the input
    'size' and call a nested function to create a buffer around the
    centroid of the feature parsed by the .map() method.The geometry
    of the parsed feature is modified in place.

    Parameters
    ----------
    size : int
        Distance of the buffering in meters

    Returns
     -------
     ee.element.Element
         Feature with a buffer of 'size' around its centroid
    """

    def create_buffer(feature):
        """Child function that creates the buffer of 'size' meters"""
        return feature.buffer(size)

    return create_buffer


def get_metrics(classifier, test_dataset, class_columns_name,
                metrics=['error_matrix'], decimal=4):
    """
    Function that requires a trained classifier and a test set to return
    a dictionary that contains the chosen metrics. metrics available are:
    'train_accuracy', 'test_accuracy', 'kappa_coefficient',
    'producers_accuracy', 'consumers_accuracy', 'error_matrix'.
    NOTE: use this function carefully, especially if wanting to return more
    than one of the available metrics. This is because the values are retrieved
    directly from the Google sevrer using the method .getInfo() and it may take
    up to 15 minutes to get all the metrics.

    Parameters
    ----------
    classifier : ee.classifier.Classifier
        Trained classifier needed to evaluate the test set
    test_dataset : ee.featurecollection.FeatureCollection
        Feature Collction that needed to evaluate the classifier
    class_column_name : string
        Name of the classification column in the input Collection
    metrics : list, optional
        List of metrics to return as a dictionary
    decimal : int, optional
        Number of decimals for each figure

    Returns
    -------
    dictionary
        A dictionary containing the selected metrics
    """

    try:
        # Clssification of the test dataset
        test = test_dataset.classify(classifier)

        # Accuracy of the training dataset classifier
        training_accuracy = classifier.confusionMatrix().accuracy()

        # Error matrix of the test set
        error_matrix = test.errorMatrix(class_columns_name, 'classification')

        # Computing consumers and producers accuracies
        producers_accuracy = error_matrix.producersAccuracy()
        consumers_accuracy = error_matrix.consumersAccuracy()

    except AttributeError:
        print("""
        Error: one of the inputs is invalid. Please only input a ee.Classifier,
        a ee.FeatureCollection and a string respectively""")
        return None

    metrics_dict = {}

    for i in metrics:
        if i == 'train_accuracy':
            metrics_dict[i] = round(training_accuracy.getInfo(), decimal)
        elif i == 'test_accuracy:':
            metrics_dict[i] = round(error_matrix.accuracy().getInfo(), decimal)
        elif i == 'kappa_coefficient:':
            metrics_dict[i] = round(error_matrix.kappa().getInfo(), decimal)
        elif i == 'producers_accuracy':
            metrics_dict[i] = producers_accuracy.getInfo()
            for i, j in enumerate(metrics_dict['producers_accuracy'][0]):
                metrics_dict['producers_accuracy'][0][i] = round(j, decimal)
        elif i == 'consumers_accuracy':
            metrics_dict[i] = consumers_accuracy.getInfo()
            for i, j in enumerate(metrics_dict['consumers_accuracy'][0]):
                metrics_dict['consumers_accuracy'][0][i] = round(j, decimal)
        elif i == 'error_matrix':
            metrics_dict[i] = error_matrix.getInfo()
        else:
            print(
                "'{}' isn't a valid metric. Please correct the input".format(i)
            )
            return None

    return metrics_dict
