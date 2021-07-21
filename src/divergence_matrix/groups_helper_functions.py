import warnings

import numpy as np
from jenkspy import jenks_breaks


def optimal_number_of_groups(array_of_values):
    """
    Finds the optimal number of groups depending on the goodness of variance fit. Group value is returned when the
    difference between the current goodness of variance and previous one is less then the smallest_change and the
    goodness variance is bigger than 0.8.

    :param array_of_values: Array of float values.
    :return: Returns the optimal number of groups.
    """
    smallest_change = 0.03

    fit_value = 0.0
    group_value = 2
    while True:
        previous_fit_value = fit_value
        fit_value = goodness_of_variance_fit(array_of_values, group_value)
        # print("current group num ", group_value, "gvf", fit_value)

        if abs(fit_value - previous_fit_value) < smallest_change and fit_value > 0.8:
            break # exit if fit changes for less than 3 per cent
        elif group_value > 100:
            raise Exception("Values didn't converge when determining optimal group number !")
        group_value += 1

    return group_value


def goodness_of_variance_fit(array, classes):
    """
    The Goodness of Variance Fit (GVF) is found by taking the difference between the squared deviations from the
    array mean (SDAM) and the squared deviations from the class means (SDCM), and dividing by the SDAM.

    This is an by hand implementation based on internet resources since no libraries were available.
    https://stats.stackexchange.com/questions/143974/jenks-natural-breaks-in-python-how-to-find-the-optimum-number-of-breaks

    :param array: Array of float values.
    :param classes: Int, proposed number of groups.
    :return: Returns the goodness of variance fit (a value between 0 and 1).
    """
    classes_limits = jenks_breaks(array, classes)
    # do the actual classification
    classified = np.array([classify(i, classes_limits) for i in array])
    # max value of zones
    maxz = max(classified)

    # nested list of zone indices
    zone_indices = [[idx for idx, val in enumerate(classified) if zone + 1 == val] for zone in range(maxz)]
    # sum of squared deviations from array mean
    sdam = np.sum((array - array.mean()) ** 2)
    # sorted polygon stats
    array_sort = [np.array([array[index] for index in zone]) for zone in zone_indices]

    # catching runtime warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # sum of squared deviations of class means
        sdcm = sum([np.sum((classified - classified.mean()) ** 2) for classified in array_sort])
    # goodness of variance fit
    gvf = (sdam - sdcm) / sdam

    return gvf


def classify(value, breaks):
    """
    Returns the index of an array where one boundary of the group is located.
    :param value: Value at which the group starts/ends.
    :param breaks: Array of values.
    """
    for i in range(1, len(breaks)):
        if value < breaks[i]:
            return i
    return len(breaks) - 1