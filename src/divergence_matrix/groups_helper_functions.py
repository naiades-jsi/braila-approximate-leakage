import numpy as np
from jenkspy import jenks_breaks
# TODO document, add groups (strings) scaling for unlimited number of groups


def optimal_number_of_groups(array_of_values):
    smallest_change = 0.03

    fit_value = 0.0
    group_value = 2
    while True:
        previous_fit_value = fit_value
        fit_value = goodness_of_variance_fit(array_of_values, group_value)
        print("classes ", group_value, "gvf", fit_value)

        if abs(fit_value - previous_fit_value) < smallest_change:
            # exit if fit changes for less than 3 per cent
            break

        group_value += 1

    return group_value


def goodness_of_variance_fit(array, classes):
    # TODO document this is from stackoverflow
    # https://stats.stackexchange.com/questions/143974/jenks-natural-breaks-in-python-how-to-find-the-optimum-number-of-breaks
    classes = jenks_breaks(array, classes)
    # do the actual classification
    classified = np.array([classify(i, classes) for i in array])
    # max value of zones
    maxz = max(classified)

    # nested list of zone indices
    zone_indices = [[idx for idx, val in enumerate(classified) if zone + 1 == val] for zone in range(maxz)]

    # sum of squared deviations from array mean
    sdam = np.sum((array - array.mean()) ** 2)

    # sorted polygon stats
    array_sort = [np.array([array[index] for index in zone]) for zone in zone_indices]

    # sum of squared deviations of class means
    sdcm = sum([np.sum((classified - classified.mean()) ** 2) for classified in array_sort])

    # goodness of variance fit
    gvf = (sdam - sdcm) / sdam

    return gvf


def classify(value, breaks):
    for i in range(1, len(breaks)):
        if value < breaks[i]:
            return i
    return len(breaks) - 1