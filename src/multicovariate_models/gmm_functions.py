import pickle
from operator import itemgetter

import numpy as np
from sklearn.mixture import GaussianMixture

from src.multicovariate_models.general_functions import prepare_training_and_test_data, \
    get_cutoff_indexes_by_jenks_natural_breaks, generate_groups_dict


def fit_gaussian_mixture_model(df, save_model=False, save_model_path="gmm_train_model.pkl"):
    """
    TODO add documentation
    :param df:
    :param save_model:
    :param save_model_path:
    :return:
    """
    # TODO optimize and find optimal parameters for GMMM
    x_cols = ["Sensor1", "Sensor2", "Sensor3", "Sensor4", "J-Apollo", "J-RN2", "J-RN1"]
    # Get train and test data
    train_set, test_set, node_count = prepare_training_and_test_data(df)

    x_data = train_set[x_cols].values
    y_data = train_set["encoded_node_with_leak"].values
    print(x_data[:2], y_data[:2])

    gm_model = GaussianMixture(n_components=node_count, random_state=0, n_init=10, init_params="kmeans").fit(x_data)

    # If you wish to save the model provide the save_model flag
    if save_model:
        with open(save_model_path, "wb") as f:
            pickle.dump(gm_model, f)

    return gm_model


def predict_groups_gmm(model, data_arr):
    """
    TODO add documentation
    :param model:
    :param data_arr:
    :return:
    """
    # TODO dictionary should be moved to conf or at least made the same for all models
    static_dict = {0.0: 'Node_-', 1.0: 'Node_J-Apollo', 2.0: 'Node_J-RN1', 3.0: 'Node_J-RN2', 4.0: 'Node_Jonctiune-1225',
                   5.0: 'Node_Jonctiune-1226', 6.0: 'Node_Jonctiune-12372', 7.0: 'Node_Jonctiune-12588', 8.0:
                       'Node_Jonctiune-1405', 9.0: 'Node_Jonctiune-1406', 10.0: 'Node_Jonctiune-1407', 11.0:
                       'Node_Jonctiune-1413', 12.0: 'Node_Jonctiune-1414', 13.0: 'Node_Jonctiune-1415', 14.0:
                       'Node_Jonctiune-1419', 15.0: 'Node_Jonctiune-1421', 16.0: 'Node_Jonctiune-1610', 17.0:
                       'Node_Jonctiune-1635', 18.0: 'Node_Jonctiune-1636', 19.0: 'Node_Jonctiune-1638', 20.0:
                       'Node_Jonctiune-1641', 21.0: 'Node_Jonctiune-1642', 22.0: 'Node_Jonctiune-1872', 23.0:
                       'Node_Jonctiune-1874', 24.0: 'Node_Jonctiune-1875', 25.0: 'Node_Jonctiune-1877', 26.0:
                       'Node_Jonctiune-1995', 27.0: 'Node_Jonctiune-1996', 28.0: 'Node_Jonctiune-1997', 29.0:
                       'Node_Jonctiune-1998', 30.0: 'Node_Jonctiune-2176', 31.0: 'Node_Jonctiune-2177', 32.0:
                       'Node_Jonctiune-2180', 33.0: 'Node_Jonctiune-2181', 34.0: 'Node_Jonctiune-2184', 35.0:
                       'Node_Jonctiune-2185', 36.0: 'Node_Jonctiune-2186', 37.0: 'Node_Jonctiune-2187', 38.0:
                       'Node_Jonctiune-2188', 39.0: 'Node_Jonctiune-2189', 40.0: 'Node_Jonctiune-2190', 41.0:
                       'Node_Jonctiune-2191', 42.0: 'Node_Jonctiune-2192', 43.0: 'Node_Jonctiune-2193', 44.0:
                       'Node_Jonctiune-2194', 45.0: 'Node_Jonctiune-2195', 46.0: 'Node_Jonctiune-2196', 47.0:
                       'Node_Jonctiune-2197', 48.0: 'Node_Jonctiune-2200', 49.0: 'Node_Jonctiune-2202', 50.0:
                       'Node_Jonctiune-2203', 51.0: 'Node_Jonctiune-2204', 52.0: 'Node_Jonctiune-2206', 53.0:
                       'Node_Jonctiune-2207', 54.0: 'Node_Jonctiune-2208', 55.0: 'Node_Jonctiune-267', 56.0:
                       'Node_Jonctiune-2729', 57.0: 'Node_Jonctiune-2734', 58.0: 'Node_Jonctiune-2736', 59.0:
                       'Node_Jonctiune-2738', 60.0: 'Node_Jonctiune-2739', 61.0: 'Node_Jonctiune-2743', 62.0: 'Node_Jonctiune-2750', 63.0: 'Node_Jonctiune-2751', 64.0: 'Node_Jonctiune-2752', 65.0: 'Node_Jonctiune-2753', 66.0: 'Node_Jonctiune-2755', 67.0: 'Node_Jonctiune-2756', 68.0: 'Node_Jonctiune-2774', 69.0: 'Node_Jonctiune-2776', 70.0: 'Node_Jonctiune-2777', 71.0: 'Node_Jonctiune-2879', 72.0: 'Node_Jonctiune-2968', 73.0: 'Node_Jonctiune-3067', 74.0: 'Node_Jonctiune-3068', 75.0: 'Node_Jonctiune-3074', 76.0: 'Node_Jonctiune-3075', 77.0: 'Node_Jonctiune-3386', 78.0: 'Node_Jonctiune-3422', 79.0: 'Node_Jonctiune-3425', 80.0: 'Node_Jonctiune-3446', 81.0: 'Node_Jonctiune-3448', 82.0: 'Node_Jonctiune-3464', 83.0: 'Node_Jonctiune-3466', 84.0: 'Node_Jonctiune-3467', 85.0: 'Node_Jonctiune-3470', 86.0: 'Node_Jonctiune-3471', 87.0: 'Node_Jonctiune-3510', 88.0: 'Node_Jonctiune-3566', 89.0: 'Node_Jonctiune-3913', 90.0: 'Node_Jonctiune-3917', 91.0: 'Node_Jonctiune-3920', 92.0: 'Node_Jonctiune-3956', 93.0: 'Node_Jonctiune-3961', 94.0: 'Node_Jonctiune-3967', 95.0: 'Node_Jonctiune-3972', 96.0: 'Node_Jonctiune-4595', 97.0: 'Node_Jonctiune-4602', 98.0: 'Node_Jonctiune-4615', 99.0: 'Node_Jonctiune-4618', 100.0: 'Node_Jonctiune-4619', 101.0: 'Node_Jonctiune-4723', 102.0: 'Node_Jonctiune-4731', 103.0: 'Node_Jonctiune-4742', 104.0: 'Node_Jonctiune-4743', 105.0: 'Node_Jonctiune-J-1', 106.0: 'Node_Jonctiune-J-15', 107.0: 'Node_Jonctiune-J-16', 108.0: 'Node_Jonctiune-J-19', 109.0: 'Node_Jonctiune-J-20', 110.0: 'Node_Jonctiune-J-21', 111.0: 'Node_Jonctiune-J-23', 112.0: 'Node_Jonctiune-J-25', 113.0: 'Node_Jonctiune-J-26', 114.0: 'Node_Jonctiune-J-27', 115.0: 'Node_Jonctiune-J-3', 116.0: 'Node_Jonctiune-J-31', 117.0: 'Node_Jonctiune-J-32', 118.0: 'Node_Jonctiune-J-34', 119.0: 'Node_PT1', 120.0: 'Node_PT2', 121.0: 'Node_PT3', 122.0: 'Node_PT4', 123.0: 'Node_Sensor1', 124.0: 'Node_Sensor2', 125.0: 'Node_Sensor3', 126.0: 'Node_Sensor4'}
    x = data_arr
    y_predicted = model.predict_proba(x)[0]
    # print(y_predicted, len(y_predicted))

    # create new array with value and its index
    value_group_index_arr = [[value, index] for index, value in enumerate(y_predicted)]

    # sort by values
    sorted_val_index_arr = sorted(value_group_index_arr, key=itemgetter(0), reverse=True)
    values_only_arr = np.array([value for value, index in sorted_val_index_arr])

    # split into groups
    groups_indexes = get_cutoff_indexes_by_jenks_natural_breaks(values_only_arr, None)
    return generate_groups_dict(groups_indexes, sorted_val_index_arr, static_dict)

def predict_knn(model, ftr_vec):
    y_hat = model.predict_proba(ftr_vec)