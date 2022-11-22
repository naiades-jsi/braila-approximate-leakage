import numpy as np
import pandas as pd
from jenkspy import jenks_breaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


def prepare_training_and_test_data(df):
    """
    TODO add documentation
    :param df:
    :return:
    """
    keep_cols = ["Sensor1", "Sensor2", "Sensor3", "Sensor4", "J-Apollo", "J-RN2", "J-RN1",
                 "encoded_node_with_leak"]

    org_df = pd.read_csv("only_relavant_data.csv", index_col=0)
    # getting unique sensor names
    node_count = len(org_df["node_with_leak"].unique())

    # encoding node names to numbers
    ord_enc = OrdinalEncoder()
    org_df["encoded_node_with_leak"] = ord_enc.fit_transform(org_df[["node_with_leak"]])

    # Prefilter
    df_filtered = org_df[keep_cols]
    # df_filtered = org_df[org_df["leak_amount"] == "0.11LPS"][keep_cols]

    # TODO find better split, this is far from optimal since data is dependent of leakages amount
    # Splitting into learn, test and validation
    train_set, test_set = train_test_split(df_filtered, test_size=0.2)
    return train_set, test_set, node_count


def get_cutoff_indexes_by_jenks_natural_breaks(values_array, num_of_groups):
    """
    TODO add documentation - similar to
    :param values_array:
    :param num_of_groups:
    :return:
    """
    if num_of_groups is None:
        num_of_groups = 4   # optimal_number_of_groups(values_array)
    group_break_values = jenks_breaks(values_array, nb_class=num_of_groups)

    cutoff_indexes = []
    cutoff_count = 0
    for index, value in enumerate(np.flip(values_array)):
        if cutoff_count < len(group_break_values) and value == group_break_values[cutoff_count]:
            cutoff_indexes.append(index)
            cutoff_count += 1
    # Since the right range is not inclusive example: [0, 22)
    cutoff_indexes[len(cutoff_indexes) - 1] = cutoff_indexes[len(cutoff_indexes) - 1] + 1

    # to ensure order
    cutoff_indexes.sort()
    return cutoff_indexes


def generate_groups_dict(cutoff_indexes, sorted_prob_node_id_tup_vec, node_id_to_node_name_map):
    """
    TODO add documentation
    :param cutoff_indexes:
    :param series_node_value:
    :param map_dictionary:
    :return:
    """
    groups_dict = dict()
    # TODO check, no need for replacing "Node_" everytime could be done statically in dict
    for index in range(0, len(cutoff_indexes) - 1):
        group_name = "{}".format(index)
        curr_prob_node_id_tup_vec = sorted_prob_node_id_tup_vec[cutoff_indexes[index]:cutoff_indexes[index + 1]]

        node_name_vec = [node_id_to_node_name_map[node_id] for _, node_id in curr_prob_node_id_tup_vec]
        node_name_vec = [node_name.replace("Node_", "") for node_name in node_name_vec]

        groups_dict[group_name] = node_name_vec

    return groups_dict


def gen_groups_dict_from_data_loader(data_loader, cutoff_idxs, sorted_prob_node_id_tup_vec):
    groups_dict = dict()
    # TODO check, no need for replacing "Node_" everytime could be done statically in dict
    for index in range(0, len(cutoff_idxs) - 1):
        group_name = "{}".format(index)
        curr_prob_node_id_tup_vec = sorted_prob_node_id_tup_vec[cutoff_idxs[index]:cutoff_idxs[index + 1]]

        node_name_vec = [data_loader.decode_node_label([node_id])[0] for _, node_id in curr_prob_node_id_tup_vec]
        node_name_vec = [node_name.replace("Node_", "") for node_name in node_name_vec]

        groups_dict[group_name] = node_name_vec

    return groups_dict
