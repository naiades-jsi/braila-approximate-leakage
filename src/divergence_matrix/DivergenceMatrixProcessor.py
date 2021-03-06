from datetime import datetime, timezone

import pandas as pd
from jenkspy import jenks_breaks
from src.divergence_matrix.groups_helper_functions import optimal_number_of_groups

import src.configfile as config

from copy import deepcopy

from src.epanet.EPANETUtils import EPANETUtils


class DivergenceMatrixProcessor:
    def __init__(self, file_name):
        """
        Takes in the name of the file or to be more specific the path to it.
        """
        self.file_name = file_name
        self.array_of_divergence_dfs = pd.read_pickle(self.file_name)

    def print_divergence_matrix_description(self):
        """
        Method provides a description of the divergence matrix that the class contains.
        """
        leak_step = round(abs(float(self.array_of_divergence_dfs[0].columns.name.split(",")[1].replace("LPS", "")) - \
                              float(self.array_of_divergence_dfs[1].columns.name.split(",")[1].replace("LPS", ""))), 3)

        description_string = \
            f"""Divergence Matrix Description:
        File: {self.file_name},
        Number of leakage scenarios: {len(self.array_of_divergence_dfs)},
        Number of nodes: {len(self.array_of_divergence_dfs[1].columns)},
        Scenario simulation seconds duration: {max(self.array_of_divergence_dfs[0].index)},
        Minimum placed leak: {self.array_of_divergence_dfs[0].columns.name.split(",")[1]},
        Maximum placed leak: {self.array_of_divergence_dfs[-1].columns.name.split(",")[1]},
        Placed leak step: {leak_step}LPS,      
        """
        print(description_string)

    def retrieve_indexes_with_specified_leak(self, leak_amount):
        """
        Finds indexes of dataframes where the desired leak was simulated.
        # this can be done with mod (%) more efficiently, but only if the order is guaranteed

        :param leak_amount: Float value of the leak that we want to find in Liters per Second.
        :return: Returns an array of int indexes which correspond to the location in array_of_divergence_dfs.
        """
        if type(leak_amount) != float:
            raise Exception("Leak amount must be of type float ! For example 4.0 !")

        leak_search_string = ", {:.1f}LPS".format(leak_amount)

        index_arr = []
        for index, temp_df in enumerate(self.array_of_divergence_dfs):
            if leak_search_string in temp_df.axes[1].name:
                index_arr.append(index)

        return index_arr

    def create_array_of_dfs_with_selected_column(self, indexes_of_dfs, selected_column):
        """
        Creates an array of dataframes from the main array. New array contains only selected indexes and the dataframe
        contains only one column per node with leak.

        :param indexes_of_dfs: An array of indexes of the dataframe that we wish to keep. Preferably created with
        retrieve_indexes_with_specified_leak method.
        :param selected_column: Name of the column of which values we want to keep.
        :return: Returns an array of selected dataframes with only the selected column.
        """
        if len(indexes_of_dfs) < 1:
            raise Exception("Length of indexes array must be at least 1 !")

        if type(selected_column) != str:
            raise Exception("Selected column must a string !")
        new_array_of_dfs = []
        # to ensure that the original list is not modified
        copy_of_original_df = deepcopy(self.array_of_divergence_dfs)

        for index in indexes_of_dfs:
            temp_df = copy_of_original_df[index][[selected_column]]

            new_column_name = "{}".format(temp_df.axes[1].name)
            temp_df.columns = [new_column_name]

            new_array_of_dfs.append(temp_df)

        return new_array_of_dfs

    def extract_df_with_specific_leak_on_one_node(self, leak_amount, selected_node):
        """
        Combines methods retrieve_indexes_with_specified_leak and create_array_of_dfs_with_selected_column to generate
        a dataframe with all the important data.

        :param selected_node: Name of the column of which values we want to keep.
        :param leak_amount: Float value of the leak that we want to find in Liters per Second.
        :returns: One dataframes containing all the nodes with the simulated leak amount but just for one node. In our
        case this will be the sensors on which we want to see the impact.
        Indexes are node names with leak, columns is timestamp in seconds.
        """
        df_indexes_with_leak = self.retrieve_indexes_with_specified_leak(leak_amount)
        array_of_dfs = self.create_array_of_dfs_with_selected_column(df_indexes_with_leak, selected_node)

        concatenated_transposed_df = pd.concat(array_of_dfs, axis=1).transpose()
        return concatenated_transposed_df

    def calculate_column_correlation(self, leak_amount, selected_node):
        """
        Returns two dataframes which are correlation matrices. First takes into account the order of the elements and
        the second just checks the contents of the arrays.

        :param selected_node: Name of the column of which values we want to keep.
        :param leak_amount: Float value of the leak that we want to find in Liters per Second.
        :returns: Returns two dataframes which are correlation matrices. First takes into account the order of the
        elements and the second just checks the contents of the arrays.
        """
        df = self.extract_df_with_specific_leak_on_one_node(leak_amount, selected_node)
        column_name = "sorted_array"

        nodes_order_df = pd.DataFrame(columns=[column_name])
        for timestamp in df.columns:
            arr_of_sorted_nodes = list(df[timestamp].sort_values(ascending=False).index)[:30]   # round(len(df.index) * 0.1)
            nodes_order_df.loc[timestamp] = [arr_of_sorted_nodes]
            # TODO this correlation should be tested with newer methods

        timestamp_indexes = list(nodes_order_df.index)
        order_correlation_df = pd.DataFrame(columns=timestamp_indexes, index=timestamp_indexes)
        basic_correlation_df = pd.DataFrame(columns=timestamp_indexes, index=timestamp_indexes)

        for timestamp_1 in timestamp_indexes:
            for timestamp_2 in timestamp_indexes:
                array_1 = nodes_order_df.at[timestamp_1, column_name]
                array_2 = nodes_order_df.at[timestamp_2, column_name]

                order_correlation = 0
                basic_correlation = 0
                num_of_elements = 0
                for element_1, element_2 in zip(array_1, array_2):
                    if element_1 == element_2:
                        order_correlation += 1

                    if element_1 in array_2:
                        basic_correlation += 1

                    num_of_elements += 1

                # no need to handle division by zero because it shouldn't happen,if it does it is an error with the data
                order_correlation_res = (order_correlation / num_of_elements)
                basic_correlation_res = (basic_correlation / num_of_elements)

                order_correlation_df.at[timestamp_1, timestamp_2] = order_correlation_res
                basic_correlation_df.at[timestamp_1, timestamp_2] = basic_correlation_res

        return order_correlation_df, basic_correlation_df

    def nodes_which_effect_the_sensors_most(self, leak_amount, selected_node):
        """
        Method creates a dataframe of nodes which have the same simulated amount. And only keeps the node affected by
        it that was passed in selected node. Method returns an array of nodes that effect the selected node the most
        at time of 36000 or 10:00.

        :param leak_amount: Amount of leakage in LPS, this string will be compared to dataframe names to choose
        the right ones. Example: 16.0
        :param selected_node: The node which interest us. A single string with node name. Example: "763-B"
        :return: Returns an array of nodes which effect the selected nodes the most in terms of leakage.
        """
        time = 36000
        df = self.extract_df_with_specific_leak_on_one_node(leak_amount, selected_node)
        column_name = "sorted_array"

        nodes_order_df = pd.DataFrame(columns=[column_name])
        for timestamp in df.columns:
            arr_of_sorted_nodes = list(df[timestamp].sort_values(ascending=False).index)[:round(len(df.index) * 0.1)]
            nodes_order_df.loc[timestamp] = [arr_of_sorted_nodes]

        return nodes_order_df.loc[time].array[0], df

    def get_affected_nodes_groups(self, leak_amount, selected_node, num_of_groups=4, method="jenks_natural_breaks"):
        """
        Method creates a dataframe of nodes which have the same simulated amount. And only keeps the node affected by
        it that was passed in selected node. Method returns an array of nodes that effect the selected node the most
        at time of 36 000 of 10:00.

        :param num_of_groups:
        :param method:
        :param leak_amount: Amount of leakage in LPS, this string will be compared to dataframe names to choose
        the right ones. Example: 16.0
        :param selected_node: The node which interest us. A single string with node name. Example: "763-B"
        :return: Returns an array of nodes which effect the selected nodes the most in terms of leakage.
        """
        simulation_time_stamp = 36000
        groups_indexes = None
        series_at_timestamp = self.extract_df_with_specific_leak_on_one_node(leak_amount, selected_node)[
            simulation_time_stamp]

        # Series must be sorted for all the following steps
        sorted_df_at_timestamp = series_at_timestamp.sort_values(ascending=False).reset_index()
        series_len = len(sorted_df_at_timestamp)
        if method == "descending_values":
            groups_indexes = self.get_cutoff_indexes_by_descending_values(series_len, num_of_groups)
        elif method == "jenks_natural_breaks":
            sorted_series = sorted_df_at_timestamp[simulation_time_stamp]
            groups_indexes = self.get_cutoff_indexes_by_jenks_natural_breaks(sorted_series, num_of_groups)
        elif method == "jenks_natural_breaks+optimal_groups":
            sorted_series = sorted_df_at_timestamp[simulation_time_stamp]
            groups_indexes = self.get_cutoff_indexes_by_jenks_natural_breaks(sorted_series, None)
        elif method == "kde":
            print("Not implemented")
        else:
            print("Not implemented")

        # print(groups_indexes)
        groups_dict = self.generate_groups_dict(groups_indexes, sorted_df_at_timestamp, simulation_time_stamp,
                                                leak_amount)
        return groups_dict

    def prepare_output_json_raw(self, groups_dict):
        """
        Method prepares the output json for service in the most basic data format.
        :param groups_dict: Dictionary of groups, each group (key) contains a list of nodes.
        :return: Dictionaary of timestamp and groups with nodes.
        """
        current_timestamp = int(datetime.now().timestamp())
        output_json = {
            config.OUTPUT_JSON_TIME_KEY: current_timestamp,
            config.OUTPUT_JSON_NODES_KEY: groups_dict
        }
        return output_json

    def get_cutoff_indexes_by_descending_values(self, series_len, num_of_groups):
        group_size = round(series_len / num_of_groups)

        cutoff_indexes = []
        for multiplier in range(num_of_groups + 1):
            index_to_add = int(multiplier * group_size)

            if (series_len - index_to_add) >= group_size:
                cutoff_indexes.append(index_to_add)
            else:
                cutoff_indexes.append(series_len)
                break

        return cutoff_indexes

    def get_cutoff_indexes_by_jenks_natural_breaks(self, series, num_of_groups):
        values_array = series.to_numpy()
        if num_of_groups is None:
            num_of_groups = optimal_number_of_groups(values_array)
        group_break_values = jenks_breaks(values_array, nb_class=num_of_groups)

        cutoff_indexes = [0]
        for index in range(1, len(group_break_values)):
            filtered_series = series[series == group_break_values[index]]
            # len ensures that the last element with specific value is selected
            cutoff_indexes.append(filtered_series.index[len(filtered_series) - 1])

        # For ensuring that the last value is the actual last index
        cutoff_indexes[len(cutoff_indexes) - 1] = len(values_array)
        # to ensure order
        cutoff_indexes.sort()
        return cutoff_indexes

    def generate_groups_dict(self, cutoff_indexes, series_node_value, timestamp, leak_amount):
        """
        Method prepares the data (replaces strings) and generates groups of nodes which it stores in a dictionary.
        The dictionary is then returned and used in other applications so the format shouldn't be changed.
        Dictionary is of form:
        {
        "0-AFFECTED-GROUP": array_of_nodes,
        "1-AFFECTED-GROUP": ['760-A', '763-A', '751-A', '748-A', 'Jonctiune-J-26', ...]
        }

        :param cutoff_indexes: Array of indexes. At each index a group either starts or ends.
        :param series_node_value: Series/array from which the node names are taken.
        :param timestamp: Time of the simulation at which the data should be extracted. Usually 10 am or 36000s.
        :return: Returns the dictionary described above.
        """
        groups_dict = dict()

        for index in range(0, len(cutoff_indexes) - 1):
            group_name = "{}".format(index)
            replace_str = ", {:.1f}LPS".format(leak_amount)

            nodes_list = list(series_node_value[cutoff_indexes[index]:cutoff_indexes[index + 1]]
                              .set_index("index")[timestamp].index)
            nodes_list = [node_name.replace(replace_str, "") for node_name in nodes_list]
            nodes_list = [node_name.replace("Node_", "") for node_name in nodes_list]
            groups_dict[group_name] = nodes_list

        return groups_dict
