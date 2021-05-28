import pandas as pd
import numpy as np

from src.state_comparator.sensor_data_preparator import load_and_prepare_sensor_data, create_epanet_pressure_df


def compare_real_data_with_simulated(real_data_dict, simulated_df, day_which_to_compare="2021-03-15"):
    """
    TODO add documentation
    :param real_data_dict:
    :param simulated_df:
    :param day_which_to_compare:
    :return:
    """
    diff_df = pd.DataFrame()

    for sensor_key in real_data_dict:
        temp_df = real_data_dict[sensor_key]
        specific_day_df = temp_df[temp_df.index.floor('D') == day_which_to_compare]

        if len(specific_day_df.index) != 24:
            raise Exception("Not enough values for specified day ! Name: " + sensor_key)
        df_hours = specific_day_df.index.hour
        specific_day_df.index = df_hours

        # Check if arrays contain NaN values
        if np.isnan(np.sum(simulated_df[sensor_key].to_numpy())) or \
                np.isnan(np.sum(specific_day_df.to_numpy().flatten())):
            raise Exception("Data for simulated day contains NaN values, please choose another day !")
        diff_array = (simulated_df[sensor_key].to_numpy() - specific_day_df.to_numpy().flatten())

        diff_df[sensor_key] = pd.Series(diff_array, index=simulated_df[sensor_key].index)

    return diff_df


def generate_error_dataframe(difference_df):
    """
    Function creates a new dataframe and calculates errors (mean, max, min values) for every sensor/column in the
    dataframe.

    :param difference_df: Dataframe of sensors and pressure values that are numeric actually errors.
    :return: Returns a dataframe which contains mean, max, min errors for every sensor in the dataframe.
    """
    error_dataframe = pd.DataFrame(columns=["Mean-error", "Max-error", "Min-error"])

    for sensor_name in difference_df.columns:
        mean_error = difference_df[sensor_name].mean()
        max_error = difference_df[sensor_name].max()
        min_error = difference_df[sensor_name].min()

        error_dataframe.loc[sensor_name] = [mean_error, max_error, min_error]
        # TODO add hour of the day
        # TODO handle abs values

    return error_dataframe


def find_most_critical_sensor(error_dataframe, error_metric_column="Mean-error"):
    """
    Function finds the node with the highest absolute value and returns its name.

    :param error_dataframe: Dataframe with error on which the search for the critical node.
    :param error_metric_column: The column of the dataframe on which to search on.
    :return: Returns a string which contains the name of the node with the highest error/absolute value.
    """
    most_critical_node = error_dataframe[error_metric_column].abs().idxmax()
    return most_critical_node


def analyse_data_and_find_critical_sensor(path_to_data_dir, sensor_files, pump_files, epanet_file, selected_nodes, day):
    """
    TODO add documentation
    :param path_to_data_dir:
    :param sensor_files:
    :param pump_files:
    :param epanet_file:
    :param selected_nodes:
    :param day:
    :return:
    """
    real_data_dict = load_and_prepare_sensor_data(path_to_data_dir, sensor_files, pump_files)
    epanet_simulated_df = create_epanet_pressure_df(epanet_file, selected_nodes=selected_nodes)

    difference_df = compare_real_data_with_simulated(real_data_dict, epanet_simulated_df, day)
    error_df = generate_error_dataframe(difference_df)

    return find_most_critical_sensor(error_df)

