from datetime import datetime, timedelta
import math
import src.configfile as conf
import pandas as pd
import numpy as np

from src.state_comparator.NaNSensorsException import NaNSensorsException
from src.state_comparator.sensor_data_preparator import load_and_prepare_sensor_data, create_epanet_pressure_df


def compare_real_data_with_simulated(real_data_dict, simulated_df, day_which_to_compare="2021-03-15"):
    """
    Compares the real data with
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

    return error_dataframe


def missing_values_check(df, minimum_present_values):
    """
    Function checks if there are any missing values in the dataframe.

    :param df: Dataframe which is checked for missing values.
    :param minimum_present_values: Int, minimum number of values that should be present in each column of the real
    dataframe.
    :return: Returns None if all sensors are OK else return all of the sensors where conditions are not met.
    """
    # TODO extra error handling can be added here
    sensors_with_missing_values = []
    for column in df.columns:
        not_nan_values = len(df[column].dropna(inplace=False))

        if not_nan_values <= minimum_present_values:
            sensors_with_missing_values.append(column)

    if len(sensors_with_missing_values) < 1:
        # If all sensors are OK return None
        return None
    else:
        raise NaNSensorsException(sensors_with_missing_values)


def find_most_critical_sensor(error_dataframe, error_metric_column="Mean-error"):
    """
    Function finds the node with the highest absolute value (depending on the error_metric_column parameter)
    and returns it along with its name.

    :param error_dataframe: Dataframe with error on which the search for the critical node.
    :param error_metric_column: The column of the dataframe on which to search on.
    :return: Returns a string which contains the name of the node with the highest error/absolute value. It also returns
    that absolute value belonging to the node.
    """
    deviation_value = error_dataframe[error_metric_column].abs().max()
    most_critical_node = error_dataframe[error_metric_column].abs().idxmax()
    return most_critical_node, deviation_value


def analyse_data_and_find_critical_sensor(path_to_data_dir, sensor_files, pump_files, epanet_file, selected_nodes, date):
    """
    Reads the actual data from dump files and compares it to the simulated pressure values. It then returns the
    sensor/node which has the biggest mean error and the actual error.

    :param path_to_data_dir: Path to the directory where the files are stored.
    :param sensor_files: Array of strings containing data for the 4 normal sensors.
    :param pump_files: Array of strings containing data for the 4 debitmeters/pumps.
    :param epanet_file: Path to the epanet file.
    :param selected_nodes: The nodes which we want to compare preferably the 8 nodes for which we have measurements.
    :param date: The date for which we want to compare real values with the simulated ones.
    :return: Returns the node with the highest mean error and the actual mean error value
    """
    real_data_dict = load_and_prepare_sensor_data(path_to_data_dir, sensor_files, pump_files)
    epanet_simulated_df = create_epanet_pressure_df(epanet_file, selected_nodes=selected_nodes)

    difference_df = compare_real_data_with_simulated(real_data_dict, epanet_simulated_df, date)
    error_df = generate_error_dataframe(difference_df)
    critical_node, deviation = find_most_critical_sensor(error_df)

    return critical_node, deviation


def analyse_kafka_topic_and_find_critical_sensor(timestamp, kafka_array, epanet_simulated_df, sensor_names,
                                                 minimum_present_values=conf.MINIMUM_PRESENT_VALUES_THRESHOLD):
    """
    Combines multiple functions to find the sensor which deviates the most and returns it along with the deviated value.
    First is converts the kafka array to a DataFrame, then it calculates the difference between the epanet simulated
    df and the actual values df. By looking at these difference the node with the highest mean error is returned.

    :param timestamp: Epoch timestamp. Used for calculating hour of the day.
    :param kafka_array: Array of floats. Values represent pressure values.
    :param epanet_simulated_df: Dataframe which was produced with EPANET and servers as a benchmark.
    :param sensor_names: Names of the sensors which will be compared.
    :param minimum_present_values: Int, minimum number of values that should be present in each column of the real
    dataframe.
    :return: Returns the most critical sensor and the error (deviation) value.
    """
    actual_values_df = create_df_from_real_values(kafka_array, timestamp, sensor_names)
    missing_values_check(actual_values_df, minimum_present_values)

    difference_df = epanet_simulated_df.sub(actual_values_df)
    error_df = generate_error_dataframe(difference_df)

    critical_node, deviation = find_most_critical_sensor(error_df, error_metric_column="Mean-error")
    return critical_node, deviation


def create_df_from_real_values(measurements_arr, epoch_timestamp, sensor_names):
    """
    Creates a Dataframe from array.

    :param measurements_arr: Array of floats. Values represent pressure values.
    :param epoch_timestamp: Epoch timestamp In milliseconds. Used for calculating hour of the day.
    :param sensor_names: Names of the sensors which will be compared.
    :return: Returns a dataframe containing the actual values mapped to sensors and hours.
    """
    hours_in_a_day = 24
    num_of_sensors = len(sensor_names)

    # Comparision if the timestamp is in milliseconds or seconds
    timestamp_digits = len(str(epoch_timestamp))
    if timestamp_digits == 10:
        epoch_seconds = epoch_timestamp
    elif timestamp_digits == 13:
        epoch_seconds = epoch_timestamp / 1000
    else:
        raise Exception("Timestamp is not in Unix milliseconds or seconds !")

    dt_time = datetime.fromtimestamp(epoch_seconds)
    actual_values_df = pd.DataFrame(columns=sensor_names,
                                    index=[hour_of_day for hour_of_day in range(0, hours_in_a_day)])

    for sensor_index in range(0, num_of_sensors):
        for hour_index in range(0, hours_in_a_day):
            hour = (hour_index % hours_in_a_day)
            current_time = dt_time - timedelta(hours=hour, minutes=0)

            actual_values_df.at[current_time.hour, sensor_names[sensor_index]] = measurements_arr[(sensor_index * 24) + hour]

    return actual_values_df

