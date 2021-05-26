import pandas as pd
import numpy as np

from src.state_comparator.sensor_data_preparator import load_and_prepare_sensor_data, create_epanet_pressure_df


def compare_real_data_with_simulated(real_data_dict, simulated_df, day_which_to_compare="2021-03-15"):
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
    error_dataframe = pd.DataFrame(columns=["Mean-error", "Max-error", "Min-error"])

    for sensor_name in difference_df.columns:
        mean_error = difference_df[sensor_name].mean()
        max_error = difference_df[sensor_name].max()
        min_error = difference_df[sensor_name].min()

        error_dataframe.loc[sensor_name] = [mean_error, max_error, min_error]
        # TODO add hour of the day
        # TODO handle abs values

    return error_dataframe


def find_most_critical_sensor(error_dataframe, error_metric="Mean-error"):
    most_critical_val = error_dataframe[error_metric].abs().idxmax()
    return most_critical_val


def analyse_data_and_find_critical_sensor(path_to_data_dir, sensor_files, pump_files, epanet_file, selected_nodes, day):
    real_data_dict = load_and_prepare_sensor_data(path_to_data_dir, sensor_files, pump_files)
    epanet_simulated_df = create_epanet_pressure_df(epanet_file, selected_nodes=selected_nodes)

    difference_df = compare_real_data_with_simulated(real_data_dict, epanet_simulated_df, day)
    error_df = generate_error_dataframe(difference_df)

    return find_most_critical_sensor(error_df)

