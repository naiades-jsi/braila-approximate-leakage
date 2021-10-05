import pandas as pd

import src.configfile as config
from src.state_comparator.sensor_data_preparator import rename_dict_keys_and_merge


def parse_data_and_save_to_csv(path_to_dir, sensor_files, pump_files, sampling_interval):
    # Reads for all the 8 sensors and prepares it in a standard form for comparison
    if len(sensor_files) < 1 and len(pump_files) < 1:
        raise Exception("Files names list must contain at least one string to file !!")

    sensor_time_column_name = "time"
    pumps_time_column_name = "time"
    pressure_column_name = "pressure_value(m)"
    sensors_dict = dict()
    pumps_dict = dict()

    for file_name in sensor_files:
        temp_df = pd.read_csv(path_to_dir + file_name)
        # converting to datetime, default values are in unix time
        temp_df[sensor_time_column_name] = pd.to_datetime(temp_df[sensor_time_column_name], unit="s")
        temp_df.columns = [sensor_time_column_name, pressure_column_name]

        # grouping by time, because some rows are duplicated
        sensors_dict[file_name] = temp_df.groupby(temp_df[sensor_time_column_name]).mean()

    for file_name in pump_files:
        temp_df = pd.read_csv(path_to_dir + file_name)
        prepared_df = pd.DataFrame()

        # converting to datetime, default values are in unix time
        prepared_df[pumps_time_column_name] = pd.to_datetime(temp_df[pumps_time_column_name], unit="s")
        # converting analog_2 to pressure in meters
        prepared_df[pressure_column_name] = ((temp_df["analog_input2"] - 0.6) * 4) * config.BARS_TO_METERS
        prepared_df["flow_rate_value(m3/h)"] = temp_df["flow_rate_value"]

        # grouping by time, because some rows could be duplicated
        pumps_dict[file_name] = prepared_df.groupby(prepared_df[pumps_time_column_name]).mean()

    # # sampling data to one hour
    # for sensor_name in sensors_dict:
    #     sensors_dict[sensor_name] = sensors_dict[sensor_name].resample(sampling_interval).mean()
    #
    # for pump_name in pumps_dict:
    #     pumps_dict[pump_name] = pumps_dict[pump_name].resample(sampling_interval).mean()

    sensors_pumps_dict = rename_dict_keys_and_merge(sensors_dict, pumps_dict)

    return sensors_pumps_dict