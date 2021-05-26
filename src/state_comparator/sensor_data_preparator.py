import pandas as pd

from src.epanet.EPANETUtils import EPANETUtils


def load_and_prepare_sensor_data(path_to_data_dir, sensor_files, pump_files):
    """

    ;returns: Returns two dicts of dataframe with timestamp and pressure in meters.
    """
    # Reads for all the 8 sensors and prepares it in a standard form for comparison
    if len(sensor_files) < 1 and len(pump_files) < 1:
        raise Exception("Files names list must contain at least one string to file !!")

    # official formula: 10^5 / /(1000 * 9,80655) = 10,197
    BARS_TO_METERS = 10.197
    sensor_time_column_name = "time"
    pumps_time_column_name = "time"
    sensors_dict = dict()
    pumps_dict   = dict()

    for file_name in sensor_files:
        temp_df = pd.read_csv(path_to_data_dir + file_name)
        # converting to datetime, default values are in unix time
        temp_df[sensor_time_column_name] = pd.to_datetime(temp_df[sensor_time_column_name], unit="s")

        # grouping by time, because some rows are duplicated
        sensors_dict[file_name] = temp_df.groupby(temp_df[sensor_time_column_name]).mean()

    for file_name in pump_files:
        temp_df = pd.read_csv(path_to_data_dir + file_name)
        prepared_df = pd.DataFrame()

        # converting to datetime, default values are in unix time
        prepared_df[pumps_time_column_name] = pd.to_datetime(temp_df[pumps_time_column_name], unit="s")
        # converting analog_2 to pressure in meters
        prepared_df["value"] = ((temp_df["analog_input2"] - 0.6) * 4) * BARS_TO_METERS

        # grouping by time, because some rows could be duplicated
        pumps_dict[file_name] = prepared_df.groupby(prepared_df[pumps_time_column_name]).mean()

    # sampling data to one hour
    for sensor_name in sensors_dict:
        sensors_dict[sensor_name] = sensors_dict[sensor_name].resample("1H").mean()

    for pump_name in pumps_dict:
        pumps_dict[pump_name] = pumps_dict[pump_name].resample("1H").mean()

    sensors_pumps_dict = rename_dict_keys_and_merge(sensors_dict, pumps_dict)
    return sensors_pumps_dict


def rename_dict_keys_and_merge(sensor_dict, pump_dict, sensor_names_dict=None, pump_names_dict=None):
    sensor__pump_dict_new = dict()

    # default values TODO move them to variable folder
    if sensor_names_dict is None:
        sensor_names_dict = dict()
        sensor_names_dict["braila_pressure5770.csv"] = "SenzorComunarzi-NatVech"
        sensor_names_dict["braila_pressure5771.csv"] = "SenzorComunarzi-castanului"
        sensor_names_dict["braila_pressure5772.csv"] = "SenzorChisinau-Titulescu"
        sensor_names_dict["braila_pressure5773.csv"] = "SenzorCernauti-Sebesului"
    if pump_names_dict is None:
        pump_names_dict = dict()                                    # from mail
        pump_names_dict["braila_flow211206H360.csv"] = "748-B"      # ("Apollo", "748-B"), ,
        pump_names_dict["braila_flow211106H360.csv"] = "751-B"      # ("GA-Braila", "751-B")
        pump_names_dict["braila_flow211306H360.csv"] = "763-B"      # ("RaduNegruMare", "763-B")
        pump_names_dict["braila_flow318505H498.csv"] = "760-B"      # ("RaduNegru2", "760-B")

    for sensor in sensor_dict:
        new_name = sensor_names_dict[sensor]
        sensor__pump_dict_new[new_name] = sensor_dict[sensor]

    for pump in pump_dict:
        new_name = pump_names_dict[pump]
        sensor__pump_dict_new[new_name] = pump_dict[pump]

    return sensor__pump_dict_new


def create_epanet_pressure_df(epanet_file, selected_nodes=None):
    epanet_instance = EPANETUtils(epanet_file, "PDD")
    return epanet_instance.generate_pressures_at_nodes(selected_nodes=selected_nodes, to_hours_round=True)