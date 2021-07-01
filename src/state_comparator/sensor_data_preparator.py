import pandas as pd

from src.epanet.EPANETUtils import EPANETUtils


def load_and_prepare_sensor_data(path_to_data_dir, sensor_files, pump_files):
    """
    Function reads the data from all the files into two dictionaries of dataframes. For the pump files it also converts
    the analog_2 column to the meters unit and discards all the other columns. It then resamples all the data to 1 hour
    for easier comparison and finally calls the rename_dict_keys_and_merge function which combines the dictionaries
    and merges the two dicts together.

    :param path_to_data_dir: Path to the directory in which the files for sensors/pumps are stored.
    :param sensor_files: Array of strings containing names of the csv sensors files.
    :param pump_files: Array of strings containing names of the csv pump files.
    :return: Returns a dictionary of dataframe with timestamps and pressure in meters.
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
    """
    Function maps/renames file names which were used as dict keys to the actual name of the nodes. It also merges
    the sensor_dict and pump_dict into one data_frame.

    :param sensor_dict: Dictionary with key being the file name and values being a dataframe of
    timestamps and pressures for nodes.
    :param pump_dict: Dictionary with key being the file name and values being a dataframe of
    timestamps and pressures for nodes.
    :param sensor_names_dict: Dict with filenames as key and values being the actual sensor names.
    :param pump_names_dict: Dict with filenames as key and values being the actual pump names.

    :return: Returns a new dict which contains all the values of the previous two dicts (sensor_dict and pump_dict),
    with new keys that are the actual sensor/pump names instead of filenames.
    """
    sensor_pump_dict_new = dict()

    # default values TODO move them to variable folder
    if sensor_names_dict is None:
        sensor_names_dict = dict()
        sensor_names_dict["braila_pressure5770.csv"] = "SenzorComunarzi-NatVech"
        sensor_names_dict["braila_pressure5771.csv"] = "SenzorComunarzi-castanului"
        sensor_names_dict["braila_pressure5772.csv"] = "SenzorChisinau-Titulescu"
        sensor_names_dict["braila_pressure5773.csv"] = "SenzorCernauti-Sebesului"
    if pump_names_dict is None:
        pump_names_dict = dict()
        # almost correct but these were not provided by CUP Braila
        # OUTDATED but needed for backwards compatibility
        # pump_names_dict["braila_flow211206H360.csv"] = "748-B"      # ("Apollo", "748-B"), ,
        # pump_names_dict["braila_flow211106H360.csv"] = "751-B"      # ("GA-Braila", "751-B")
        # pump_names_dict["braila_flow211306H360.csv"] = "763-B"      # ("RaduNegruMare", "763-B")
        # pump_names_dict["braila_flow318505H498.csv"] = "760-B"      # ("RaduNegru2", "760-B")

        # from mail (Marius)
        pump_names_dict["braila_flow211206H360.csv"] = "Jonctiune-3974"     # ("Apollo", "Jonctiune-3974"), ,
        pump_names_dict["braila_flow211106H360.csv"] = "Jonctiune-J-3"      # ("GA-Braila", "Jonctiune-J-3")
        pump_names_dict["braila_flow318505H498.csv"] = "Jonctiune-J-19"     # ("RaduNegru2", "Jonctiune-J-19")
        pump_names_dict["braila_flow211306H360.csv"] = "Jonctiune-2749"     # ("RaduNegruMare", "Jonctiune-2749")

    for sensor in sensor_dict:
        new_name = sensor_names_dict[sensor]
        sensor_pump_dict_new[new_name] = sensor_dict[sensor]

    for pump in pump_dict:
        new_name = pump_names_dict[pump]
        sensor_pump_dict_new[new_name] = pump_dict[pump]

    return sensor_pump_dict_new


def create_epanet_pressure_df(epanet_file, selected_nodes=None):
    """
    Creates an instance of an EPANETUtils class and runs the simulation.

    :param epanet_file: File name of the EPANET water network model.
    :param selected_nodes: Array of nodes, only data for the nodes in this array will be returned if an array is passed.
    :return: Returns a dataframe of nodes and their pressures at different times in the simulation duration.
    """
    epanet_instance = EPANETUtils(epanet_file, "PDD")
    return epanet_instance.generate_pressures_at_nodes(selected_nodes=selected_nodes, to_hours_round=True)