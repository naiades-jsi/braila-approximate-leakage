import http

import src.configfile as config
from src.epanet.EPANETUtils import EPANETUtils
from datetime import datetime


def prepare_output_json_meta_data(timestamp, sensor_with_leak, sensor_deviation, groups_dict, method, epanet_file):
    """
    TODO add documentation
    :param timestamp:
    :param sensor_with_leak:
    :param sensor_deviation:
    :param groups_dict:
    :param method:
    :param epanet_file:
    :return:
    """
    # Get meta data for nodes
    epanet_instance = EPANETUtils(epanet_file, "PDD")
    # Convert to flat structure from group dictionary
    groups_arr = epanet_instance.generate_node_array_with_meta_data(groups_dict)

    output_json = {
        config.OUTPUT_JSON_TIME_KEY: retrieve_unix_seconds_from_timestamp(timestamp),
        config.OUTPUT_JSON_TIME_PROCESSED_KEY: get_current_timestamp(),
        config.OUTPUT_JSON_STATUS_KEY: 200,
        config.OUTPUT_JSON_CRITICAL_SENSOR_KEY: sensor_with_leak,
        config.OUTPUT_JSON_DEVIATION_KEY: round(sensor_deviation, 4),
        config.OUTPUT_JSON_METHOD_KEY: method,
        config.OUTPUT_JSON_EPANET_F_KEY: get_epanet_file_version(epanet_file),
        config.OUTPUT_JSON_NODES_KEY: groups_arr
    }

    return output_json


def error_response(timestamp, nan_sensors_list, epanet_file):
    """
    # TODO add documentation
    :param timestamp:
    :param nan_sensors_list:
    :param epanet_file:
    :return:
    """
    # Get meta data for sensors with nan
    epanet_instance = EPANETUtils(epanet_file, "PDD")

    output_json = {
        config.OUTPUT_JSON_TIME_KEY: retrieve_unix_seconds_from_timestamp(timestamp),
        config.OUTPUT_JSON_TIME_PROCESSED_KEY: get_current_timestamp(),
        config.OUTPUT_JSON_STATUS_KEY: 412,
        config.OUTPUT_JSON_NODES_KEY: epanet_instance.generate_nan_sensors_meta_data(nan_sensors_list)
    }
    return output_json


def retrieve_unix_seconds_from_timestamp(timestamp):
    """
    Formats UNIX timestamp to UNIX timestamp in seconds.

    :param timestamp: UNIX timestamp in seconds or UNIX timestamp in milliseconds.
    :return: Returns UNIX timestamp in int seconds.
    """
    timestamp_digits = len(str(timestamp))
    if timestamp_digits == 10:
        epoch_seconds = timestamp
    elif timestamp_digits == 13:
        epoch_seconds = timestamp // 1000
    else:
        raise Exception("Timestamp is not in Unix milliseconds or seconds !")

    return int(epoch_seconds)


def get_epanet_file_version(epanet_file_name):
    return epanet_file_name.split("/")[-1].replace("_2.2.inp", "")


def get_current_timestamp():
    # TODO make timestamp bullet proof, hardcode timezone
    return int(datetime.now().timestamp())