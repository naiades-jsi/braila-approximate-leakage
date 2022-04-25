import http

import src.configfile as config
from src.epanet.EPANETUtils import EPANETUtils
from datetime import datetime

from src.state_comparator.comparator_functions import convert_timestamp_to_epoch_seconds


def prepare_output_json_meta_data(timestamp, sensor_with_leak, sensor_deviation, groups_dict, method, epanet_file):
    """
    Function prepares the dictionary (JSON) which will be outputted by the application to the output kafka topic.
    It contains all the meta data that is seen below, the most important part is the groups_arr which contains the data
    about the nodes which are the most likely to have caused the leak.

    :param timestamp: Timestamp, in epoch seconds from when the data was collected.
    :param sensor_with_leak: String, of the sensor which is the most probable to have caused the leak.
    :param sensor_deviation: Float, the amount by which the sensor deviates from its simulated values. This value is not
    always provided since it doesn't provide enough meaningful information in some use cases
    (there you can set it to 0.0).
    :param groups_dict: Dictionary, with keys from 0...n which represent the group number and values which are lists of
    nodes in that group.
    :param method: String, method which was used to generate the groups and find the leaks.
    Methods: TODO add a list of all possible methods
    :param epanet_file: String, name of the EPANET file which was used to find the meta data about the sensors in the
    groups.
    :return: Dictionary, containing all the data to find the leak. Example of the output can be seen in the README.md
    """
    # Get meta data for nodes
    epanet_instance = EPANETUtils(epanet_file, "PDD")
    # Convert to flat structure from group dictionary
    groups_arr = epanet_instance.generate_node_array_with_meta_data(groups_dict)

    output_json = {
        config.OUTPUT_JSON_TIME_KEY: convert_timestamp_to_epoch_seconds(timestamp),
        config.OUTPUT_JSON_TIME_PROCESSED_KEY: get_current_timestamp(),
        config.OUTPUT_JSON_STATUS_KEY: 200,
        config.OUTPUT_JSON_CRITICAL_SENSOR_KEY: sensor_with_leak,
        config.OUTPUT_JSON_DEVIATION_KEY: round(sensor_deviation, 4),
        config.OUTPUT_JSON_METHOD_KEY: method,
        config.OUTPUT_JSON_EPANET_F_KEY: get_epanet_file_version(epanet_file),
        config.OUTPUT_JSON_NODES_KEY: groups_arr
    }

    return output_json


def generate_error_response_json(timestamp, nan_sensors_list, epanet_file):
    """
    Function produces a dictionary which is meant as an error response from the main application. The dictionary
    contains all the meta data that is available at the moment, timestamp of data collection, timestamp of when the
    data was processed in this function, status code 412 which means that incorrect data was sent to the server, and
    the list of sensors that didn't provide meaningful data.

    :param timestamp: Timestamp, in epoch seconds from when the data was collected.
    :param nan_sensors_list: List, of sensors that did contain zero or nan values.
    :param epanet_file: String, name of the EPANET file which was used to find the meta data about the sensors which
    contained wrong data.
    :return: Dictionary, containing all the meta data that is available at the moment.
    """
    # Get meta data for sensors with nan
    epanet_instance = EPANETUtils(epanet_file, "PDD")

    output_json = {
        config.OUTPUT_JSON_TIME_KEY: convert_timestamp_to_epoch_seconds(timestamp),
        config.OUTPUT_JSON_TIME_PROCESSED_KEY: get_current_timestamp(),
        config.OUTPUT_JSON_STATUS_KEY: 412,
        config.OUTPUT_JSON_NODES_KEY: epanet_instance.generate_nan_sensors_meta_data(nan_sensors_list)
    }
    return output_json


# def retrieve_unix_seconds_from_timestamp(timestamp):
#     """
#     Formats UNIX timestamp to UNIX timestamp in seconds or returns the original timestamp if already in seconds.
#
#     :param timestamp: UNIX timestamp in seconds or UNIX timestamp in milliseconds.
#     :return: Returns UNIX timestamp in int seconds.
#     """
#     timestamp_digits = len(str(timestamp))
#     if timestamp_digits == 10:
#         epoch_seconds = timestamp
#     elif timestamp_digits == 13:
#         epoch_seconds = timestamp // 1000
#     else:
#         raise Exception(f"Timestamp '{timestamp}' is not in Unix milliseconds or seconds !")
#
#     return int(epoch_seconds)


def get_epanet_file_version(epanet_file_name):
    return epanet_file_name.split("/")[-1].replace("_2.2.inp", "")


def get_current_timestamp():
    # TODO make timestamp bullet proof, hardcode timezone
    return int(datetime.now().timestamp())