import logging
from datetime import datetime
from json import loads

from kafka.consumer.group import KafkaConsumer

import src.configfile as config
from src.helper_functions import visualize_node_groups
from src.kafka.output_json_functions import generate_error_response_json, generate_output_json
from src.multicovariate_models.gmm_functions import predict_groups_gmm
from src.state_comparator.NaNSensorsException import NaNSensorsException
from src.state_comparator.comparator_functions import convert_timestamp_to_epoch_seconds, prepare_input_kafka_1d_array


def process_kafka_msg_and_output_to_topic(producer, kafka_msg, ml_model, epanet_file):
    """
    Function acts as the main handler of the function. First it acquires the groups dictionary and other relevant
    information, it then visualizes the groups using plotly and saves the ouput to an html file, finally it converts
    the groups and other meta information to a json compliant object and sends it to the output topic.

    Exceptions are also handled here to abstract them away from the main function, function catches NaNSensorsException
    which is thrown when the kafka msg contains missing or wrong values, but it also captures all the other exceptions
    that may occur, but with less detail logging.

    :param producer: Kafka producer object. It is used to send the output json to the output topic.
    :param kafka_msg: Kafka message object. From here the sensors values and timestamp is extracted.
    :param ml_model: sklearn.mixture.GaussianMixture model. Currently only this model is supported, but the name implies
    generality since it will be adapted to support other models.
    :param epanet_file: String. Path to the epanet file.
    """
    try:
        # Check if the message contains any missing values and returns the groups
        groups_dict, diverged_node, epoch_timestamp = analyse_topic_and_find_leakage_groups(
            topic_msg_value=kafka_msg.value, ml_model=ml_model)

        # visualizes the network with the groups and saves it to the output html file
        visualize_node_groups(diverged_node, groups_dict, epanet_file, config.LEAK_AMOUNT,
                              filename="./grafana-files/braila_network.html")

        output_json = generate_output_json(epoch_seconds=epoch_timestamp, diverged_node=diverged_node,
                                           groups_dict=groups_dict, epanet_file=epanet_file)

        future = producer.send(config.OUTPUT_TOPIC, output_json)
        logging.info(f"Sent json msg to topic {config.OUTPUT_TOPIC}!")
        try:
            record_metadata = future.get(timeout=10)
        except Exception as e:
            logging.info("Producer error: " + str(e))

    except NaNSensorsException as e:
        logging.info("Sensor input data missing: " + str(e))
        error_output = generate_error_response_json(e.epoch_timestamp, e.sensor_list, epanet_file)

        producer.send(config.OUTPUT_TOPIC, error_output)

    # TODO improve exception handling -> more logging and less general exceptions catching


def find_msg_with_most_recent_timestamp(meta_signal_timestamp, leakage_detection_consumer=None):
    """
    Function loops through the messages on consumer (if given, else it creates its own) and finds the message
    closest (in terms of time) to the meta signal timestamp.

    CAUTION!: The kafka consumer should have the "consumer_timeout_ms" parameter set to a value greater then 0. Else
    this function can result in an infinite loop!

    :param meta_signal_timestamp: Epoch seconds timestamp. Of the meta signal from another topic.
    :param leakage_detection_consumer: Kafka consumer object, Optional. Will be used to loop through the messages.
    :return: Kafka message object. The message closest (in terms of time) to the meta signal timestamp.
    """
    t_key = "timestamp"
    if leakage_detection_consumer is None:
        # Build a new consumer, from scratch, option: auto_offset_reset="latest"
        leakage_detection_consumer = KafkaConsumer(bootstrap_servers=config.HOST_AND_PORT,
                                                   auto_offset_reset="earliest",
                                                   consumer_timeout_ms=2000,
                                                   value_deserializer=lambda v: loads(v.decode("utf-8"))
                                                   )
        leakage_detection_consumer.subscribe(config.TOPIC_V3)
        logging.info("Consumer 2: Subscribed to topic: " + config.TOPIC_V3)

    closest_time_msg = None
    for sensor_values_msg in leakage_detection_consumer:
        curr_timestamp = convert_timestamp_to_epoch_seconds(sensor_values_msg.value[t_key])
        t_diff = abs(curr_timestamp - meta_signal_timestamp)

        # The message closest to the meta sig. timestamp is taken, even if it is after it
        if closest_time_msg is None or t_diff < abs(closest_time_msg.value[t_key] - meta_signal_timestamp):
            sensor_values_msg.value[t_key] = curr_timestamp
            closest_time_msg = sensor_values_msg

    if closest_time_msg is None:
        raise RuntimeError(f"No message found on topic {config.TOPIC_V3}, when meta signal timestamp was "
                           f"{meta_signal_timestamp}!")

    return closest_time_msg


def analyse_topic_and_find_leakage_groups(topic_msg_value, ml_model):
    """
    Function analyses the message, first it checks for missing values and errors in the kafka 'ftr_vector', then it
    invokes the given ml model in this case we have only one model (GMM). Finally it extracts the first node in the
    first group as the most probable to have cause the leak and writes the information to the log file.

    :param topic_msg_value: Kafka message value object (dict). The msg value should contain 'timestamp' and 'ftr_vector'
    keys. The 'ftr_vector' should be a list of floats and the timestamp should be a unix timestamp either in seconds or
    milliseconds.
    :param ml_model: sklearn.mixture.GaussianMixture model. Currently only this model is supported, but the name implies
    generality since it will be adapted to support other models.
    :return: Tuple of three elements: (dictionary of groups with arrays of nodes as values, the most diverged node,
    the unix timestamp in seconds).
    """
    curr_epoch_seconds = convert_timestamp_to_epoch_seconds(topic_msg_value["timestamp"])
    feature_arr = topic_msg_value["ftr_vector"]
    prepared_array = prepare_input_kafka_1d_array(curr_epoch_seconds, feature_arr)

    groups_dict = predict_groups_gmm(ml_model, prepared_array)
    # Most diverged node is the one in the first group on the first index
    diverged_node = groups_dict["0"][0]

    # extra logging
    dt_time = datetime.fromtimestamp(curr_epoch_seconds)
    diverged_str = f"Most diverged node is: {diverged_node}. For values at datetime: {dt_time}"
    logging.info(diverged_str)

    return groups_dict, diverged_node, curr_epoch_seconds
