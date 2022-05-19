import pickle
from datetime import datetime
from json import loads, dumps

from kafka import KafkaConsumer, KafkaProducer
import logging

import src.configfile as config
from src.helper_functions import visualize_node_groups
from src.multicovariate_models.gmm_functions import predict_groups_gmm
from src.output_json_functions import generate_error_response_json, prepare_output_json_meta_data
from src.state_comparator.NaNSensorsException import NaNSensorsException
from src.state_comparator.comparator_functions import prepare_input_kafka_1d_array


def main_multiple_sensors_new_topic(path_to_model_pkl, epanet_file):
    """
    Function to run the main program which is subscribed to the kafka topic (config.TOPIC_V3) and predicts the nodes
    which are the mostly likely responsible for the leak.

    :param path_to_model_pkl: String. Path to the pickle file containing the trained machine learning model.
    """
    logging.info("Started the application v3!")

    # if you want to read msgs from start use: auto_offset_reset="earliest".
    # group_id="braila_sensor_group" makes sure msgs are committed
    consumer = KafkaConsumer(bootstrap_servers=config.HOST_AND_PORT, auto_offset_reset="earliest",
                             value_deserializer=lambda v: loads(v.decode("utf-8")))
    producer = KafkaProducer(bootstrap_servers=config.HOST_AND_PORT,
                             value_serializer=lambda v: dumps(v).encode("utf-8"))
    consumer.subscribe(config.TOPIC_V3)
    logging.info("Subscribed to topic: " + config.TOPIC_V3)

    with open(path_to_model_pkl, "rb") as model_file:
        gmm_model = pickle.load(model_file)

    for latest_msg in consumer:
        try:
            output_json = analyse_leakage_topic_and_generate_output(latest_msg.value, gmm_model, epanet_file)

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

        except Exception as e:
            logging.info("Consumer error: " + str(e))


def analyse_leakage_topic_and_generate_output(topic_msg_value, gmm_model, epanet_file):
    """
    TODO add documentation
    :param epanet_file:
    :param topic_msg_value:
    :param gmm_model:
    :return:
    """
    current_timestamp = topic_msg_value["timestamp"]
    feature_arr = topic_msg_value["ftr_vector"]
    prepared_array, epoch_sec = prepare_input_kafka_1d_array(current_timestamp, feature_arr)

    groups_dict = predict_groups_gmm(gmm_model, prepared_array)
    # Most diverged node is the one in the first group on the first index
    diverged_node = groups_dict["0"][0]

    # extra logging
    dt_time = datetime.fromtimestamp(epoch_sec)
    diverged_str = f"Most diverged node is: {diverged_node}. For values at datetime: {dt_time}"
    logging.info(diverged_str)
    visualize_node_groups(diverged_node, groups_dict, epanet_file, config.LEAK_AMOUNT,
                          filename="./grafana-files/braila_network.html")

    output_json = prepare_output_json_meta_data(
        timestamp=current_timestamp,
        sensor_with_leak=diverged_node,
        sensor_deviation=0.0,  # "Information not available, when using this method"
        groups_dict=groups_dict,
        method="gmm+jenks_natural_breaks",
        epanet_file=config.EPANET_NETWORK_FILE_V2
    )

    return output_json


def main_multiple_sensors_new_topic_new_version(path_to_model_pkl):
    """
    Function to combine the functionality of this service, with the already existing service which finds
    anomalies on the input signal. If this meta signal reaches over threshold specified in
    config.ANOMALY_META_SIGNAL_THRESHOLD the service analyses the most recent data on the leakage topic and
    sends an output to the output topic.

    :param path_to_model_pkl: String. Path to the pickle file containing the trained machine learning model.
    """
    logging.info("Started the application v3!")
    meta_signal_consumer = KafkaConsumer(bootstrap_servers=config.HOST_AND_PORT, auto_offset_reset="earliest",
                                         value_deserializer=lambda v: loads(v.decode("utf-8")))
    meta_signal_consumer.subscribe(topics=config.ANOMALY_META_SIGNAL_TOPICS)
    logging.info("Consumer 1: Subscribed to topics: " + config.ANOMALY_META_SIGNAL_TOPICS)

    leakages_consumer = KafkaConsumer(bootstrap_servers=config.HOST_AND_PORT, auto_offset_reset="earliest",
                                      value_deserializer=lambda v: loads(v.decode("utf-8")))
    leakages_consumer.subscribe(config.TOPIC_V3)
    logging.info("Consumer 2: Subscribed to topic: " + config.TOPIC_V3)

    producer = KafkaProducer(bootstrap_servers=config.HOST_AND_PORT,
                             value_serializer=lambda v: dumps(v).encode("utf-8"))

    with open(path_to_model_pkl, "rb") as model_file:
        gmm_model = pickle.load(model_file)

    for latest_msg in meta_signal_consumer:
        try:
            msg_topic = latest_msg.topic
            meta_signal_timestamp = latest_msg.value["timestamp"]
            meta_signal_date = datetime.fromtimestamp(meta_signal_timestamp)
            meta_signal_value = latest_msg.value["status_code"]

            if meta_signal_value >= config.ANOMALY_META_SIGNAL_THRESHOLD:
                logging.info(f"Meta signal on topic '{msg_topic}' at time '{meta_signal_date}' is over threshold, "
                             f"with value '{meta_signal_value}'")

                try:
                    # TODO make sure to retrieve the latest message
                    output_json = analyse_leakage_topic_and_generate_output(latest_msg.value, gmm_model)

                    future = producer.send(config.OUTPUT_TOPIC, output_json)
                    logging.info(f"Sent json msg to topic {config.OUTPUT_TOPIC}!")
                    try:
                        record_metadata = future.get(timeout=10)
                    except Exception as e:
                        logging.info("Producer error: " + str(e))

                except NaNSensorsException as e:
                    logging.info("Sensor input data missing: " + str(e))
                    error_output = generate_error_response_json(e.epoch_timestamp, e.sensor_list,
                                                                config.EPANET_NETWORK_FILE_V2)
                    producer.send(config.OUTPUT_TOPIC, error_output)
            else:
                logging.info(f"Meta signal on topic '{msg_topic}' at time '{meta_signal_date}' is below threshold, "
                             f"with value '{meta_signal_value}'")

        except Exception as e:
            # TODO make error handling more specific
            logging.info("Consumer error: " + str(e))


if __name__ == "__main__":
    logging.basicConfig(filename=config.LOG_FILE_PRETRAINED,
                        level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S"
                        )
    main_multiple_sensors_new_topic("./data/trained_models/gmm_trained_model_30_03_2022.pkl",
                                    config.EPANET_NETWORK_FILE_V2)
    # TODO finish PCA reduction and find out how leakage amount actually effects simulated data
