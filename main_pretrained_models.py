import pickle
from datetime import datetime
from json import loads, dumps

from kafka import KafkaConsumer, KafkaProducer
import logging

import src.configfile as config
from kafka.preprocessing_functions import process_kafka_msg_and_output_to_topic, \
    find_msg_with_most_recent_timestamp
from src.state_comparator.comparator_functions import convert_timestamp_to_epoch_seconds
# TODO finish PCA reduction and find out how leakage amount actually effects simulated data


def main_multiple_sensors_new_topic(path_to_model_pkl, epanet_file):
    """
    Function to run the main program which is subscribed to the kafka topic (config.TOPIC_V3) and predicts the nodes
    which are the mostly likely responsible for the leak.

    :param path_to_model_pkl: String. Path to the pickle file containing the trained machine learning model.
    :param epanet_file: String. Path to the epanet file.
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
        process_kafka_msg_and_output_to_topic(producer=producer, kafka_msg=latest_msg, ml_model=gmm_model,
                                              epanet_file=epanet_file)


def main_multiple_sensors_new_topic_new_version(path_to_model_pkl, epanet_file):
    """
    Function to combine the functionality of this service, with the already existing service which finds
    anomalies on the input signal. If this meta signal reaches over threshold specified in
    config.ANOMALY_META_SIGNAL_THRESHOLD the service analyses the most recent data on the leakage topic and
    sends an output to the output topic.

    :param path_to_model_pkl: String. Path to the pickle file containing the trained machine learning model.
    :param epanet_file: String. Path to the epanet file.
    """
    logging.info("Started the application v3!")
    meta_signal_consumer = KafkaConsumer(bootstrap_servers=config.HOST_AND_PORT, auto_offset_reset="earliest",
                                         value_deserializer=lambda v: loads(v.decode("utf-8")))
    meta_signal_consumer.subscribe(topics=config.ANOMALY_META_SIGNAL_TOPICS)
    logging.info(f"Consumer 1: Subscribed to topics: {str(config.ANOMALY_META_SIGNAL_TOPICS)}")

    # auto_offset_reset="latest"
    leakages_consumer = KafkaConsumer(bootstrap_servers=config.HOST_AND_PORT,
                                      auto_offset_reset="earliest",
                                      enable_auto_commit=True,
                                      group_id="braila_approx_leakage_sensor_group",
                                      auto_commit_interval_ms=2000,
                                      consumer_timeout_ms=5000,
                                      value_deserializer=lambda v: loads(v.decode("utf-8"))
                                      )
    leakages_consumer.subscribe(config.TOPIC_V3)
    logging.info("Consumer 2: Subscribed to topic: " + config.TOPIC_V3)

    producer = KafkaProducer(bootstrap_servers=config.HOST_AND_PORT,
                             value_serializer=lambda v: dumps(v).encode("utf-8"))

    with open(path_to_model_pkl, "rb") as model_file:
        gmm_model = pickle.load(model_file)

    for latest_msg in meta_signal_consumer:
        try:
            msg_topic = latest_msg.topic
            meta_signal_timestamp = convert_timestamp_to_epoch_seconds(latest_msg.value["timestamp"])
            meta_signal_date = datetime.fromtimestamp(meta_signal_timestamp)
            meta_signal_value = latest_msg.value["status_code"]

            if meta_signal_value >= config.ANOMALY_META_SIGNAL_THRESHOLD:
                logging.info(f"Meta signal on topic '{msg_topic}' at time '{meta_signal_date}' is over threshold, "
                             f"with value '{meta_signal_value}'")

                closest_timestamp_msg = find_msg_with_most_recent_timestamp(leakages_consumer, meta_signal_timestamp)
                process_kafka_msg_and_output_to_topic(producer=producer, kafka_msg=closest_timestamp_msg,
                                                      ml_model=gmm_model, epanet_file=epanet_file)
            else:
                logging.info(f"Meta signal on topic '{msg_topic}' at time '{meta_signal_date}' is below threshold, "
                             f"with value '{meta_signal_value}'")

        except Exception as e:
            logging.info("Meta signal Consumer error: " + str(e))


if __name__ == "__main__":
    logging.basicConfig(filename=config.LOG_FILE_PRETRAINED,
                        level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S"
                        )
    main_multiple_sensors_new_topic("./data/trained_models/gmm_trained_model_30_03_2022.pkl",
                                    config.EPANET_NETWORK_FILE_V2)
    # main_multiple_sensors_new_topic_new_version("./data/trained_models/gmm_trained_model_30_03_2022.pkl",
    #                                             config.EPANET_NETWORK_FILE_V2)
