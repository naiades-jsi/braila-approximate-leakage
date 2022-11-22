import os
import json

import pickle
from datetime import datetime
from json import loads, dumps

from kafka import KafkaConsumer, KafkaProducer
import logging

import src.configfile as config
from src.kafka.preprocessing_functions import process_kafka_msg, \
    find_msg_with_most_recent_timestamp
from src.kafka.output_json_functions import generate_error_response_json, generate_output_json
from src.state_comparator.comparator_functions import convert_timestamp_to_epoch_seconds
from src.state_comparator.NaNSensorsException import NaNSensorsException
# TODO - finish PCA reduction and find out how leakage amount actually effects simulated data
# TODO - test the new service on Atena


def main_multiple_sensors_new_topic(path_to_model_pkl, path_to_data_loader, epanet_file):
    """
    Function to run the main program which is subscribed to the kafka topic (config.TOPIC_V3) and predicts the nodes
    which are the mostly likely responsible for the leak.

    :param path_to_model_pkl: String. Path to the pickle file containing the trained machine learning model.
    :param epanet_file: String. Path to the epanet file.
    """
    logging.info("Started the application v3!")

    # connect_url = config.HOST_AND_PORT
    connect_url = 'localhost:9092'
    client_id = 'approximate-leakage'

    logging.info(f"connecting to {connect_url}")
    print('connecting')

    # if you want to read msgs from start use: auto_offset_reset="earliest".
    # group_id="braila_sensor_group" makes sure msgs are committed
    consumer = KafkaConsumer(bootstrap_servers=[connect_url], auto_offset_reset="earliest",
                             value_deserializer=lambda v: loads(v.decode("utf-8")),
                             client_id=client_id)
    producer = KafkaProducer(bootstrap_servers=[connect_url],
                             value_serializer=lambda v: dumps(v).encode("utf-8"),
                             client_id=client_id)
    consumer.subscribe(config.TOPIC_V3)
    logging.info("Subscribed to topic: " + config.TOPIC_V3)

    with open(path_to_model_pkl, "rb") as model_file:
        model = pickle.load(model_file)

    with open(path_to_data_loader, "rb") as model_file:
        data_loader = pickle.load(model_file)
    
    logging.info('model loaded, receiving messages')

    for latest_msg in consumer:
        try:
            logging.info(f'received message: {latest_msg}')

            response_json = process_kafka_msg(
                kafka_msg=latest_msg,
                ml_model=model,
                data_loader=data_loader,
                epanet_file=epanet_file
            )
        
            logging.info(f'sending message to {config.OUTPUT_TOPIC}: {response_json}')
            future = producer.send(config.OUTPUT_TOPIC, response_json)
            logging.info(f"Sent json msg to topic {config.OUTPUT_TOPIC}!")

            try:
                future.get(timeout=10)
            except Exception as e:
                logging.info("Producer error: " + str(e))
        except NaNSensorsException as sens_exc:
            error_output = generate_error_response_json(sens_exc.epoch_timestamp, sens_exc.sensor_list, epanet_file)
            producer.send(config.OUTPUT_TOPIC, error_output)
        except Exception as e:
            # write to errors.log
            with open(os.path.join(config.LOG_DIR, 'errors.log'), 'a') as f_out:
                err_str = repr(e)
                print(err_str)
                f_out.write(err_str + '\n')
            logging.info(f"Sensor input data missing: {str(s)}")


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

    producer = KafkaProducer(bootstrap_servers=config.HOST_AND_PORT,
                             value_serializer=lambda v: dumps(v).encode("utf-8"))

    with open(path_to_model_pkl, "rb") as model_file:
        gmm_model = pickle.load(model_file)

    # TODO add SPECIFIC exception handling ! 1. runtime error with specific msg!
    for latest_msg in meta_signal_consumer:
        msg_topic = latest_msg.topic
        meta_signal_timestamp = convert_timestamp_to_epoch_seconds(latest_msg.value["timestamp"])
        meta_signal_date = datetime.fromtimestamp(meta_signal_timestamp)
        meta_signal_value = latest_msg.value["status_code"]

        if meta_signal_value >= config.ANOMALY_META_SIGNAL_THRESHOLD:
            logging.info(f"Meta signal on topic '{msg_topic}' at time '{meta_signal_date}' is over threshold, "
                         f"with value '{meta_signal_value}'")

            closest_timestamp_msg = find_msg_with_most_recent_timestamp(meta_signal_timestamp)
            process_kafka_msg_and_output_to_topic(producer=producer, kafka_msg=closest_timestamp_msg,
                                                  ml_model=gmm_model, epanet_file=epanet_file)
        else:
            logging.info(f"Meta signal on topic '{msg_topic}' at time '{meta_signal_date}' is below threshold, "
                         f"with value '{meta_signal_value}'")


if __name__ == "__main__":
    logging.basicConfig(#filename=config.LOG_FILE_PRETRAINED,
                        level=logging.DEBUG,
                        format="%(asctime)s %(levelname)-8s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S"
                        )
    # Rerouting kafka logging to separate file
    kafka_logger = logging.getLogger("kafka")
    # TODO it doesn't actually reroute the logging to a file, it just changed the logging level and duplicates the out
    kafka_logger.setLevel(logging.WARNING)
    kafka_logger.addHandler(logging.FileHandler(config.LOG_FILE_KAFKA))

    # Old service, works as a standalone and outputs to topic on every message
    main_multiple_sensors_new_topic(
        "./data/models/model-knn-v2.pkl",
        "./data/models/model-knn-v2-data_loader.pkl",
        config.EPANET_NETWORK_FILE_V2
    )

    # New service, only triggers when meta signal is above threshold -> less resource consumption
    # main_multiple_sensors_new_topic_new_version("./data/trained_models/gmm_trained_model_30_03_2022.pkl",
    #                                             config.EPANET_NETWORK_FILE_V2)
