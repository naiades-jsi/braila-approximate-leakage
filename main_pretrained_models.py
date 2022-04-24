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
from src.state_comparator.comparator_functions import analyse_kafka_topic_and_check_for_missing_values, \
    prepare_input_kafka_1d_array


def main_multiple_sensors_new_topic(path_to_model_pkl):
    """
    Function to run the main program which is subscribed to the kafka topic (config.TOPIC_V3) and predicts the nodes
    which are the mostly likely responsible for the leak.
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

    for msg in consumer:
        try:
            values = msg.value
            current_timestamp = values["timestamp"]
            feature_arr = values["ftr_vector"]
            prepared_array, epoch_sec = prepare_input_kafka_1d_array(current_timestamp, feature_arr)

            groups_dict = predict_groups_gmm(gmm_model, prepared_array)
            # Most diverged node is the one in the first group on the first index
            diverged_node = groups_dict["0"][0]

            # extra logging
            dt_time = datetime.fromtimestamp(epoch_sec)
            diverged_str = f"Most diverged node is: {diverged_node}. For values at datetime: {dt_time}"
            logging.info(diverged_str)

            output_json = prepare_output_json_meta_data(
                timestamp=current_timestamp,
                sensor_with_leak=diverged_node,
                sensor_deviation=0.0,  # "Information not available, when using this method"
                groups_dict=groups_dict,
                method="gmm+jenks_natural_breaks",
                epanet_file=config.EPANET_NETWORK_FILE_V2
            )

            future = producer.send(config.OUTPUT_TOPIC, output_json)
            visualize_node_groups(diverged_node, groups_dict, config.EPANET_NETWORK_FILE, config.LEAK_AMOUNT,
                                  filename="../grafana-files/braila_network.html")

            log_msg = f"Alert !! Deviation reached over threshold -Sensor: {diverged_node} -Time: {dt_time}"
            logging.info(log_msg)
            logging.info("")
            try:
                record_metadata = future.get(timeout=10)
            except Exception as e:
                logging.info("Producer error: " + str(e))

        except NaNSensorsException as e:
            logging.info("Sensor input data missing: " + str(e))
            error_output = generate_error_response_json(e.epoch_timestamp, e.sensors_list, config.EPANET_NETWORK_FILE)

            producer.send(config.OUTPUT_TOPIC, error_output)

        except Exception as e:
            logging.info("Consumer error: " + str(e))


if __name__ == "__main__":
    logging.basicConfig(filename=config.LOG_FILE_PRETRAINED,
                        level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S"
                        )
    main_multiple_sensors_new_topic("./data/trained_models/gmm_trained_model_30_03_2022.pkl")


# OUTDATED CODE, TODO remove when the newest method is tested
# def main_multiple_sensors(path_to_pkl_model):
#     logging.info("Started the application v2!")
#
#     # if you want to read msgs from start use: auto_offset_reset="earliest".
#     # group_id="braila_sensor_group" makes sure msgs are committed
#     consumer = KafkaConsumer(bootstrap_servers=config.HOST_AND_PORT, auto_offset_reset="earliest",
#                              value_deserializer=lambda v: loads(v.decode("utf-8")))
#     producer = KafkaProducer(bootstrap_servers=config.HOST_AND_PORT,
#                              value_serializer=lambda v: dumps(v).encode("utf-8"))
#     consumer.subscribe(config.TOPIC)
#     logging.info("Subscribed to topic: " + config.TOPIC)
#
#     # If you wished to generate a new one the function below should be used with prepared data.
#     # fit_gaussian_mixture_model(df)
#     # Load the model
#     with open(path_to_pkl_model, "rb") as model_file:
#         gmm_model = pickle.load(model_file)
#
#     for msg in consumer:
#         try:
#             values = msg.value
#             current_timestamp = values["timestamp"]
#             feature_arr = values["ftr_vector"]
#
#             actual_values_df = analyse_kafka_topic_and_check_for_missing_values(current_timestamp, feature_arr,
#                                                                                 config.KAFKA_NODES_ORDER_FALSE_USE_FOR_PARSING_ONLY, 8)
#             keep_cols = ["nan_sensor", "Jonctiune-3974", "Jonctiune-2749", "SenzorComunarzi-NatVech",
#                          "SenzorComunarzi-castanului", "SenzorChisinau-Titulescu", "SenzorCernauti-Sebesului"]
#             prepared_df = actual_values_df[keep_cols]
#             latest_data_row = prepared_df.iloc[0]
#             latest_prepared_arr = [np.array(latest_data_row.values, dtype=np.double)]
#             # hard-coding could be done until fixed
#             # latest_prepared_arr[0][0] = 16.039
#
#             groups_dict = predict_groups_gmm(gmm_model, latest_prepared_arr)
#             # Most diverged node is the one in the first group on the first index
#             diverged_node = groups_dict["0"][0]
#
#             # extra logging
#             dt_time = datetime.fromtimestamp(current_timestamp / 1000)
#             diverged_str = f"Most diverged node is: {diverged_node}. For values at datetime: {dt_time}"
#             logging.info(diverged_str)
#
#             output_json = prepare_output_json_meta_data(
#                 timestamp=current_timestamp,
#                 sensor_with_leak=diverged_node,
#                 sensor_deviation=0.0,  # "Information not available, when using this method"
#                 groups_dict=groups_dict,
#                 method="gmm+jenks_natural_breaks",
#                 epanet_file=config.EPANET_NETWORK_FILE_V2
#             )
#
#             future = producer.send(config.OUTPUT_TOPIC, output_json)
#             visualize_node_groups(diverged_node, groups_dict, config.EPANET_NETWORK_FILE, config.LEAK_AMOUNT,
#                                   filename="../grafana-files/braila_network.html")
#
#             log_msg = f"Alert !! Deviation reached over threshold -Sensor: {diverged_node} -Time: {dt_time}"
#             logging.info(log_msg)
#             logging.info("")
#             try:
#                 record_metadata = future.get(timeout=10)
#             except Exception as e:
#                 logging.info("Producer error: " + str(e))
#
#         except NaNSensorsException as e:
#             logging.info("Sensor input data missing: " + str(e))
#             error_output = error_response(e.epoch_timestamp, e.sensors_list, config.EPANET_NETWORK_FILE)
#
#             producer.send(config.OUTPUT_TOPIC, error_output)
#
#         except Exception as e:
#             logging.info("Consumer error: " + str(e))
