
from datetime import datetime
from json import loads, dumps
from kafka import KafkaConsumer, KafkaProducer
import logging

import src.configfile as config
from helper_functions import visualize_node_groups
from multicovariate_models.gmm_functions import predict_groups_gmm
from output_json_functions import error_response, prepare_output_json_meta_data
from state_comparator.NaNSensorsException import NaNSensorsException
from state_comparator.comparator_functions import analyse_kafka_topic_and_check_for_missing_values


def main_multiple_sensors():
    logging.info("Started the application !")

    # if you want to read msgs from start use: auto_offset_reset="earliest".
    # group_id="braila_sensor_group" makes sure msgs are committed
    consumer = KafkaConsumer(bootstrap_servers=config.HOST_AND_PORT, auto_offset_reset="earliest",
                             value_deserializer=lambda v: loads(v.decode("utf-8")))
    producer = KafkaProducer(bootstrap_servers=config.HOST_AND_PORT,
                             value_serializer=lambda v: dumps(v).encode("utf-8"))
    consumer.subscribe(config.TOPIC)
    logging.info("Subscribed to topic: " + config.TOPIC)

    # If you wished to generate a new one the function below should be used with prepared data.
    # fit_gaussian_mixture_model(df)

    # Load the model
    gmm_model = None    # TODO load model

    for msg in consumer:
        try:
            values = msg.value
            current_timestamp = values["timestamp"]
            feature_arr = values["ftr_vector"]
            actual_values_df = analyse_kafka_topic_and_check_for_missing_values(current_timestamp, feature_arr,
                                                                                config.KAFKA_NODES_ORDER, 8)
            # extra logging
            dt_time = datetime.fromtimestamp(current_timestamp / 1000)
            diverged_str = f"Most diverged node is: {'TODO'}. For values at datetime: {dt_time}"
            logging.info(diverged_str)

            groups_dict = predict_groups_gmm(actual_values_df, producer)
            diverged_node = groups_dict[0][0]

            output_json = prepare_output_json_meta_data(
                timestamp=current_timestamp,
                sensor_with_leak=diverged_node,
                sensor_deviation="Information not available, when using this method",
                groups_dict=groups_dict,
                method="jenks_natural_breaks",
                epanet_file=config.EPANET_NETWORK_FILE
            )

            future = producer.send(config.OUTPUT_TOPIC, output_json)
            visualize_node_groups(diverged_node, groups_dict, config.EPANET_NETWORK_FILE, config.LEAK_AMOUNT,
                                  filename="../grafana-files/braila_network.html")

            log_msg = "Alert !! Deviation reached over threshold -Sensor: {} -Time: {}" \
                .format(diverged_node, dt_time)
            logging.info(log_msg)
            logging.info("")
            try:
                record_metadata = future.get(timeout=10)
            except Exception as e:
                logging.info("Producer error: " + str(e))

        except NaNSensorsException as e:
            logging.info("Sensor input data missing: " + str(e))
            error_output = error_response(e.epoch_timestamp, e.sensors_list, config.EPANET_NETWORK_FILE)

            producer.send(config.OUTPUT_TOPIC, error_output)

        except Exception as e:
            logging.info("Consumer error: " + str(e))


if __name__ == "__main__":
    main_multiple_sensors()









