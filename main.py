from kafka.output_json_functions import generate_error_response_json, prepare_output_json_meta_data
from src.divergence_matrix.DivergenceMatrixProcessor import DivergenceMatrixProcessor
from src.epanet.EPANETUtils import EPANETUtils
from src.state_comparator.comparator_functions import *
from src.helper_functions import visualize_node_groups
import src.configfile as config

from kafka import KafkaConsumer, KafkaProducer
from json import dumps, loads
import logging

# TODO: fix paths in all of the files
# TODO: integrate pylint into code checking


def main(date):
    # logging.info("Started the application !")
    print("Starting analysis ... ... ")
    # Change nodes order if epanet file changes
    diverged_node, deviation = analyse_data_and_find_critical_sensor(config.SENSOR_DIR, config.SENSOR_FILES,
                                                                     config.PUMP_FILES, config.EPANET_NETWORK_FILE,
                                                                     config.LOCAL_TESTING_NODES_ORDER, date)

    print("Most diverged node is: " + diverged_node + ". Deviation is: " + str(deviation))
    if deviation > config.PRESSURE_DIFF_THRESHOLD:
        instance = DivergenceMatrixProcessor(config.DIVERGENCE_MATRIX_FILE)
        node_groups_dict = instance.get_affected_nodes_groups(config.LEAK_AMOUNT, diverged_node, num_of_groups=4,
                                                              method="jenks_natural_breaks")
        print("The nodes which influence this node the most are: ")
        print(node_groups_dict)

        # arr_of_nodes, df = instance.nodes_which_effect_the_sensors_most(16.0, diverged_node)
        visualize_node_groups(diverged_node, node_groups_dict, config.EPANET_NETWORK_FILE,
                              config.LEAK_AMOUNT)


def service_main():
    logging.info("Started the application !")

    # if you want to read msgs from start use: auto_offset_reset="earliest".
    # group_id="braila_sensor_group" makes sure msgs are committed
    consumer = KafkaConsumer(bootstrap_servers=config.HOST_AND_PORT, auto_offset_reset="earliest",
                             value_deserializer=lambda v: loads(v.decode("utf-8")))
    producer = KafkaProducer(bootstrap_servers=config.HOST_AND_PORT,
                             value_serializer=lambda v: dumps(v).encode("utf-8"))
    consumer.subscribe(config.TOPIC)
    logging.info("Subscribed to topic: " + config.TOPIC)

    instance = DivergenceMatrixProcessor(config.DIVERGENCE_MATRIX_FILE)
    epanet_simulated_df = create_epanet_pressure_df(config.EPANET_NETWORK_FILE, selected_nodes=config.KAFKA_NODES_ORDER)

    for msg in consumer:
        try:
            values = msg.value
            current_timestamp = values["timestamp"]
            feature_arr = values["ftr_vector"]
            diverged_node, deviation = analyse_kafka_topic_and_find_critical_sensor(current_timestamp, feature_arr,
                                                                                    epanet_simulated_df,
                                                                                    config.KAFKA_NODES_ORDER)
            # extra logging
            dt_time = datetime.fromtimestamp(current_timestamp / 1000)
            diverged_str = "Most diverged node is: {}. Deviation is: {:.2f}. For values at datetime: {}" \
                .format(diverged_node, deviation, dt_time)
            logging.info(diverged_str)

            if deviation > config.PRESSURE_DIFF_THRESHOLD:
                method = "jenks_natural_breaks"
                groups_dict = instance.get_affected_nodes_groups(config.LEAK_AMOUNT, diverged_node,
                                                                 num_of_groups=4,
                                                                 method=method)
                output_json = prepare_output_json_meta_data(timestamp=current_timestamp,
                                                            sensor_with_leak=diverged_node,
                                                            sensor_deviation=deviation,
                                                            groups_dict=groups_dict,
                                                            method=method,
                                                            epanet_file=config.EPANET_NETWORK_FILE)

                future = producer.send(config.OUTPUT_TOPIC, output_json)
                visualize_node_groups(diverged_node, groups_dict, config.EPANET_NETWORK_FILE, config.LEAK_AMOUNT,
                                      filename="./grafana-files/braila_network.html")

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
            error_output = generate_error_response_json(e.epoch_timestamp, e.sensor_list, config.EPANET_NETWORK_FILE)

            producer.send(config.OUTPUT_TOPIC, error_output)

        except Exception as e:
            logging.info("Consumer error: " + str(e))


def main_extraction():
    """Used for generating a map of the network"""
    epanet_instance = EPANETUtils(conf.EPANET_NETWORK_FILE, "PDD")
    epanet_instance.generate_network_json_in_wgs84()


if __name__ == "__main__":
    logging.basicConfig(filename=config.LOG_FILE, level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    # Kafka function
    service_main()  # if used without the correct topic replace feature array with fake data

    # # Local testing
    # main("2021-04-12")

    # Visualization
    # water_model = EPANETUtils(config.EPANET_NETWORK_FILE, "PDD").get_original_water_network_model()
    # # print([i for i in water_model.valves()])
    # interactive_visualization(water_network_model=water_model)
