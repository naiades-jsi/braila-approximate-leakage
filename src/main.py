from src.divergence_matrix.DivergenceMatrixProcessor import DivergenceMatrixProcessor
from src.epanet.EPANETUtils import EPANETUtils
from src.epanet.NetworkVisualisation import plot_interactive_network, interactive_visualization
from src.state_comparator.comparator_functions import *
from src.helper_functions import visualize_node_groups
import src.configfile as config

from kafka import KafkaConsumer, KafkaProducer
from json import dumps, loads


def main(date):
    print("Starting analysis ... ... ")
    diverged_node, deviation = analyse_data_and_find_critical_sensor(config.SENSOR_DIR, config.SENSOR_FILES,
                                                                     config.PUMP_FILES, config.EPANET_NETWORK_FILE,
                                                                     config.SELECTED_NODES, date)

    print("Most diverged node is: " + diverged_node + ". Deviation is: " + str(deviation))
    if deviation > config.PRESSURE_DIFF_THRESHOLD:
        instance = DivergenceMatrixProcessor(config.DIVERGENCE_MATRIX_FILE)
        node_groups_dict = instance.get_affected_nodes_groups(config.LEAK_AMOUNT, diverged_node, num_of_groups=4,
                                                              method="jenks_natural_breaks")
        print("The nodes which influence this node the most are: ")
        print(node_groups_dict)

        # arr_of_nodes, df = instance.nodes_which_effect_the_sensors_most(16.0, diverged_node)
        visualize_node_groups(diverged_node, node_groups_dict, config.EPANET_NETWORK_FILE, config.LEAK_AMOUNT)


def service_main():
    print("Started the application !")
    # Data/objects that need to be calculated just one time
    consumer = KafkaConsumer(bootstrap_servers=config.HOST_AND_PORT, auto_offset_reset='earliest',
                             value_deserializer=lambda v: loads(v.decode('utf-8')))
    producer = KafkaProducer(bootstrap_servers=config.HOST_AND_PORT,
                             value_serializer=lambda v: dumps(v).encode('utf-8'))
    consumer.subscribe(config.TOPICS)

    instance = DivergenceMatrixProcessor(config.DIVERGENCE_MATRIX_FILE)
    epanet_simulated_df = create_epanet_pressure_df(config.EPANET_NETWORK_FILE, selected_nodes=config.SELECTED_NODES)

    print("Subscribed to topics: ", config.TOPICS)
    for msg in consumer:
        try:
            values = msg.value
            current_timestamp = values["time"]
            feature_arr = values["value"]
            print("Topic", msg.topic, "Timestamp ", current_timestamp, " ", feature_arr)
            diverged_node, deviation = analyse_kafka_topic_and_find_critical_sensor(current_timestamp, feature_arr,
                                                                                    epanet_simulated_df,
                                                                                    config.SELECTED_NODES)

            print("Most diverged node is: " + diverged_node + ". Deviation is: " + str(deviation))
            if deviation > config.PRESSURE_DIFF_THRESHOLD:
                output_groups_dict = instance.get_affected_nodes_groups(config.LEAK_AMOUNT, diverged_node,
                                                                        num_of_groups=4,
                                                                        method="jenks_natural_breaks+optimal_groups")

                output_topic = "predictions_{}".format("xy")
                future = producer.send(output_topic, output_groups_dict)
                print(output_topic + ": " + str(output_groups_dict))

                try:
                    record_metadata = future.get(timeout=10)
                except Exception as e:
                    print('Producer error: ' + str(e))

        except Exception as e:
            print('Consumer error: ' + str(e))


if __name__ == "__main__":
    # Kafka function
    service_main()

    # Local testing
    # main("2021-04-12")

    # Visualization
    # water_model = EPANETUtils(config.EPANET_NETWORK_FILE, "PDD").get_original_water_network_model()
    # print([i for i in water_model.valves()])
    # interactive_visualization(water_network_model=water_model)
