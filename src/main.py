from src.divergence_matrix.DivergenceMatrixProcessor import DivergenceMatrixProcessor
from src.state_comparator.comparator_functions import *

from src.helper_functions import pretty_print
import src.configfile as config


def main(date):
    print("Starting analysis ... ... ")
    diverged_node = analyse_data_and_find_critical_sensor(config.SENSOR_DIR, config.SENSOR_FILES, config.PUMP_FILES,
                                                          config.EPANET_NETWORK_FILE, config.SELECTED_NODES, date)
    print("Most diverged node is: " + diverged_node)

    instance = DivergenceMatrixProcessor(config.DIVERGENCE_MATRIX_FILE)
    arr_of_nodes, df = instance.nodes_which_effect_the_sensors_most(16.0, diverged_node)

    print("The nodes which influence this node the most are: ")
    print(pretty_print(arr_of_nodes))


if __name__ == "__main__":
    # Day which to compare to the simulated data
    main("2021-04-12")