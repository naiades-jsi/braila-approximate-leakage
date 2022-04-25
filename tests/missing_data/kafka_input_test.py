import unittest

import src.configfile as config
import numpy as np

from src.state_comparator.NaNSensorsException import NaNSensorsException
from src.state_comparator.comparator_functions import analyse_kafka_topic_and_find_critical_sensor
from src.state_comparator.sensor_data_preparator import create_epanet_pressure_df


class MissingKafkaDataTest(unittest.TestCase):
    EPANET_INPUT_FILE = "./../../data/epanet_networks/RaduNegru24May2021_2.2.inp"

    def test_missing_values(self):
        test_timestamp = 1638446992000
        feature_arr = [10.15, 10.15, 10.15, np.NAN, np.NAN, np.NAN, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15,
                       10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15,
                       10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15,
                       10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15,
                       10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15,
                       10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15,
                       10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15,
                       10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15,
                       10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15,
                       10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15,
                       10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15,
                       10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15, 10.15]

        epanet_simulated_df = create_epanet_pressure_df(self.EPANET_INPUT_FILE, selected_nodes=config.KAFKA_NODES_ORDER)

        with self.assertRaises(NaNSensorsException) as e:
            div_node, dev = analyse_kafka_topic_and_find_critical_sensor(test_timestamp,
                                                                         feature_arr,
                                                                         epanet_simulated_df,
                                                                         config.KAFKA_NODES_ORDER,
                                                                         minimum_present_values=21)

        self.assertEqual("Missing values in the following sensors" in e.exception.error_msg, True)

    def test_correct_values(self):
        feature_arr = [20] * len(config.KAFKA_NODES_ORDER) * 24
        test_timestamp = 1638446992000
        epanet_simulated_df = create_epanet_pressure_df(self.EPANET_INPUT_FILE, selected_nodes=config.KAFKA_NODES_ORDER)

        div_node, dev = analyse_kafka_topic_and_find_critical_sensor(test_timestamp,
                                                                     feature_arr,
                                                                     epanet_simulated_df,
                                                                     config.KAFKA_NODES_ORDER,
                                                                     minimum_present_values=0)

        self.assertEqual(type(div_node), str)
        self.assertEqual(type(dev), float)


if __name__ == '__main__':
    unittest.main()
