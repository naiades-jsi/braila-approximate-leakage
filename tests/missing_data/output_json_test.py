import unittest

import src.configfile as config
from src.divergence_matrix.DivergenceMatrixProcessor import DivergenceMatrixProcessor
# TODO update tests to new format


class OutputJsonTest(unittest.TestCase):
    # make this responsive by always progressing from top directory
    EPANET_INPUT_FILE = "./../../data/epanet_networks/RaduNegru24May2021_2.2.inp"
    DIVERGENCE_MATRIX_FILE = "./../../data/divergence_matrix/Divergence_M.pickle"

    def test_json_with_meta_data(self):
        diverged_node = config.KAFKA_NODES_ORDER[0]
        method = "jenks_natural_breaks"
        timestamp = 1638446992000
        deviation = 4.5

        instance = DivergenceMatrixProcessor(self.DIVERGENCE_MATRIX_FILE)
        groups_dict = instance.get_affected_nodes_groups(config.LEAK_AMOUNT, diverged_node,
                                                         num_of_groups=4,
                                                         method=method)

        output_json = instance.prepare_output_json_meta_data(timestamp=timestamp,
                                                             sensor_with_leak=diverged_node,
                                                             sensor_deviation=deviation,
                                                             groups_dict=groups_dict,
                                                             epanet_file=self.EPANET_INPUT_FILE,
                                                             method=method)
        print("Output:", output_json)

        self.assertEqual(config.OUTPUT_JSON_TIME_KEY in output_json, True)
        self.assertEqual(config.OUTPUT_JSON_TIME_PROCESSED_KEY in output_json, True)
        self.assertEqual(config.OUTPUT_JSON_CRITICAL_SENSOR_KEY in output_json, True)
        self.assertEqual(config.OUTPUT_JSON_DEVIATION_KEY in output_json, True)
        self.assertEqual(config.OUTPUT_JSON_METHOD_KEY in output_json, True)
        self.assertEqual(config.OUTPUT_JSON_EPANET_F_KEY in output_json, True)
        self.assertEqual(config.OUTPUT_JSON_NODES_KEY in output_json, True)

        node_data = output_json[config.OUTPUT_JSON_NODES_KEY][0]
        self.assertEqual(config.OUTPUT_JSON_NODE_NAME_KEY in node_data, True)
        self.assertEqual(config.OUTPUT_JSON_NODE_LAT_KEY in node_data, True)
        self.assertEqual(config.OUTPUT_JSON_NODE_LONG_KEY in node_data, True)
        self.assertEqual(config.OUTPUT_JSON_NODE_GROUP_KEY in node_data, True)


if __name__ == '__main__':
    unittest.main()
