"""
This file stores the configuration variables used in many files and is meant as a central
location for all configuration variables.
"""

# Minimum count of values that need to be present to process data
MINIMUM_PRESENT_VALUES_THRESHOLD = 14

# Amount with which to run the simulation
LEAK_AMOUNT = 4.0

# If difference between simulated and actual data is bigger than this then alert
PRESSURE_DIFF_THRESHOLD = 1.1

# official formula: 10^5 / /(1000 * 9,80655) = 10,197
BARS_TO_METERS = 10.197

# Data directories
DATA_DIRECTORY = "./data/"
SENSOR_DIR = DATA_DIRECTORY + "sensor_data/"
DIVERGENCE_DATA_DIR = DATA_DIRECTORY + "divergence_matrix/"
EPANET_NETWORKS = DATA_DIRECTORY + "epanet_networks/"

EPANET_NETWORK_FILE = EPANET_NETWORKS + "RaduNegru24May2021_2.2.inp"
EPANET_NETWORK_FILE_V2 = EPANET_NETWORKS + "Braila_V2022_2_2.inp"

# EPANET_NETWORK_FILE = EPANET_NETWORKS + "RaduNegru23July2021_2.2.inp"
DIVERGENCE_MATRIX_FILE = DIVERGENCE_DATA_DIR + "Divergence_M.pickle"

LOG_FILE = "logs/service-log.log"
LOG_FILE_PRETRAINED = "logs/service-log-pretrained.log"

# Sensor and pump file names arrays
SENSOR_FILES = ["braila_pressure5770.csv", "braila_pressure5771.csv", "braila_pressure5772.csv",
                "braila_pressure5773.csv"]
SENSORS_TUPLES = [("braila_pressure5770.csv", "SenzorComunarzi-NatVech"),
                  ("braila_pressure5771.csv", "SenzorComunarzi-castanului"),
                  ("braila_pressure5772.csv", "SenzorChisinau-Titulescu"),
                  ("braila_pressure5773.csv", "SenzorCernauti-Sebesului")]

PUMP_FILES = ["braila_flow211206H360.csv", "braila_flow211106H360.csv", "braila_flow211306H360.csv",
              "braila_flow318505H498.csv"]

PUMPS_TUPLES = [("braila_flow211206H360.csv", "Jonctiune-3974"),
                ("braila_flow211106H360.csv", "Jonctiune-J-3"),
                ("braila_flow318505H498.csv", "Jonctiune-J-19"),
                ("braila_flow211306H360.csv", "Jonctiune-2749")
                ]

LOCAL_TESTING_NODES_ORDER = ["Jonctiune-3974", "Jonctiune-J-3", "Jonctiune-2749", "Jonctiune-J-19",
                             "SenzorComunarzi-NatVech", "SenzorComunarzi-castanului", "SenzorChisinau-Titulescu",
                             "SenzorCernauti-Sebesului"]

# # KAFKA
# order in array is important !!, since it is mapped by index
KAFKA_NODES_ORDER = ["Jonctiune-3974", "Jonctiune-2749", "Jonctiune-J-19", "SenzorComunarzi-NatVech",
                     "SenzorComunarzi-castanului", "SenzorChisinau-Titulescu", "SenzorCernauti-Sebesului"]

# KAFKA NODES false order, but used for parsing
KAFKA_NODES_ORDER_FALSE_USE_FOR_PARSING_ONLY = ["nan_sensor", "Jonctiune-3974",
                                                "Jonctiune-2749", "Jonctiune-J-19",
                                                "SenzorComunarzi-NatVech", "SenzorComunarzi-castanului",
                                                "SenzorChisinau-Titulescu", "SenzorCernauti-Sebesului"]

# order for the new approach, order in array is important!!
KAFKA_NODES_ORDER_LATEST = ["J-Apollo", "J-RN1", "J-RN2", "Sensor1", "Sensor3", "Sensor4", "Sensor2"]
KAFKA_NODES_ORDER_LATEST_WRONG = ["nan_sensor", "J-Apollo", "J-RN1", "J-RN2", "Sensor1", "Sensor3", "Sensor4",
                                  "Sensor2"]

# KAFKA related
HOST_AND_PORT = "194.249.231.11:9092"

# Topic used in version 1 and 2, left in for backwards compatibility
TOPIC = "features_braila_leakage_detection"

TOPIC_V3 = "features_braila_leakage_detection_updated"
OUTPUT_TOPIC = "braila_leakage_groups"

ANOMALY_META_SIGNAL_TOPICS = ["anomalies_braila_flow211106H360_meta_signal",
                              "anomalies_braila_flow211206H360_meta_signal",
                              "anomalies_braila_flow211306H360_meta_signal",
                              "anomalies_braila_flow318505H498_meta_signal",
                              "anomalies_braila_pressure5770_meta_signal",
                              "anomalies_braila_pressure5771_meta_signal",
                              "anomalies_braila_pressure5772_meta_signal",
                              "anomalies_braila_pressure5773_meta_signal"]
ANOMALY_META_SIGNAL_THRESHOLD = 0.1

# Service output json keys
OUTPUT_JSON_TIME_KEY = "timestamp"
OUTPUT_JSON_TIME_PROCESSED_KEY = "timestamp-processed-at"
OUTPUT_JSON_STATUS_KEY = "status"
OUTPUT_JSON_CRITICAL_SENSOR_KEY = "critical-sensor"
OUTPUT_JSON_DEVIATION_KEY = "deviation"
OUTPUT_JSON_METHOD_KEY = "method"
OUTPUT_JSON_EPANET_F_KEY = "epanet-file"

OUTPUT_JSON_NODES_KEY = "data"
OUTPUT_JSON_NODE_NAME_KEY = "node-name"
OUTPUT_JSON_NODE_LAT_KEY = "latitude"
OUTPUT_JSON_NODE_LONG_KEY = "longitude"
OUTPUT_JSON_NODE_GROUP_KEY = "group"

# OUTDATED
#
# The current tuples are made from the mail (Marius)
# almost correct but these were not provided by CUP Braila
# "braila_flow211206H360.csv"] = "748-B"      # ("Apollo", "748-B"), ,
# "braila_flow211106H360.csv"] = "751-B"      # ("GA-Braila", "751-B")
# "braila_flow211306H360.csv"] = "763-B"      # ("RaduNegruMare", "763-B")
# "braila_flow318505H498.csv"] = "760-B"      # ("RaduNegru2", "760-B")

# # Nodes which to keep in the dataframes
# SELECTED_NODES = ["SenzorComunarzi-NatVech", "SenzorCernauti-Sebesului", "SenzorChisinau-Titulescu",
#                   "SenzorComunarzi-castanului", "Jonctiune-3974", "Jonctiune-J-3", "Jonctiune-J-19", "Jonctiune-2749"]
#
# KAFKA_NODES_ORDER = ["Jonctiune-J-3", "Jonctiune-3974", "Jonctiune-2749", "Jonctiune-J-19", "SenzorComunarzi-NatVech",
#                      "SenzorComunarzi-castanului", "SenzorChisinau-Titulescu", "SenzorCernauti-Sebesului"]

# PUMPS_TUPLES = [("braila_flow211206H360.csv", "Apollo"),
#                 ("braila_flow211106H360.csv", "GA-Braila"),
#                 ("braila_flow318505H498.csv", "RaduNegru 2"),
#                 ("braila_flow211306H360.csv", "RaduNegruMare")
#                 ]
