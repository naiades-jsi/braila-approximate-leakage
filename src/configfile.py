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
# EPANET_NETWORK_FILE = EPANET_NETWORKS + "RaduNegru23July2021_2.2.inp"
DIVERGENCE_MATRIX_FILE = DIVERGENCE_DATA_DIR + "Divergence_M.pickle"

LOG_FILE = "logs/service-log.log"

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

# KAFKA related
HOST_AND_PORT = "194.249.231.11:9092"

TOPIC = "features_braila_leakage_detection"
OUTPUT_TOPIC = "braila_leakage_groups"

# Service output json keys
OUTPUT_JSON_TIME_KEY = "timestamp"
OUTPUT_JSON_TIME_PROCESSED_KEY = "timestamp-processed-at"
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
