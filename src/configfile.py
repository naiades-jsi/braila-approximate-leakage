# Amount with which to run the simulation
LEAK_AMOUNT = 4.0

# If difference between simulated and actual data is bigger than this then alert
PRESSURE_DIFF_THRESHOLD = 1.1

# Data directories
DATA_DIRECTORY = "./../data/"
SENSOR_DIR = DATA_DIRECTORY + "sensor_data/"
DIVERGENCE_DATA_DIR = DATA_DIRECTORY + "divergence_matrix/"
EPANET_NETWORKS = DATA_DIRECTORY + "epanet_networks/"

EPANET_NETWORK_FILE = EPANET_NETWORKS + "RaduNegru24May2021_2.2.inp"
DIVERGENCE_MATRIX_FILE = DIVERGENCE_DATA_DIR + "Divergence_M.pickle"

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
# The current tuples are made from the mail (Marius)
# almost correct but these were not provided by CUP Braila
# "braila_flow211206H360.csv"] = "748-B"      # ("Apollo", "748-B"), ,
# "braila_flow211106H360.csv"] = "751-B"      # ("GA-Braila", "751-B")
# "braila_flow211306H360.csv"] = "763-B"      # ("RaduNegruMare", "763-B")
# "braila_flow318505H498.csv"] = "760-B"      # ("RaduNegru2", "760-B")


# Nodes which to keep in the dataframes
SELECTED_NODES = ["SenzorComunarzi-NatVech", "SenzorCernauti-Sebesului", "SenzorChisinau-Titulescu",
                  "SenzorComunarzi-castanului", "Jonctiune-3974", "Jonctiune-J-3", "Jonctiune-J-19", "Jonctiune-2749"]



# KAFKA related
HOST_AND_PORT = "194.249.231.11:9092"

# Names of the topics that the Kafka consumer should consume
TOPICS = ["measurements_node_braila_pressure5770", "measurements_node_braila_pressure5771",
          "measurements_node_braila_pressure5772", "measurements_node_braila_pressure5773",
          "anomalies_braila_flow_211106H360", "anomalies_braila_flow_211206H360",
          "anomalies_braila_flow_211306H360", "anomalies_braila_flow_318505H498",
          ]
