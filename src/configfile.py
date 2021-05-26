
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
PUMP_FILES = ["braila_flow211206H360.csv", "braila_flow211106H360.csv", "braila_flow211306H360.csv",
              "braila_flow318505H498.csv"]


# Nodes which to keep in the dataframes
SELECTED_NODES = ["SenzorComunarzi-NatVech", "SenzorCernauti-Sebesului", "SenzorChisinau-Titulescu",
                  "SenzorComunarzi-castanului", "751-B", "763-B", "748-B", "760-B"]