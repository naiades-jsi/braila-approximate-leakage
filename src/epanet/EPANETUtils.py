import copy
import json
import logging

import networkx as nx
import wntr
import pandas as pd
import src.configfile as config
from pyproj import Transformer


class EPANETUtils:
    """
    This class implements methods not yet available in EPANET/WNTR for use on the NAIADES project.
    The main objective of this class is to provide easy and simple to use methods for simulation and visualization of
    leakages.

    Methods of the class can work with the EPANET water network model that was provided to but some can also work with
    other models without instantiating them from the file. But can for example use a water network model that was modified
    already with one of the class functions.

    Class for now contains 3 class variables with aren't meant to be touched.
    LEAKED_PATTERN - is the name which will be given to the newly generated pattern which is supposed to
    represent a leakage.
    SECONDS_IN_HOUR - The name explains it self. This variable is used for converting data to hour timestamp
    METERS_TO_BARS_COEF - This is the coefficient which will be used for converting pressure data in meters to bars.
    """
    LEAKED_PATTERN = "Leaked-Pattern"
    SECONDS_IN_HOUR = 3600
    METERS_TO_BARS_COEF = 0.09804139432

    def __init__(self, path_to_epanet_input_file, simulation_mode="PDD"):
        """
        The constructor takes in the path to the EPANET file and also simulation mode.
        Class saves the input file name if there will be a need to use it in the future, then it reads from the
        epanet ".inp" file and creates a Python water network model.
        After this it also sets the hydraulics demand model which can either DD or PDD. One thing to note here is that
        usually only INP files made with EPANET 2.2 version or later support PDD demand model.

        :param path_to_epanet_input_file:   String - Path to the INP file, can either be absolute or relative.
        :param simulation_mode:     String - Simulation model can either be DD or PDD
        """
        if not str.endswith(path_to_epanet_input_file, ".inp"):
            raise Exception("Input file must be in INP format !!")

        if simulation_mode != "PDD" and simulation_mode != "DD":
            raise Exception("The simulation is only available in PDD and DD modes. Your input was ", simulation_mode)

        self.input_file_name = path_to_epanet_input_file
        self.water_network_model = wntr.network.WaterNetworkModel(path_to_epanet_input_file)
        self.water_network_model.options.hydraulic.demand_model = simulation_mode
        # it has to be 23 hours because it already includes time 0 meaning there are actually 24 hours
        self.water_network_model.options.time.duration = 23 * self.SECONDS_IN_HOUR

    def get_original_water_network_model(self):
        """
        Method returns a copy of the original water network model. This is very useful when we want to keep the
        original model as the class variable but we also want to change the model. So with this function we retrieve
        a deep copy of the Python water network model.
        :return:
        """
        return copy.deepcopy(self.water_network_model)

    def run_simulation(self, water_network_model=None):
        """
        Runs the epanet simulation on the water model of the instance or the model that was passed to it.

        :return: Returns the wntr.sim.results.SimulationResults object. Important properties of the object are
        SimulationResults.link and SimulationResults.node with which we can access the pressures/flowrate of the
        simulation.
        """
        if water_network_model is None:
            water_network_model = self.get_original_water_network_model()

        sim = wntr.sim.EpanetSimulator(water_network_model)
        simulation_results = sim.run_sim(version=2.2)

        return simulation_results

    def add_json_ending_if_not_in_file_name(self, file_name):
        """
        This method is meant for internal class use.
        Adds a ".json" ending to the file if it was not provided in the file name.
        :param file_name:   String - Name of the file.
        :return:    String - Name of the file with added JSON ending if it didn't already have it.
        """
        new_file_name = file_name
        if not str.endswith(new_file_name, ".json"):
            new_file_name = new_file_name + ".json"
        return new_file_name

    def get_pipe_length(self, pipe=None, water_network_model=None):
        """
        Methods gets the lengths in a water model and return all lengths if no pipe was provided or just the length of
        one pipe.

        :param pipe: The name of the pipe of which we would like to get the length.
        :param water_network_model: Water network model in which to search for the length of the pipe.
        :return: Returns the length of the pipe in float.
        """
        if water_network_model is None:
            water_network_model = self.get_original_water_network_model()

        all_objects_with_lengths = water_network_model.query_link_attribute('length')
        if pipe is not None:
            # Returns an int which means meters
            return all_objects_with_lengths.loc[pipe]

        # Returns a pandas series with all objects and their lengths
        return all_objects_with_lengths

    def generate_network_json(self, water_network_model=None, file_name="water_network.json"):
        """
        Generates all the data needed to visualize a EPANET/WNTR network.
        Data is then written in the format like this:
        main_node_name: {
            "x": 730350.55,
            "y": 420432.79,
            "connections": {
                "sub_node_name_1": "name_of_the_pipe_to_that_node",
                "sub_node_name_2": "name_of_the_pipe_to_that_node",
                .....
            }
        },
        {...},
        {...}

        The method by default uses the network model which was provided in the constructor of the class but it can also
        use any other model if specified.

        If no file name is provided the method will use the default file name which is "water_network.json".
        :param water_network_model: WNTR water network model to use, if none is provided the default model is used.
        :param file_name:   Name of the file we want to store the data in.
        :return:    The method doesn't return nothing.
        """
        if water_network_model is None:
            water_network_model = self.get_original_water_network_model()

        networkx_graph = water_network_model.get_graph()
        dict_of_connections = nx.to_dict_of_lists(nx.Graph(networkx_graph))

        # Filling the list of links, pipe name is key and the value is an array of the two nodes connected by it
        dict_of_links = dict()
        for node in dict_of_connections:
            list_of_links = water_network_model.get_links_for_node(node)
            for link in list_of_links:
                if link not in dict_of_links:
                    dict_of_links[link] = []
                dict_of_links[link].append(node)

        # The main dict which will be returned in the specified format
        node_dict = dict()
        for node in networkx_graph.nodes():
            x, y = networkx_graph.nodes[node]['pos']
            node_dict[node] = dict()
            node_dict[node]["x"] = x
            node_dict[node]["y"] = y
            node_dict[node]["connections"] = dict()

            for sub_node in dict_of_connections[node]:
                for pipe in dict_of_links:
                    if node in dict_of_links[pipe] and sub_node in dict_of_links[pipe]:
                        node_dict[node]["connections"][sub_node] = pipe
                        # Break because there is usually just one connection between two nodes
                        # TODO check if this works if two nodes are connected with more than one pipe
                        break

        # Writing to the file
        with open(self.add_json_ending_if_not_in_file_name(file_name), 'w', encoding='utf-8') as outfile:
            json.dump(node_dict, outfile, indent=4)

    def generate_network_json_in_wgs84(self, water_network_model=None, file_name="water_network.json"):
        """
        Generates all real coordinates for nodes from EPANET/WNTR network.
        Data is then written in the format like this:
        node_name: {
            "x": 730350.55,
            "y": 420432.79
        },
        {...},
        {...}

        The method by default uses the network model which was provided in the constructor of the class but it can also
        use any other model if specified.

        If no file name is provided the method will use the default file name which is "water_network.json".
        :param water_network_model: WNTR water network model to use, if none is provided the default model is used.
        :param file_name:   Name of the file we want to store the data in.
        """
        if water_network_model is None:
            water_network_model = self.get_original_water_network_model()

        networkx_graph = water_network_model.get_graph()
        dict_of_connections = nx.to_dict_of_lists(nx.Graph(networkx_graph))

        # Filling the list of links, pipe name is key and the value is an array of the two nodes connected by it
        dict_of_links = dict()
        for node in dict_of_connections:
            list_of_links = water_network_model.get_links_for_node(node)
            for link in list_of_links:
                if link not in dict_of_links:
                    dict_of_links[link] = []
                dict_of_links[link].append(node)

        # The main dict which will be returned in the specified format
        node_dict = dict()
        transformer = Transformer.from_crs("epsg:3844", "WGS84")
        for node in networkx_graph.nodes():
            x, y = networkx_graph.nodes[node]['pos']
            node_dict[node] = dict()

            lat, lon = transformer.transform(y, x)
            node_dict[node]["latitude"] = lat
            node_dict[node]["longitude"] = lon

        # Writing to the file
        with open(self.add_json_ending_if_not_in_file_name(file_name), 'w', encoding='utf-8') as outfile:
            json.dump(node_dict, outfile, indent=4)

    def generate_pressures_at_nodes(self, water_network_model=None, selected_nodes=None, file_name=None,
                                    to_hours_round=False, to_bars=False):
        """
        Method takes in only optional parameters.

        Method will run the simulation of the provided network model and then retrieve pressures from all the nodes
        and put them in a DataFrame. If to_hours is set to true the method will normalize index of the DataFrame which
        is in seconds to hours.
        If to file argument is provided the method will also save the result in the file with the name that was provided.

        :param water_network_model: WNTR model - If none is provided the model of the class will be used.
        :param selected_nodes: Array of string - If this is provided the DataFrame/file will contain only the data about
        the provided nodes.
        :param file_name: String - Name of the file can be with a JSON ending or not.
        :param to_hours_round: Boolean - By default is False, if provided the index of the DataFrame will be converted to
        hours.
        :param to_bars: Boolean - By default is False, if provided all the pressure data values will be converted from
        m to bars.
        :return: DataFrame of the simulation results. Indexes are seconds or hours depending on the parameters, column
        is the node name and the values are pressures either in meter or bars is so specified.
        """
        if water_network_model is None:
            # if no water model is provided the default unmodified instance will be returned
            water_network_model = self.get_original_water_network_model()

        simulation_results = self.run_simulation(water_network_model)
        df_simulations_results = simulation_results.node["pressure"]

        if to_hours_round:
            df_simulations_results.index.name = "Hour of the day"
            index_list = list(df_simulations_results.index / self.SECONDS_IN_HOUR)
            df_simulations_results.index = [round(i) for i in index_list]
        else:
            df_simulations_results.index.name = "Second of the day"

        if selected_nodes is not None:
            if len(selected_nodes) > 0:
                df_simulations_results = df_simulations_results[selected_nodes]
            else:
                raise Exception("The input array of selected nodes must be of length at least 1 !")

        if to_bars:
            df_simulations_results = df_simulations_results\
                                         .select_dtypes(exclude=['object', 'datetime']) * self.METERS_TO_BARS_COEF

        if file_name is not None:
            df_simulations_results.to_json(self.add_json_ending_if_not_in_file_name(file_name), indent=4)

        return df_simulations_results

    def generate_flowrate_on_pipes(self, water_network_model=None, selected_pipes=None, file_name=None, to_hours=True):
        """
        Method takes in only optional parameters.

        Method will run the simulation of the provided network model and then retrieve flow rates from all the pipes
        and put them in a DataFrame. If to_hours is set to true the method will normalize index of the DataFrame which
        is in seconds to hours.
        If to file argument is provided the method will also save the result in the file with the name that was provided.

        :param water_network_model: WNTR model - If none is provided the model of the class will be used.
        :param selected_pipes: Array of string - If this is provided the DataFrame/file will contain only the data about
        the provided pipes.
        :param file_name:   String - Name of the file can be with a JSON ending or not.
        :param to_hours:    Boolean - By default is False, if provided the index of the DataFrame will be converted to hours.
        :return: DataFrame of the simulation results. Indexes are seconds or hours depending on the parameters, column is
        the pipe name and the values are flow rates in m^3/s
        """
        if water_network_model is None:
            water_network_model = self.get_original_water_network_model()

        simulation_results = self.run_simulation(water_network_model)
        df_simulations_results = simulation_results.link["flowrate"]

        if to_hours:
            df_simulations_results.index.name = "Hour of the day"
            df_simulations_results.index = df_simulations_results.index / self.SECONDS_IN_HOUR
        else:
            df_simulations_results.index.name = "Second of the day"

        if selected_pipes is not None:
            if len(selected_pipes) > 0:
                df_simulations_results = df_simulations_results[selected_pipes]
            else:
                raise Exception("The input array of selected nodes must be of length at least 1 !")

        if file_name is not None:
            df_simulations_results.to_json(self.add_json_ending_if_not_in_file_name(file_name), indent=4)

        return df_simulations_results

    def add_leakage_on_node_and_run_simulation(self, node, leak_to_simulate, water_network_model=None):
        """
        Method takes in the node on which we want to add a leak and the amount of leakage in m^3/s.
        Leakage is simulated as extra constant demand.

        :param node: The name of the node as a string. Example: "751-B"
        :param leak_to_simulate: Amount of leak in m^3/s. Example: 0.003 TODO change this to LPS
        :param water_network_model: Water network model on which we would like to add a leak. If none is provided the
        model of the instance will be taken.
        :return: Returns a dataframe with pressures (in m) on all nodes.
        """
        if leak_to_simulate < 0:
            raise Exception("Leak to simulate must be bigger or equal to 0, please readjust the parameter: " + leak_to_simulate)

        if water_network_model is None:
            # if no water model is provided the default unmodified instance will be returned
            water_network_model = self.get_original_water_network_model()

        # Needs to be a 24 value array
        leaked_values_24h_array = [leak_to_simulate for _ in range(0, 24 + 1)]

        # Adding the pattern to the water network instance
        water_network_model.add_pattern(name=self.LEAKED_PATTERN, pattern=leaked_values_24h_array)
        # Adding additional demand on the specified node, base=1 means that the pattern is not multiplied
        water_network_model.get_node(node).add_demand(base=1, pattern_name=self.LEAKED_PATTERN)

        simulation_results = self.run_simulation(water_network_model=water_network_model)
        pressure_df = simulation_results.node["pressure"].rename_axis(f"Node_ {node}, {leak_to_simulate:.4f} m3/s", axis=1)

        return pressure_df

    def run_leakage_scenario(self, leaks_arr=None, generate_diff_dict=False, retrieve_specific_nodes_arr=None):
        """
        Methods simulates leaks on every node in the network
        # TODO add documentation - this is won't be used since we get this data from DELFT
        Generate a JSON with structure:
        1 LiterPerSecond is 0.001 m^3/s
            Leakednode:
                leak50:
                    all_nodes (including this one): pressure
                leak100:
                    all_nodes (including this one): pressure
                leak150:
                    all_nodes (including this one): pressure
        4. Make an IF flag to generate from this JSON or DataFrame a correlation matrix

        :param leaks_arr:
        :param generate_diff_dict:
        :param retrieve_specific_nodes_arr:
        :return:
        """

        diff_dict = None

        if leaks_arr is None:
            leaks_arr = [0.003, 0.006, 0.012]
        leak_names = [str(leak) + "m3/s - Leak" for leak in leaks_arr]

        water_network_model = self.get_original_water_network_model()
        # TODO add pattern here ?

        # getting all the nodes names
        nodes_list = water_network_model.junction_name_list
        if retrieve_specific_nodes_arr is None or len(retrieve_specific_nodes_arr) < 1:
            retrieve_specific_nodes_arr = nodes_list

        node_leak_nodes_dict = dict()
        for node in nodes_list:
            node_leak_nodes_dict[node] = dict()
            for leak, leak_name in zip(leaks_arr, leak_names):
                node_leak_nodes_dict[node][leak_name] = self.add_leakage_on_node_and_run_simulation(node, leak).to_dict()

        if retrieve_specific_nodes_arr is not None and len(retrieve_specific_nodes_arr) > 0:
            for node_key in node_leak_nodes_dict.keys():
                for leak_name in leak_names:
                    temp_df = pd.DataFrame.from_dict(node_leak_nodes_dict[node_key][leak_name])
                    node_leak_nodes_dict[node_key][leak_name] = temp_df[list(retrieve_specific_nodes_arr)].to_dict()

        if generate_diff_dict:
            diff_dict = self.generate_difference_dict(node_leak_nodes_dict, leak_names)

            for node_key in diff_dict.keys():
                for leak_name in leak_names:
                    temp_df = pd.DataFrame.from_dict(diff_dict[node_key][leak_name])
                    diff_dict[node_key][leak_name] = temp_df[list(retrieve_specific_nodes_arr)].to_dict()

        return node_leak_nodes_dict, diff_dict

    def generate_difference_dict(self, leak_data_dict, leak_names):
        """
        Generates a dictionary which contains the difference between real values and the simulated ones.

        :param leak_data_dict: TODO add documentation - this is won't be used since we get this data from DELFT
        :param leak_names:
        :return:
        """
        original_df = self.generate_pressures_at_nodes()
        all_node_names = leak_data_dict.keys()
        # TODO get this from DataFrame columns ? - this is won't be used since we get this data from DELFT
        leaks_df = pd.DataFrame.from_dict(leak_data_dict)

        node_leaks_diff_dict = dict()
        for node_name in all_node_names:
            node_leaks_diff_dict[node_name] = dict()
            for leak_name in leak_names:
                temp_dict = leaks_df.loc[leak_name, node_name]
                node_leaks_diff_dict[node_name][leak_name] = self.generate_difference_between_two_dataframes\
                    (temp_dict, original_df=original_df).to_dict()

        return node_leaks_diff_dict

    def generate_difference_between_two_dataframes(self, leak_data, leak_data_is_dict=True, original_df=None):
        """
        Subtracts a dataframe (or dictionary) from the given dataframe or generates a dataframe with no leaks from
        generate_pressures_at_nodes method(). Dataframes are then subtracted.

        :param leak_data: Dataframe or dictionary with indexes as time and a single column
        :param leak_data_is_dict: Boolean, True if the leak_data is a dictionary.
        :param original_df: Dataframe, which we want to subtract from leak data.
        :return: Dataframe.
        """
        if original_df is None:
            original_df = self.generate_pressures_at_nodes()

        if leak_data_is_dict:
            leak_data = pd.DataFrame.from_dict(leak_data)   # Transforming data into a dataframe

        return leak_data.subtract(original_df)

    def get_node_base_demand(self, node_name, water_network_model=None):
        """
        Method returns the base demand of the node which was provided to it. This node name must be in the water network
        model otherwise an exception will be thrown.
        :param node_name:   String - Name of the node for which we want to retrieve the base demand.
        :param water_network_model: WNTR water object model which should contain the node.
        :return:    Return the node base demand in m/s
        """
        if water_network_model is None:
            water_network_model = self.get_original_water_network_model()

        return water_network_model.get_node(node_name).base_demand

    def generate_node_array_with_meta_data(self, groups_dict):
        """
        Method generates an array of dictionaries (json), each dictionary contains data about the specific node.
        Meta data (longitude, latitude) is retrieved from the wntr-epanet water network model.

        :param groups_dict: Dictionary which contains groups as main keys and an array of nodes as values. Can be
        acquired from DivergenceMatrixProcessor.get_affected_nodes_groups() and the data format can be found in
        generate_groups_dict method.
        :return: Returns an array of dictionaries (json), each dictionary contains data about the specific node.
        """
        transformer = Transformer.from_crs("epsg:3844", "WGS84")
        networkx_graph = self.water_network_model.get_graph()
        groups_arr = []

        for group_num in groups_dict.keys():
            for node_name in groups_dict[group_num]:
                if node_name not in networkx_graph.nodes:
                    # TODO better way to handle this, custom EPANET file everytime?
                    continue
                    # raise Exception("Node name {} is not in the network!".format(node_name))
                # TODO add logic that if the node is not found its name will be still be included in the output
                x, y = networkx_graph.nodes[node_name]["pos"]
                lat, lon = transformer.transform(y, x)

                current_node_json = {
                    config.OUTPUT_JSON_NODE_NAME_KEY: node_name,
                    config.OUTPUT_JSON_NODE_LAT_KEY: lat,
                    config.OUTPUT_JSON_NODE_LONG_KEY: lon,
                    config.OUTPUT_JSON_NODE_GROUP_KEY: int(group_num)
                }
                groups_arr.append(current_node_json)
        return groups_arr

    def generate_nan_sensors_meta_data(self, nan_sensors):
        """
        Method finds meta data about the sensors and returns an array of dictionaries (json) containg this data.

        :param nan_sensors: Array of strings, each string is the name of a sensor.
        :return: Returns an array of dictionaries (json), each dictionary contains data about the specific sensor.
        """
        transformer = Transformer.from_crs("epsg:3844", "WGS84")
        networkx_graph = self.water_network_model.get_graph()
        groups_arr = []

        for sensor_name in nan_sensors:
            if sensor_name not in networkx_graph.nodes:
                # raise Exception("Sensor name {} is not in the network!".format(sensor_name))
                logging.warning("Sensor name {} is not in the network!".format(sensor_name))

            x, y = networkx_graph.nodes[sensor_name]["pos"]
            lat, lon = transformer.transform(y, x)

            current_node_json = {
                config.OUTPUT_JSON_NODE_NAME_KEY: sensor_name,
                config.OUTPUT_JSON_NODE_LAT_KEY: lat,
                config.OUTPUT_JSON_NODE_LONG_KEY: lon
            }
            groups_arr.append(current_node_json)
        return groups_arr