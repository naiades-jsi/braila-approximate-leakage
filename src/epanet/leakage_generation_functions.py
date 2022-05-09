import copy
import pickle
import time

import numpy as np
import pandas as pd
import wntr


SECONDS_IN_HOUR = 3600
NUMBER_OF_DAYS = 1


def one_leak_node_file_generation(epanet_file_name, min_leak=0.5, max_leak=3.0):
    # TODO replace prints with logging
    # TODO add multithreading
    # TODO add documentation to all functions
    print(f"Epanet file name: {epanet_file_name}")
    demand_model_method = "PDD"
    leak_flow_step = 0.01
    leak_flow_threshold = 0.5
    start_time = time.time()

    water_network_model = wntr.network.WaterNetworkModel(epanet_file_name)
    simulation_results = run_simulation(water_network_model)

    base_demands_arr, base_demands_mean, node_names_arr = get_node_base_demands_and_names(water_network_model)

    # Prepare general water network model for the leak simulation
    water_network_model.options.time.duration = NUMBER_OF_DAYS * 24 * SECONDS_IN_HOUR  # Time of simulation
    water_network_model.options.hydraulic.demand_model = demand_model_method

    # Run = 1 # 1 m³/s = 1000 L/s,  L/s = m³/s * 1000
    min_max_leak_arr = [round(i, 3) for i in np.arange(min_leak, max_leak + leak_flow_step, leak_flow_step)]

    for index, curr_leak in enumerate(min_max_leak_arr[:-1]):
        minimum_leak = min_max_leak_arr[index]
        maximum_leak = min_max_leak_arr[index + 1]

        print(f"Current leak: {minimum_leak} - {maximum_leak}")
        # TODO testing, + optimization, leak steps should be generate at only one place
        run_one_leak_per_node_simulation(run_id=1,
                                         wn_model=water_network_model,
                                         node_names_arr=node_names_arr,
                                         base_sim_results=simulation_results,
                                         minimum_leak=minimum_leak,
                                         maximum_leak=maximum_leak,
                                         leak_flow_threshold=leak_flow_threshold,
                                         output_file_name="testing.pkl")
    print(f"Ended execution in {time.time() - start_time} seconds")


def run_one_leak_per_node_simulation(run_id, wn_model, node_names_arr, base_sim_results, minimum_leak, maximum_leak,
                                     leak_flow_threshold, output_file_name, include_extra_info=False):
    round_leak_to = 4
    start_time = time.time()

    # copy original data so it will not be modified
    water_network_model_f = copy.deepcopy(wn_model)
    org_simulation_results = copy.deepcopy(base_sim_results)

    steps_in_a_day = int(water_network_model_f.options.time.duration / water_network_model_f.options.time.hydraulic_timestep)
    last_hour_seconds = steps_in_a_day * SECONDS_IN_HOUR

    # df of pressures, with timestamps as index and node names as columns
    base_pressures_df = org_simulation_results.node["pressure"].loc[1:last_hour_seconds, node_names_arr]

    print(f"Number of nodes {len(node_names_arr)}")
    print(f"leaks {minimum_leak}, {maximum_leak}")
    # added rounding to prevent floating point errors
    leak_amounts_arr = [round(i, 3) for i in np.arange(minimum_leak, maximum_leak, 0.001)]

    len_leak_amounts_arr = len(leak_amounts_arr)
    # TODO optimize run time
    for curr_node_name in node_names_arr[:2]:
        start2 = time.time()

        for index, curr_leak_flow in enumerate(leak_amounts_arr):
            curr_axis_name = f"Node_{curr_node_name}, {str(round(curr_leak_flow, round_leak_to))}LPS"

            # TODO take care of the units m3/s -> liters/s etc., is it ok now?
            # Converting from LPS to m3/s
            lps_leak = round(curr_leak_flow / 1000, 6)
            curr_leak_flow_arr = [lps_leak] * (24 * NUMBER_OF_DAYS + 1)
            temp_water_network_model = copy.deepcopy(water_network_model_f)

            # adding leak to existing model
            temp_water_network_model.add_pattern(name="New", pattern=curr_leak_flow_arr)  # Add New Patter To the model
            temp_water_network_model.get_node(curr_node_name).add_demand(base=1, pattern_name="New")  # Add leakflow

            sim_results_with_leak = run_simulation(temp_water_network_model).node["pressure"].loc[1:last_hour_seconds, node_names_arr]
            # renaming axis to match node that has the current leak
            sim_results_with_leak = sim_results_with_leak.rename_axis(curr_axis_name, axis=1)

            divergence_df = base_pressures_df.sub(sim_results_with_leak[node_names_arr], fill_value=0) \
                .abs().rename_axis(curr_axis_name, axis=1)

            used_leak_flows_df = pd.DataFrame([k * 1000 for k in curr_leak_flow_arr[1:]], columns=["LeakFlow"],
                                              index=list(range(SECONDS_IN_HOUR, last_hour_seconds + 3600, 3600))) \
                .rename_axis(curr_axis_name, axis=1)
            # TODO remove this
            # used_leak_flows_df = pd.DataFrame([curr_leak_flow], columns=["LeakFlow"]).rename_axis(curr_axis_name, axis=1)

            # prepare dictionary for saving, TODO restructure?
            main_data_dict = {
                "LPM": sim_results_with_leak,
                "DM": divergence_df,
                "LM": used_leak_flows_df,
                "Meta": {"Leakmin": minimum_leak,
                         "Leakmax": maximum_leak,
                         "Run": run_id,
                         "Run Time": time.time() - start_time
                         }
            }
            if include_extra_info:
                leak_i_time_arr, water_loss_arr = \
                    get_leak_time_identification_and_water_loss_from_df(divergence_df, used_leak_flows_df,
                                                                        leak_flow_threshold)
                main_data_dict["TM_l"] = leak_i_time_arr
                main_data_dict["WLM"] = water_loss_arr

            # saving to file
            append_dict_to_file(main_data_dict, out_f_name=output_file_name)
            print(f"Index = {index}/{len_leak_amounts_arr} and value {curr_leak_flow}, LeakNode={curr_node_name}, "
                  f"{curr_axis_name}")
        print("____**____")
        print(f"All leaks nodes {curr_node_name} Time= {time.time() - start2}")


def append_dict_to_file(main_data_dict, out_f_name):
    # TODO try different data formats, parquet etc
    if not out_f_name.endswith(".pkl"):
        raise Exception("Output file must be a .pkl file")
    with open(out_f_name, "ab") as file:
        pickle.dump(main_data_dict, file)


def get_leak_time_identification_and_water_loss_from_df(divergence_df, leak_flows_df, leak_flow_threshold):
    """
    TODO

    # This determines which nodes are over the threshold for leak and what is their commutative leak flow
    :param divergence_df:
    :param leak_flows_df:
    :param leak_flow_threshold:
    :return:
    """
    leak_identified_time_arr = []  # time when the leak was identified
    water_loss_arr = []  # Water loss arr (L/s), how much water is wasted
    seconds_in_hour = 3600

    for node_name_i in divergence_df.columns:
        comm_leak_flow = 0
        leak_detected_hour = 0
        for hour_in_day in range(seconds_in_hour, 25 * seconds_in_hour, seconds_in_hour):
            comm_leak_flow += leak_flows_df["LeakFlow"][hour_in_day] * 3600

            if divergence_df.loc[hour_in_day, node_name_i] > leak_flow_threshold:
                leak_detected_hour = hour_in_day
                break
        leak_identified_time_arr.append(leak_detected_hour)
        water_loss_arr.append(round(comm_leak_flow, 4))

    return leak_identified_time_arr, water_loss_arr


def run_simulation(wntr_network_instance, file_prefix="temp_"):
    """
    Runs the epanet simulation on the water model of the instance or the model that was passed to it.

    :return: Returns the wntr.sim.results.SimulationResults object. Important properties of the object are
    SimulationResults.link and SimulationResults.node with which we can access the pressures/flow rate of the
    simulation.
    """
    sim = wntr.sim.EpanetSimulator(wntr_network_instance)
    simulation_results = sim.run_sim(version=2.2, file_prefix=file_prefix)

    return simulation_results


def get_node_base_demands_and_names(epanet_network, junction_name_arr=None):
    """
    Function loops through all the nodes in the network or through the list of junction names that was passed to it.
    It then generates a list of base demands for each node and a list of node names based on the junction names.
    Demands are given in m3/s.

    :param epanet_network: wntr.network.WaterNetworkModel object. The network that we want to get the base demands for.
    :param junction_name_arr: list of strings. The junction names that we want to get the base demands for.
    :return: (list of floats, a float, list of strings). The first element is the base demands for each node, second
    is the average base demand, and the third is the list of node names.
    """
    epanet_net_instance = epanet_network
    base_demands_arr = []
    # node names could be directly copied from junction_name_arr, but due to including only nodes with base demands
    # higher than zero they are kept
    node_names_arr = []

    if junction_name_arr is None:
        junction_name_arr = epanet_net_instance.junction_name_list

    for junction_name in junction_name_arr:
        node_instance = epanet_net_instance.get_node(junction_name)
        if node_instance.base_demand > 0:
            base_demands_arr.append(node_instance.base_demand)
            node_names_arr.append(node_instance.name)
        else:
            base_demands_arr.append(0)

    base_demands_arr = np.array(base_demands_arr)
    base_demands_mean = base_demands_arr.mean()

    return base_demands_arr, base_demands_mean, node_names_arr


if __name__ == "__main__":
    one_leak_node_file_generation("../../data/epanet_networks/Braila_V2022_2_2.inp")
