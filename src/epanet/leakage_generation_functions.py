import copy
import pickle
import time

import numpy as np
import pandas as pd
import wntr


def one_leak_node_file_generation(epanet_file_name):
    # TODO, replace prints with logging
    print(f"Epanet file name: {epanet_file_name}")
    water_network_model = wntr.network.WaterNetworkModel(epanet_file_name)
    base_demands_arr, base_demands_mean, node_names_arr = get_node_base_demands_and_names(water_network_model)

    # saving current epanet file to a temporary .pkl file
    no_leak_file_name = "no_leak_network.pkl"
    with open(no_leak_file_name, "wb") as f:
        pickle.dump(water_network_model, f)

    simulation_results = run_simulation(water_network_model)
    # saving the results of the simulation to a temporary .pkl file
    sim_res_file_name = "simulation_results.pkl"
    with open(sim_res_file_name, "wb") as f:
        pickle.dump(simulation_results, f)

    # TODO new function for the code below
    # RunCycle2_v2(Run=RunId, Threshold=0.5, DirPath=DirInput, Lnz=Lnz, ModName=Mod)
    # def RunCycle_v2(Run=1, Threshold=0.5, DirPath='test', Lnz='', ModName='NotSent'):
    Run = 1
    # 1 m³/s = 1000 L/s
    # L/s = m³/s * 1000
    # TODO refactor this to variable parameters
    minimum_leak = Run / 10
    maximum_leak = (Run + 1) / 10

    Threshold = 0.5
    temp_epanet_file_prefix = f"temp_{epanet_file_name}"
    ModName = 'NotSent'
    demand_model_method = "PDD"
    number_of_days = 1
    start_time = time.time()

    with open(no_leak_file_name, 'rb') as f:
        water_network_model_f = pickle.load(f)

    with open(sim_res_file_name, 'rb') as f:
        org_simulation_results = pickle.load(f)

    water_network_model_f.options.time.duration = number_of_days * 24 * 3600  # Time of simulation
    water_network_model_f.options.hydraulic.demand_model = demand_model_method

    base_leak_instance = copy.deepcopy(water_network_model_f)
    steps_in_a_day = int(water_network_model_f.options.time.duration / water_network_model_f.options.time.hydraulic_timestep)

    # df of pressures, with timestamps as index and node names as columns
    base_pressures_df = org_simulation_results.node["pressure"].loc[1:steps_in_a_day * 3600, water_network_model_f.junction_name_list]
    base_pressures_df = base_pressures_df[node_names_arr]

    leak_pressure_matrix = []
    leak_matrix = []
    divergence_matrix = []
    main_data_dict = {}

    # Leak_Nodes = node_names_arr
    # Sensor_Nodes = node_names_arr
    print(f"Number of nodes {len(node_names_arr)}")

    print(f"leaks {minimum_leak}, {maximum_leak}")
    # added rounding to prevent floating point errors
    leak_amounts_arr = [round(i, 3) for i in np.arange(minimum_leak, maximum_leak, 0.001)]

    to_row = steps_in_a_day * 3600
    len_leak_amounts_arr = len(leak_amounts_arr)
    round_leak_to = 4
    # TODO optimize run time
    for curr_node_name in node_names_arr:
        start2 = time.time()

        for index, curr_leak_flow in enumerate(leak_amounts_arr):
            curr_axis_name = f"Node_{curr_node_name}, {str(round(curr_leak_flow, round_leak_to))}LPS"

            # TODO take care of the units m3/s -> liters/s etc., is it ok now?
            # Converting from LPS to m3/s
            lps_leak = round(curr_leak_flow / 1000, 6)
            curr_leak_flow_arr = [lps_leak] * (24 * number_of_days + 1)
            water_network_model_f = copy.deepcopy(base_leak_instance)

            # adding leak to existing model
            water_network_model_f.add_pattern(name="New", pattern=curr_leak_flow_arr)  # Add New Patter To the model
            water_network_model_f.get_node(curr_node_name).add_demand(base=1, pattern_name="New")  # Add leakflow

            sim_results_with_leak = run_simulation(water_network_model_f).node["pressure"].loc[1:to_row, node_names_arr]
            # renaming axis to match node that has the current leak
            sim_results_with_leak = sim_results_with_leak.rename_axis(curr_axis_name, axis=1)

            temp_divergence_df = base_pressures_df.sub(sim_results_with_leak[node_names_arr], fill_value=0)\
                .abs().rename_axis(curr_axis_name, axis=1)

            # TODO remove this
            # used_leak_flows_df = pd.DataFrame([k * 1000 for k in curr_leak_flow_arr[1:]], columns=["LeakFlow"],
            #                                   index=list(range(3600, to_row + 3600, 3600)))\
            #     .rename_axis(curr_axis_name, axis=1)
            used_leak_flows_df = pd.DataFrame([curr_leak_flow], columns=["LeakFlow"]).rename_axis(curr_axis_name, axis=1)

            # saving to dictionary -> TODO refactor to directly append to file to save memory
            leak_pressure_matrix.append(sim_results_with_leak)
            divergence_matrix.append(temp_divergence_df)
            leak_matrix.append(used_leak_flows_df)

            # print(f"Index = {index}/{len_leak_amounts_arr} and value {curr_leak_flow}, LeakNode={curr_node_name}, actual_l_: {curr_leak_flow_arr[0]}, {curr_axis_name}")
        print("____**____")
        print(f"All leaks nodes {curr_node_name} Time= {time.time() - start2}")

    main_data_dict["LPM"] = leak_pressure_matrix
    main_data_dict["DM"] = divergence_matrix
    main_data_dict["LM"] = leak_matrix

    print(f"Finish Time 1= {time.time() - start_time}")
    leak_identified_time = []  # time when the leak was identified
    water_loss_matrix = []  # Water loss Matrix (L/s), how much water is wasted

    start_3 = time.time()
    # TODO this can be moved in the upper for, or calculated without for loop
    for curr_node_name in range(len(divergence_matrix)):
        TMtemp = set()
        WLMtemp = set()
        for j in node_names_arr:
            WLMtemp2 = []
            for curr_leak_flow in range(len(divergence_matrix[0])):
                if divergence_matrix[curr_node_name][j][(curr_leak_flow + 1) * 3600] <= Threshold:
                    WLMtemp2.append(leak_matrix[curr_node_name].LeakFlow[(curr_leak_flow + 1) * 3600] * 3600)
                else:
                    WLMtemp2.append(leak_matrix[curr_node_name].LeakFlow[(curr_leak_flow + 1) * 3600] * 3600)
                    break
            TMtemp.add(len(divergence_matrix[0]) + 1)
            WLMtemp.add(sum(WLMtemp2))
        leak_identified_time.append(list(TMtemp))
        water_loss_matrix.append(list(WLMtemp))
    print(f"Finish Time 2= {time.time() - start_3}")

    main_data_dict["LPM"] = leak_pressure_matrix
    main_data_dict["DM"] = divergence_matrix
    main_data_dict["LM"] = leak_matrix
    main_data_dict["Meta"] = {"Leakmin": minimum_leak, "Leakmax": maximum_leak, "Run": Run, "Run Time": time.time() - start_time}
    main_data_dict["TM_l"] = leak_identified_time
    main_data_dict["WLM"] = water_loss_matrix

    # Better way to save the data
    out_f_name = f"/scratch-shared/NAIADES/ijs_simulations_v1/{ModName}/1Leak_{str(Run)}_{ModName}_{str(minimum_leak)}.pkl"
    with open(out_f_name, 'wb') as end_file:
        pickle.dump(main_data_dict, end_file)

    return "success"


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

