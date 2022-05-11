import copy
import logging
import math
import multiprocessing
import os
import pickle
import time
from multiprocessing import Process

import numpy as np
import pandas as pd
import wntr


class EpanetLeakGenerator:
    SECONDS_IN_HOUR = 3600
    NUMBER_OF_DAYS = 1
    DEMAND_MODEL = "PDD"
    TEMPORARY_EPA_FILES_PREFIX = "tmp_file"
    LEAKS_PER_FILE_INTERVAL = 0.2

    def __init__(self, epanet_file_name, number_of_threads, min_leak, max_leak, leak_flow_step, leak_flow_threshold,
                 output_dir, log_file):
        """
        :param number_of_threads: Int. Number of threads to use and which are available.
        :param min_leak: Float. Minimum leak flow in L/s.
        :param max_leak: Float. Maximum leak flow in L/s.
        :param leak_flow_step: Float. Step size for leak flow in L/s.

        :param epanet_file_name:
        :param number_of_threads:
        :param min_leak:
        :param max_leak:
        :param leak_flow_step:
        :param output_dir:
        :param log_file:
        """
        # check if parameters contain correct data
        self.input_arguments_check(epanet_file_name, number_of_threads, min_leak, max_leak, leak_flow_step,
                                   leak_flow_threshold, output_dir,
                                   log_file)
        self.epanet_file_name = epanet_file_name
        self.number_of_threads = number_of_threads
        self.min_leak = min_leak
        self.max_leak = max_leak
        self.leak_flow_step = leak_flow_step
        self.leak_flow_threshold = leak_flow_threshold
        self.output_dir = output_dir
        self.log_file = log_file
        self.main_logger = self.create_logger(log_file)

        # Prepare base model information
        self.water_network_model = wntr.network.WaterNetworkModel(epanet_file_name)

        # these two lines must be run directly after loading the model, no modifications to it!
        self.base_simulation_results = self.run_simulation(self.water_network_model, file_prefix="base_model")
        self.base_demands_arr, self.base_demands_mean, self.node_names_arr = self \
            .get_node_base_demands_and_names(self.water_network_model)

        # Prepare general water network model for the leak simulation
        self.water_network_model.options.time.duration = self.NUMBER_OF_DAYS * 24 * self.SECONDS_IN_HOUR
        self.water_network_model.options.hydraulic.demand_model = self.DEMAND_MODEL

        self.main_logger.info("Initialization completed successfully!")

    def multi_thread_data_generation(self):
        """
        TODO
        :return:
        """
        process_array = []
        self.main_logger.info("Started preparing multi thread processes!")

        leak_thread_arr = self.generate_leaks_arrays()
        self.main_logger.info(f"Leak distribution per thread: {leak_thread_arr}")

        for thread_num, thread_arr in enumerate(leak_thread_arr):
            tmp_process = Process(target=self.single_thread_data_generation,
                                  args=[thread_num + 1, thread_arr])
            process_array.append(tmp_process)

        self.main_logger.info("Executing processes!")
        for process in process_array:
            process.start()

        for process in process_array:
            process.join()

        self.main_logger.info("Ended execution of all threads!")
        self.clear_temporary_epanet_files()

    def single_thread_data_generation(self, thread_num, min_max_leak_arr):
        """
        TODO add description, TODO add documentation, testing
        :param thread_num:
        :param min_max_leak_arr:
        :return:
        """
        start_time = time.time()

        logger_inst = self.create_logger(self.log_file)
        logger_inst.info(f"Started leak simulation on thread {thread_num}")
        print(f"Started leak simulation on thread {thread_num}", flush=True)

        for index, curr_leak in enumerate(min_max_leak_arr[:-1]):
            minimum_leak = min_max_leak_arr[index]
            maximum_leak = min_max_leak_arr[index + 1]

            output_file_name = f"{self.output_dir}/1_leak_{minimum_leak:.3f}-{maximum_leak:.3f}_t_{thread_num}.pkl"
            if os.path.isfile(output_file_name):
                raise FileExistsError(f"Unexpected error, file {output_file_name} already exists!")

            logger_inst.info(
                f"Writing to file: {output_file_name}. On thread {thread_num} with leak {minimum_leak}-{maximum_leak}")
            logger_inst.info(f"Executing thread {thread_num} with leak {minimum_leak} - {maximum_leak}")
            self.run_one_leak_per_node_simulation(run_id=thread_num,
                                                  minimum_leak=minimum_leak,
                                                  maximum_leak=maximum_leak,
                                                  logger_objc=logger_inst,
                                                  output_file_name=output_file_name)

        logger_inst.info(f"Ended execution on thread |{thread_num}| in {time.time() - start_time} seconds")

    def run_one_leak_per_node_simulation(self, run_id, minimum_leak, maximum_leak, logger_objc, output_file_name,
                                         include_extra_info=False):
        round_leak_to = 4
        start_time = time.time()

        # copy original data so it will not be modified
        water_network_model_f = copy.deepcopy(self.water_network_model)
        org_simulation_results = copy.deepcopy(self.base_simulation_results)

        steps_in_a_day = int(
            water_network_model_f.options.time.duration / water_network_model_f.options.time.hydraulic_timestep)
        last_hour_seconds = steps_in_a_day * self.SECONDS_IN_HOUR

        # df of pressures, with timestamps as index and node names as columns
        base_pressures_df = org_simulation_results.node["pressure"].loc[1:last_hour_seconds, self.node_names_arr]

        # creates leaks in given interval, added rounding to prevent floating point errors
        leak_amounts_arr = [round(i, 3) for i in np.arange(minimum_leak, maximum_leak, self.leak_flow_step)]
        print(f"T:{run_id} - Executing simulation for {len(leak_amounts_arr)} leaks between: {minimum_leak}, {maximum_leak}")
        logger_objc.info(f"T:{run_id} - Executing simulation for {len(leak_amounts_arr)} leaks between: "
                         f"{minimum_leak}, {maximum_leak}")

        for curr_node_name in self.node_names_arr:
            start2 = time.time()

            for index, curr_leak_flow in enumerate(leak_amounts_arr):
                curr_axis_name = f"Node_{curr_node_name}, {str(round(curr_leak_flow, round_leak_to))}LPS"

                # TODO take care of the units m3/s -> liters/s etc., is it ok now?
                # Converting from LPS to m3/s
                lps_leak = round(curr_leak_flow / 1000, 6)
                curr_leak_flow_arr = [lps_leak] * (24 * self.NUMBER_OF_DAYS + 1)
                temp_water_network_model = copy.deepcopy(water_network_model_f)

                # adding leak to existing model
                temp_water_network_model.add_pattern(name="New",
                                                     pattern=curr_leak_flow_arr)  # Add New Patter To the model
                temp_water_network_model.get_node(curr_node_name).add_demand(base=1, pattern_name="New")  # Add leakflow

                sim_results_with_leak = \
                    self.run_simulation(temp_water_network_model, file_prefix=f"temp_{run_id}").node[
                        "pressure"].loc[1:last_hour_seconds, self.node_names_arr]
                # renaming axis to match node that has the current leak
                sim_results_with_leak = sim_results_with_leak.rename_axis(curr_axis_name, axis=1)

                divergence_df = base_pressures_df.sub(sim_results_with_leak[self.node_names_arr], fill_value=0) \
                    .abs().rename_axis(curr_axis_name, axis=1)

                # TODO convert in more efficient data structure, no duplicated values etc.
                used_leak_flows_df = pd.DataFrame([k * 1000 for k in curr_leak_flow_arr[1:]], columns=["LeakFlow"],
                                                  index=list(
                                                      range(self.SECONDS_IN_HOUR, last_hour_seconds + 3600, 3600))) \
                    .rename_axis(curr_axis_name, axis=1)

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
                        self.get_leak_time_identification_and_water_loss_from_df(divergence_df, used_leak_flows_df)
                    main_data_dict["TM_l"] = leak_i_time_arr
                    main_data_dict["WLM"] = water_loss_arr

                # saving to file
                self.append_dict_to_file(main_data_dict, out_f_name=output_file_name)
                # print(f"Index = {index + 1}/{len_leak_amounts_arr} and value {curr_leak_flow},
                # LeakNode={curr_node_name}, {curr_axis_name}")
            print("\n------")
            print(f"T:{run_id} - All leaks nodes {curr_node_name} Time= {time.time() - start2}")
            logger_objc.info(f"T:{run_id} - Executed simulation for node: {curr_node_name}, "
                             f"Time= {time.time() - start2}")

    @staticmethod
    def input_arguments_check(epanet_file_name, number_of_threads, min_leak, max_leak, leak_flow_step,
                              leak_flow_threshold, output_dir, log_file):
        # TODO more checks
        # leak flow
        if leak_flow_step < 0.00001:
            raise Exception(
                f"Chosen leak flow step {leak_flow_step} is too small! Please change it a value bigger than "
                f"0.00001")

        # min and maximum leakages<
        if min_leak >= max_leak:
            raise Exception(f"Min leak {min_leak} can't be bigger or equal to max leak {max_leak}!")
        # threads
        if number_of_threads < 1:
            raise Exception(f"Number of threads {number_of_threads} must be bigger or equal to 1!")
        if not isinstance(number_of_threads, int):
            raise Exception(f"Number of threads {number_of_threads} must be an integer! "
                            f"Current type is {type(number_of_threads)}!")

    def generate_leaks_arrays(self):
        """
        Function generates leak range depending on min, max and step variables. It then almost equally splits the load between
        threads.
        TODO update documentatiom step is fixed since this step concerns file sizes, the actual step size implemented in the means function
        :return: 2D array. First dimension tells which thread should use which leak flow and the second dimension contains
        leak values that should be used by that thread.
        IMPORTANT: This function can be used for single threaded use also in which the function also returns a 2D array, but
        with just one dimension.
        """
        # TODO cover edge cases, to equally distribute the load between threads -> Possible solution, step that ensures
        #  that each thread gets at least one leakage?
        leak_thread_arr = []
        num_of_steps = math.floor((self.max_leak - self.min_leak) / self.LEAKS_PER_FILE_INTERVAL)
        min_max_leak_arr = [round(i, 3) for i in np.linspace(self.min_leak, self.max_leak, num_of_steps, endpoint=True)]

        if self.number_of_threads != 1:
            # number of leaks placed on each node, last thread can get a different number
            leaks_per_thread = len(min_max_leak_arr) // self.number_of_threads

            # for could be without ifs, but it's more this way readable
            for i_thread in range(self.number_of_threads):
                if i_thread == self.number_of_threads - 1:
                    # last thread gets the rest of the leaks, usually more than others
                    tmp_leak_arr = min_max_leak_arr[i_thread * leaks_per_thread:]
                else:
                    tmp_leak_arr = min_max_leak_arr[i_thread * leaks_per_thread: ((i_thread + 1) * leaks_per_thread) + 1]

                leak_thread_arr.append(tmp_leak_arr)
            print(f"Leak thread array: {leak_thread_arr}")
            return leak_thread_arr
        else:
            return [min_max_leak_arr]

    def get_leak_time_identification_and_water_loss_from_df(self, divergence_df, leak_flows_df):
        """
        Function generates two arrays which contain information about the first time that the leakage on a node went
        over the specified threshold.
         TODO if a node doesn't have a leak bigger than the specified threshold, it will not be included which leads to
           inconsistency, this should be solved before this function is properly used.

        :param divergence_df: Dataframe. Contains the difference in change of flow on the nodes in the network.
        :param leak_flows_df: Dataframe. Contains information about how much leakage was placed on the network
        at every timestamp.
        :param leak_flow_threshold: Float. The threshold after which the leakage is considered to be happening. In liters
        per second.
        :return: Tuple of two arrays. First array contains the times of the first leak on the node, second array contains
        commutative leakage in the network over the whole simulation time.
        """
        leak_identified_time_arr = []  # time when the leak was identified
        water_loss_arr = []  # Water loss arr (L/s), how much water is wasted

        for node_name_i in divergence_df.columns:
            comm_leak_flow = 0
            leak_detected_hour = 0
            for hour_in_day in range(self.SECONDS_IN_HOUR, 25 * self.SECONDS_IN_HOUR, self.SECONDS_IN_HOUR):
                comm_leak_flow += leak_flows_df["LeakFlow"][hour_in_day] * 3600

                if divergence_df.loc[hour_in_day, node_name_i] > self.leak_flow_threshold:
                    leak_detected_hour = hour_in_day
                    break
            leak_identified_time_arr.append(leak_detected_hour)
            water_loss_arr.append(round(comm_leak_flow, 4))

        return leak_identified_time_arr, water_loss_arr

    def run_simulation(self, wntr_network_instance, file_prefix):
        """
        # TODO is it ok to be static?
        Runs the epanet simulation on the water model of the instance or the model that was passed to it.

        :return: Returns the wntr.sim.results.SimulationResults object. Important properties of the object are
        SimulationResults.link and SimulationResults.node with which we can access the pressures/flow rate of the
        simulation.
        """
        sim = wntr.sim.EpanetSimulator(wntr_network_instance)
        simulation_results = sim.run_sim(version=2.2, file_prefix=f"{self.TEMPORARY_EPA_FILES_PREFIX}_{file_prefix}")

        return simulation_results

    @staticmethod
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

    @staticmethod
    def append_dict_to_file(main_data_dict, out_f_name):
        """
        Function appends a dictionary to a file.
        # TODO add option of saving to more memory friendly formats parquet etc

        :param main_data_dict: Dictionary. Dictionary which we want to append to the file.
            In general it should be of the following format or similar format:
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
        :param out_f_name: String. Name of the file to which we want to append to.
        """
        if not out_f_name.endswith(".pkl"):
            raise Exception("Output file must be a .pkl file")
        with open(out_f_name, "ab") as file:
            pickle.dump(main_data_dict, file)

    @staticmethod
    def create_logger(log_file_name):
        """
        Creates an instance of a logger. Used for multithreading so that each logger object can write to the same file.

        :param log_file_name: String. Path to the log file to which you want to write.
        """
        logger = multiprocessing.get_logger()
        logger.setLevel(logging.INFO)

        log_string = "%(asctime)s [Thread ID: %(thread)-5d, Process: %(processName)-10s] %(levelname)-8s %(message)s"
        formatter = logging.Formatter(log_string, datefmt="%Y-%m-%d %H:%M:%S")

        handler = logging.FileHandler(log_file_name)
        handler.setFormatter(formatter)

        # this bit will make sure you won't have duplicated messages in the output
        if not len(logger.handlers):
            logger.addHandler(handler)
        return logger

    def clear_temporary_epanet_files(self):
        """
        TODO add description
        :return:
        """
        self.main_logger.info(f"Removing temporary files ...")
        extensions_to_delete = (".inp", ".rpt", ".bin")

        for file in os.listdir():   # get all files in current directory
            if file.startswith(self.TEMPORARY_EPA_FILES_PREFIX) and file.endswith(extensions_to_delete):
                os.remove(file)
                self.main_logger.info(f"Removing temporary file: {file}")


