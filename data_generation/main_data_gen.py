from data_generation.EpanetLeakGenerator import EpanetLeakGenerator


if __name__ == "__main__":
    output_dir = "./one_leak_per_node_output"
    leak_generator_instance = EpanetLeakGenerator(epanet_file_name="./../data/epanet_networks/Braila_V2022_2_2.inp",
                                                  number_of_threads=6,
                                                  min_leak=0.5,
                                                  max_leak=23.1,
                                                  leak_flow_step=0.001,
                                                  leak_flow_threshold=0.5,
                                                  output_dir=output_dir,
                                                  log_file=f"{output_dir}/data_generation.log")

    leak_generator_instance.multi_thread_data_generation()