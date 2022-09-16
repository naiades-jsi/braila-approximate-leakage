import os

from data_generation.EpanetLeakGenerator import EpanetLeakGenerator
from src.data_parsing_and_saving.large_data_parsing import condense_pickle_files_to_relevant_data


if __name__ == "__main__":
    output_dir = "data/sim-single-leak"
    output_csv_fname = os.path.join(output_dir, 'simulated_sensor_data_8_cols.csv')

    # original
    leak_generator_instance = EpanetLeakGenerator(epanet_file_name="./data/epanet_networks/Braila_V2022_2_2.inp",
                                                  number_of_threads=16,
                                                  min_leak=0.5,
                                                  max_leak=23.1,
                                                  leak_flow_step=0.001,
                                                  leak_flow_threshold=0.5,
                                                  output_dir=output_dir,
                                                  log_file=f"{output_dir}/data_generation.log")
    # 1/50 samples
    # leak_generator_instance = EpanetLeakGenerator(epanet_file_name="./data/epanet_networks/Braila_V2022_2_2.inp",
    #                                               number_of_threads=16,
    #                                               min_leak=0.5,
    #                                               max_leak=23.1,
    #                                               leak_flow_step=0.05,
    #                                               leak_flow_threshold=0.5,
    #                                               output_dir=output_dir,
    #                                               log_file=f"{output_dir}/data_generation.log")
    # leak_generator_instance = EpanetLeakGenerator(epanet_file_name="./data/epanet_networks/Braila_V2022_2_2.inp",
    #                                               number_of_threads=16,
    #                                               min_leak=20.0,
    #                                               max_leak=20.2,
    #                                               leak_flow_step=0.1,
    #                                               leak_flow_threshold=0.5,
    #                                               output_dir=output_dir,
    #                                               log_file=f"{output_dir}/data_generation.log")

    print('running simulations')
    leak_generator_instance.multi_thread_data_generation()

    print('extracting relevant data')
    condense_pickle_files_to_relevant_data(output_dir, output_csv_fname, ijs_data_format=True)
