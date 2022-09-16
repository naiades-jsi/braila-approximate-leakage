import os
from data_generation.EpanetLeakGenerator import EpanetLeakGenerator

if __name__ == '__main__':
    base_dir = 'data/sim-single-leak'
    output_fpath = os.path.join(base_dir, 'base-pressure.csv')

    leak_generator_instance = EpanetLeakGenerator(epanet_file_name="./data/epanet_networks/Braila_V2022_2_2.inp",
                                                  number_of_threads=16,
                                                  min_leak=0.5,
                                                  max_leak=23.1,
                                                  leak_flow_step=0.001,
                                                  leak_flow_threshold=0.5,
                                                  output_dir=base_dir,
                                                  log_file=f"{base_dir}/data_generation.log")

    leak_generator_instance.export_base_simulation(output_fpath)

