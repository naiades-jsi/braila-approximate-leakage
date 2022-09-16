from src.data_parsing_and_saving.large_data_parsing import condense_pickle_files_to_relevant_data

"""
This file is used for extraction of data from the large dataset generated from simulations with EpanetLeakGenerator.
In our case to reduce the data located in ircai with a 1size of 140gb to 0.5gb

"""
if __name__ == "__main__":
    data_dir = '/home/lstopar/data/naiadas/sim-single-leak'
    # data_dir = "/home/lstopar/data/naiadas/sim-single-leak"
    condense_pickle_files_to_relevant_data(data_dir, "data/dataframessimulations-step0.01.csv", ijs_data_format=True)
