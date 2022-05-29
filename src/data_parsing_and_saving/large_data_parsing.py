import os

import pandas as pd
from sklearn.cluster import MiniBatchKMeans, Birch


def find_file_names_in_dir(data_dir):
    """
    Function finds all files in a directory that end with .pkl or .pickle. The function then returns a list of
    file names.

    :param data_dir: String. The directory where the files are located.
    :return: List of strings. The file names.
    """
    file_names_arr = []

    for file_name in os.listdir(data_dir):
        full_file_path = os.path.join(data_dir, file_name)
        if full_file_path.endswith((".pkl", ".pickle")):
            file_names_arr.append(full_file_path)
        else:
            print(f"File at location '{full_file_path}', doesn't end with '.pkl', so it won't be used!")

    return file_names_arr


def get_sensor_arr_and_column_arr():
    """
    Function returns sensor names and column names that are used in all other functions which manipulate the original
    and transformed data

    :return: Two lists of strings. The first one contains sensor names and the second one contains column names.
    """
    sensor_names_arr = ["Sensor1", "Sensor2", "Sensor3", "Sensor4", "J-Apollo", "J-RN2", "J-RN1"]
    column_names_arr = ["node_with_leak", "leak_amount", "origin_file"]

    return sensor_names_arr, column_names_arr


def prepare_df_from_file(file_name, sensor_names_arr, column_names_arr):
    """
    Function loads a data frame from a file extract only the relevant information at the timestamp of leak detection.
    It only keep the columns which are provided in the sensor_names_arr and adds meta data in the column_names_arr.

    :param file_name: String. The name of the file to be loaded.
    :param sensor_names_arr: List of strings. The sensor names to be kept.
    :param column_names_arr: List of strings. The column names to be added.
    :return: Dataframe. The dataframe which contains all the extracted information.
    """
    keep_f_name_in_df = False
    if len(column_names_arr) == 3 and column_names_arr[2] == "origin_file":
        keep_f_name_in_df = True

    file_df = pd.DataFrame(columns=column_names_arr)
    try:
        file_dict = pd.read_pickle(file_name)

        # take just the "LPM"-key or the leakage matrix of each dataframe
        for df_index, temp_df in enumerate(file_dict["LPM"]):
            # We take the first value since others are the same, original value is in hours,
            # if we want to get seconds: * 3600, -1 because it has hours from 1...24, array is 0...23
            # get timestamp when leak would be most noticeable
            timestamp_of_leak = int(file_dict["TM_l"][df_index][0]) * 3600
            temp_df = temp_df[sensor_names_arr]
            # filter to only get one row
            prepared_df = temp_df.loc[timestamp_of_leak].to_frame().T

            node_name_and_leak_tup = [i.strip() for i in prepared_df.columns.name.split(",")]
            prepared_df.at[timestamp_of_leak, column_names_arr[0]] = node_name_and_leak_tup[0]
            prepared_df.at[timestamp_of_leak, column_names_arr[1]] = node_name_and_leak_tup[1]

            if keep_f_name_in_df:
                prepared_df.at[timestamp_of_leak, column_names_arr[2]] = file_name.split("\\")[-1]

            file_df = pd.concat([file_df, prepared_df], ignore_index=True)

    except FileNotFoundError as e:
        print(f"File not found, for file '{file_name}': {e}")

    except EOFError as e:
        print(f"Error when reading the file, for file '{file_name}': {e}")

    return file_df


def prepare_df_from_file_ijs_data(file_name, sensor_names_arr, column_names_arr):
    """
    Function loads a data frame from a file extract only the relevant information at the timestamp of leak detection.
    It only keep the columns which are provided in the sensor_names_arr and adds meta data in the column_names_arr.

    The main difference between this function and the prepare_df_from_file function is that this function process data
    in the format which was used on IJS-s (our side) to generate the data.

    :param file_name: String. The name of the file to be loaded.
    :param sensor_names_arr: List of strings. The sensor names to be kept.
    :param column_names_arr: List of strings. The column names to be added.
    :return: Dataframe. The dataframe which contains all the extracted information.
    """
    keep_f_name_in_df = False
    if len(column_names_arr) == 3 and column_names_arr[2] == "origin_file":
        keep_f_name_in_df = True

    file_df = pd.DataFrame(columns=column_names_arr)
    try:
        file_dict = pd.read_pickle(file_name)

        # take just the "LPM"-key or the leakage matrix of each dataframe
        for df_index, temp_dict in enumerate(file_dict):
            # We take the first value since others are the same, original value is in hours,
            # if we want to get seconds: * 3600, -1 because it has hours from 1...24, array is 0...23
            # get timestamp when leak would be most noticeable
            timestamp_of_leak = int(temp_dict["TM_l"][0]) * 3600
            temp_df = temp_dict["LPM"][sensor_names_arr]
            # filter to only get one row
            prepared_df = temp_df.loc[timestamp_of_leak].to_frame().T

            node_name_and_leak_tup = [i.strip() for i in prepared_df.columns.name.split(",")]
            prepared_df.at[timestamp_of_leak, column_names_arr[0]] = node_name_and_leak_tup[0]
            prepared_df.at[timestamp_of_leak, column_names_arr[1]] = node_name_and_leak_tup[1]

            if keep_f_name_in_df:
                prepared_df.at[timestamp_of_leak, column_names_arr[2]] = file_name.split("\\")[-1]

            file_df = pd.concat([file_df, prepared_df], ignore_index=True)

    except FileNotFoundError as e:
        print(f"File not found, for file '{file_name}': {e}")

    except EOFError as e:
        print(f"Error when reading the file, for file '{file_name}': {e}")

    return file_df


def read_dfs_and_generate_feature_vectors(data_dir, clustering_method):
    """
    Function calls the function find_file_names_in_dir to find all files in a directory which it needs for processing.
    It then extracts only the required information from the files and periodically trains the model on the data.

    :param data_dir: String. The directory where the files are located.
    :param clustering_method: String. The clustering method to be used.
    """
    # get file names
    file_names_arr = find_file_names_in_dir(data_dir)
    sensor_names_arr, extra_columns_arr = get_sensor_arr_and_column_arr()

    # create clustering model
    if clustering_method == "mini_batch_kmeans":
        clustering_model = MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=6)
    elif clustering_method == "birch":
        clustering_model = Birch(n_clusters=None)   # TODO should warm start be used?
    else:
        raise Exception(f"Unknown clustering method: {clustering_method}!")

    batch_df = pd.DataFrame(columns=sensor_names_arr + extra_columns_arr)
    for file_index, file_name in enumerate(file_names_arr):
        print(f"Preparing file  {file_index + 1}/{len(file_names_arr)} - '{file_name}',")

        file_df = prepare_df_from_file(file_name, sensor_names_arr, extra_columns_arr)
        batch_df = pd.concat([batch_df, file_df], ignore_index=True)

        # perform batch (online) learning
        current_batch_size = round(batch_df.memory_usage(index=True).sum() / 1000000, 2)
        if current_batch_size >= 100.0:
            print(f"Performing training at batch size {current_batch_size} MB")
            clustering_model.partial_fit(batch_df[sensor_names_arr].to_numpy())

            # drop the dataframe from memory and prepare it for the next batch
            batch_df = pd.DataFrame(columns=sensor_names_arr + extra_columns_arr)
        else:
            print(f"Current estimated batch size {current_batch_size} MB")

    # Train the model on the last batch on which it wasn't yet trained on
    clustering_model.partial_fit(batch_df[sensor_names_arr].to_numpy())
    print("\n\nExecution finished")


def condense_pickle_files_to_relevant_data(data_dir, output_file_name, ijs_data_format=True):
    """
    Function reads all pickle files in a directory and condenses them into a single file which contains only
    the information we need as specified in the sensor_names_arr, extra_columns_arr variables.

    :param data_dir: String. The directory where the files are located.
    :param output_file_name: String. The name of the output file.
    :param ijs_data_format: Boolean. If True, the files will be parsed in the:
     IJS data format which is
        [dict("LPM": pd.DataFrame, "TM_l": pd.DataFrame, ...), dict("LPM": pd.DataFrame, "TM_l": pd.DataFrame, ...),
        ...]
    else it will be parsed as a dictionary of arrays:
        {"LPM": [pd.DataFrame, ...] "TM_l": [pd.DataFrame, ...] ...}
    """
    # get file names
    file_names_arr = find_file_names_in_dir(data_dir)
    sensor_names_arr, extra_columns_arr = get_sensor_arr_and_column_arr()
    batch_df = pd.DataFrame(columns=sensor_names_arr + extra_columns_arr)

    # create file with headers
    if not os.path.isfile(output_file_name):
        batch_df.to_csv(output_file_name, header=True, index=False)
    else:
        raise Exception(f"Output file '{output_file_name}' already exists!")

    # set the correct function for processing the data either DELFT or IJS
    process_data_func = prepare_df_from_file_ijs_data if ijs_data_format else prepare_df_from_file

    for file_index, file_name in enumerate(file_names_arr):
        print(f"Preparing file  {file_index + 1}/{len(file_names_arr)} - '{file_name}',")

        file_df = process_data_func(file_name, sensor_names_arr, extra_columns_arr)
        batch_df = pd.concat([batch_df, file_df], ignore_index=True)

        # perform batch (online) learning
        current_df_size = round(batch_df.memory_usage(index=True).sum() / 1000000, 2)
        if current_df_size >= 100.0:
            # save/append the dataframe to file and release memory
            batch_df.to_csv(output_file_name, mode='a', header=False, index=False)
            batch_df = pd.DataFrame(columns=sensor_names_arr + extra_columns_arr)
        else:
            print(f"Current estimated dataframe size {current_df_size} MB")

    # Save the last batch which hasn't been saved yet
    batch_df.to_csv(output_file_name, mode='a', header=False, index=False)
    print("\n\nExecution finished")

