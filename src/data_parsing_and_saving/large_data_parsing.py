import os

import pandas as pd
from sklearn.cluster import MiniBatchKMeans, Birch


def find_file_names_in_dir(data_dir):
    """
    TODO documentation
    :param data_dir:
    :return:
    """
    file_names_arr = []
    # TODO upgrade - implement a multi directory approach
    for file_name in os.listdir(data_dir):
        full_file_path = os.path.join(data_dir, file_name)
        if full_file_path.endswith(".pkl"):
            file_names_arr.append(full_file_path)
            # print("File", full_file_path)
        else:
            print(f"File at location '{full_file_path}', doesn't end with '.pkl', so it won't be used!")
    return file_names_arr


def read_dfs_and_generate_feature_vectors(data_dir, clustering_method):
    """
    TODO documentation
    :param data_dir:
    :param clustering_method:
    :return:
    """
    # sensor names array
    tmp_sens_arr = ["Sensor1", "Sensor2", "Sensor3", "Sensor4", "J-Apollo", "J-RN2", "J-RN1"]
    extra_columns = ["node_with_leak", "leak_amount"]
    # get file names
    file_names_arr = find_file_names_in_dir(data_dir)

    # create clustering model
    clustering_model = None
    if clustering_method == "mini_batch_kmeans":
        clustering_model = MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=6)
    elif clustering_method == "birch":
        clustering_model = Birch(n_clusters=None)   # TODO should warm start be used?
    else:
        raise Exception(f"Unknown clustering method: {clustering_method}!")

    batch_df = pd.DataFrame(columns=tmp_sens_arr + extra_columns)
    for file_index, file_name in enumerate(file_names_arr):
        print(f"Preparing file  {file_index + 1}/{len(file_names_arr)} - '{file_name}',")
        try:
            temporary_dict = pd.read_pickle(file_name)
        except Exception as e:
            print(f"Error while reading file '{file_name}': {e}")
            continue

        # reduce each dataframe to one row, check the amount of dfs, print("len", len(temporary_dict["LPM"]))
        for df_index, temp_df in enumerate(temporary_dict["LPM"]):
            # We take the first value since others are the same, original value is in hours,
            # if we want to get seconds: * 3600, -1 because it has hours from 1...24, array is 0...23
            # get timestamp when leak would be most noticeable
            timestamp_of_leak = int(temporary_dict["TM_l"][df_index][0]) * 3600

            # filter data to only keep the columns relevant to us
            temp_df = temp_df[tmp_sens_arr]
            # filter to only get one column
            prepared_df = temp_df.loc[timestamp_of_leak].to_frame().T
            node_name_and_leak_tup = [i.strip() for i in prepared_df.columns.name.split(",")]
            prepared_df.at[timestamp_of_leak, extra_columns[0]] = node_name_and_leak_tup[0]
            prepared_df.at[timestamp_of_leak, extra_columns[1]] = node_name_and_leak_tup[1]

            # TODO also store file name as metadata, so that the original data can easily be found?
            # adding to main df
            batch_df = pd.concat([batch_df, prepared_df], ignore_index=True)

        current_batch_size = round(batch_df.memory_usage(index=True).sum() / 1000000, 2)
        # perform batch (online) learning
        if current_batch_size >= 100.0:
            print(f"Performing training at batch size {current_batch_size} MB")
            clustering_model.partial_fit(batch_df[tmp_sens_arr].to_numpy())

            # drop the dataframe from memory and prepare it for the next batch
            batch_df = pd.DataFrame(columns=tmp_sens_arr + extra_columns)
        else:
            print(f"Current estimated batch size {current_batch_size} MB")

    print("\n\nExecution finished")
    return batch_df