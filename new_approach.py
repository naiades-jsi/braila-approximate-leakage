import os

import pandas as pd
from sklearn.cluster import MiniBatchKMeans, Birch


def func_1():
    """
    TODO function 1:
        - one matches simulated data with the real data and finds the best fit (df, with smallest error)
        - return the dataframes and their meta data (node with leak and its amount)
        - optimization approach:
            - presort data on disk and search for matches each time from disk
    """


def func_2():
    """
    TODO function 2:
        - Machine learning approach: generate feature vector X and feature vector Y:
        - Feature vectors X contains:
            - 1. real sensor data for all 8 sensors
            - 2. another 8 features (one for each sensor) which means if the sensor was flagged as anomalous or not
            with other approaches before (matic anomaly detection), add this later, first just 8 features
            - 3. timestamp? optional, first just the 16 features
        - Feature vector Y contains:
            - amount of leak
            - nodes that were in that group
    """
    # TODO try clustering approach that only keeps centroid and discard other data
    #   X: | timestamp | s1 | s2 | ... | s8 |
    #   Y: | (node, leak), (node, leak), .... |

    # TODO read data from disk and generate feature vectors
    read_dfs_and_generate_feature_vectors("./data/")

    # TODO plot the data

    # TODO train the model: either k-means, gaussian mixture, dbscan, or anything that precomputes centroids
    #   - Suitable implementations in sklearn:
    #       - sklearn.cluster.MiniBatchKMeans
    #       - sklearn.cluster.Birch
    #       - ? ELKI's DBSCAN

    pass


def find_file_names_in_dir(data_dir):
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
        else:
            print(f"Current estimated batch size {current_batch_size} MB")

    print("\n\nExecution finished")


def multiple_sensor_approach():
    # TODO
    pass


if __name__ == "__main__":
    multiple_sensor_approach()









