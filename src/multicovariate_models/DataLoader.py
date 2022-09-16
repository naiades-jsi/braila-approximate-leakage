import numpy as np
import pandas as pd
import scipy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import src.analytics.linalg as linalg


class DataLoader:
    ENC_NODE_COL = "encoded_node_with_leak"
    NODE_COL = "node_with_leak"
    SENSOR_COLS = ["Sensor1", "Sensor2", "Sensor3", "Sensor4", "J-Apollo", "J-RN2", "J-RN1"]
    SENSOR_DPRESSURE_COLS = ["Sensor1_dpressure", "Sensor2_dpressure", "Sensor3_dpressure", "Sensor4_dpressure", "J-Apollo_dpressure", "J-RN2_dpressure", "J-RN1_dpressure"]
    LEAK_A_COL = "leak_amount"
    LEAK_A_F_COL = "leak_amount_f"

    TRAIN_SIZE = 0.9

    def __init__(self, data_df):
        """
        Constructor for the DataLoader class. It checks if the prepared_data_df is of the correct type and contains all
        the columns, it then splits the data into x_train, x_test, y_train, y_test numpy arrays.

        :param data_df: DataFrame. The prepared dataframe containing the data to be used for training and
        testing with the correct columns. It SHOULD NOT include duplicates!
        """
        self.data_df = data_df

        node_col_series = self.data_df[self.NODE_COL]
        # encoding node names to numbers
        self.label_enc = LabelEncoder()
        self.data_df[self.ENC_NODE_COL] = self.label_enc.fit_transform(node_col_series)
        self.node_to_enc_node_dict = dict(zip(self.label_enc.classes_,
                                              self.label_enc.transform(self.label_enc.classes_)))

    def encode_node_id(self, node_id):
        return self.label_enc.transform(node_id)

    def decode_node_label(self, node_label):
        return self.label_enc.inverse_transform(node_label)

    def check_dataframe(self, df):
        # Check if prepared_data_df is a dataframe
        if not isinstance(df, pd.DataFrame):
            raise Exception(f"Input parameter 'prepared_data_df' must be of type pd.DataFrame! "
                            f"Got {type(df)} instead!")

        # Check if prepared_data_df has all required columns
        if not set(self.SENSOR_COLS + [self.NODE_COL]).issubset(set(df.columns)):
            raise Exception(f"Prepared data does not have expected columns! It should contain the "
                            f"following columns: {self.SENSOR_COLS + [self.NODE_COL]}, but it has only these: "
                            f"{list(df.columns)}")

    def prepare_training_and_test_data(self, mode, subsample_size=None):
        """
        Method prepares the data for training and testing. First it encodes the node names with LabelEncoder, then it
        splits the data into training and testing data, and finally it splits the data into x_train, x_test,
        y_train, y_test numpy arrays depending on the columns specified in global class attributes.

        :param mode: String. The mode of the data split, can be either "RANDOM" or "SEQUENTIAL".
        :param subsample_size: Float. The size of the subsample to be used for sequential subsampling.
        :return: Tuple of five elements. The tuple contains the following:
            - x_train: Numpy array. The training data.
            - x_test: Numpy array. The testing data.
            - y_train: Numpy array. The training labels.
            - y_test: Numpy array. The testing labels.
            - node_to_enc_node_dict: Dictionary. The mapping between node names and encoded node names.
        """
        if mode == "SEQUENTIAL-SUBSAMPLING" and subsample_size is None:
            raise Exception("If mode is 'SEQUENTIAL-SUBSAMPLING', subsample_size must be specified!")

        train_df, test_df = None, None
        if mode == "RANDOM":
            # Splitting data into train and test dataframes by index
            train_df, test_df = self.data_split_by_enc_node(self.data_df)
        elif mode == "SEQUENTIAL-SUBSAMPLING":
            # Splitting data into train and test dataframes by sequential subsampling
            train_df, test_df = self.subset_df_data(self.data_df, frac_of_data_to_subsample=subsample_size)
        else:
            raise Exception(f"Unsupported mode: {mode}")

        print(f"Train data shape: {train_df.shape}, test data shape: {test_df.shape}")
        # Prepare X matrices
        x_train = train_df[self.SENSOR_COLS].to_numpy()
        x_test = test_df[self.SENSOR_COLS].to_numpy()

        # Prepare output vectors
        y_train = train_df[self.ENC_NODE_COL].to_numpy()
        y_test = test_df[self.ENC_NODE_COL].to_numpy()

        print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
        return x_train, x_test, y_train, y_test, self.node_to_enc_node_dict

    def data_split_by_enc_node(self, df):
        """
        Method splits the dataframe into train and test dataframes so that the training dataset contains
        self.train_size % of rows of each node, and the test dataset contains the remaining rows.
        Dataframe is grouped by self.ENC_NODE_COL and aggregates all indexes of each node into a list. This list of
        indexes is then shuffled and split into train and test dataframes.

        Really important to understand that the split is done by the encoded node name, and not by randomly splitting
        on indexes!

        :param df: DataFrame. The dataframe to be split into training and test dataframes.
        :return: (DataFrame, DataFrame). Two DataFrames, one with training and with test data.
        """
        train_arr, test_arr = np.array([]), np.array([])
        grouped_node_indexes = df.reset_index().groupby(self.ENC_NODE_COL).agg(list)["index"]

        for node_name, index_arr in grouped_node_indexes.items():
            node_train_indexes, node_test_indexes = train_test_split(index_arr, train_size=self.TRAIN_SIZE)
            # combine to main arrays
            train_arr = np.append(train_arr, node_train_indexes)
            test_arr = np.append(test_arr, node_test_indexes)

        # TODO test if this works as it should
        return df.loc[train_arr], df.loc[test_arr]

    def subset_df_data(self, df, frac_of_data_to_subsample):
        """
        Method sub-samples the dataframe to a certain fraction of the data, it does this sequentially by reducing the
        array of leaks to a certain fraction of the data. Dataframe should contain the column self.LEAK_A_COL!
        Method is meant to be used for testing so that the models can be evaluated on a smaller dataset, but the
        training and testing data are also split sequentially not randomly. This should result in higher accuracy
        of the model since it ensures that model has already seen similar data to the ones that we want to predict for,
        compared to the random sampling which could result in model getting only a certain leak amount not the full
        spectrum.

        :param df: DataFrame. The dataframe to be sub-sampled.
        :param frac_of_data_to_subsample: Float. The fraction of the data we want to keep.
        :return: (Dataframe, DataFrame). The train and test subsampled dataframes.
        """
        # Convert column to a float
        df[self.LEAK_A_F_COL] = df[self.LEAK_A_COL].str.replace("LPS", "").astype(float)
        # Get all unique leak amounts in a sorted array
        leak_amounts_arr = sorted(list(df[self.LEAK_A_F_COL].unique()))

        # Assign which leaks go into training and test set, random sampling of step sampling
        frac_of_arr = leak_amounts_arr[:int(len(leak_amounts_arr) * frac_of_data_to_subsample)]
        training_l_arr, test_l_arr = self.split_leak_amount_arr(leak_arr=frac_of_arr)

        # Get only rows with specified leak amounts
        train_df = df[df[self.LEAK_A_F_COL].isin(training_l_arr)]
        test_df = df[df[self.LEAK_A_F_COL].isin(test_l_arr)]

        # We don't need to sample for each node, since the sampling is already done on leak basis, and every node has
        # the same amount of leaks/rows
        return train_df, test_df

    def split_leak_amount_arr(self, leak_arr):
        """
        Method splits the array of leaks into two arrays, one with the training leaks and one with the test leaks. The
        split is not done randomly but sequentially so that the leak values are not too dispersed in the array.

        :param leak_arr: List. The array of leak amounts.
        :return: (List, List). Two lists, one with the training leaks and one with the test leaks.
        """
        inp_arr_len = len(leak_arr)

        # generate indexes of the train leak array
        train_l_indexes = np.round(np.linspace(0, inp_arr_len - 1, int(inp_arr_len * self.TRAIN_SIZE))).astype(int)
        train_leaks = np.array(leak_arr)[train_l_indexes]
        test_leaks = np.setdiff1d(leak_arr[:inp_arr_len], train_leaks)
        return train_leaks, test_leaks

    def preprocess_dpressure(self, pressure_df_mat):
        """
        Normalizes the rows so that they sum to 1. If the row is 0 it is left intact.
        """
        # n_rows = pressure_df_mat.shape[0]
        # n_cols = pressure_df_mat.shape[1]

        # # normalize the rows
        # pressure_df_abs_mat = np.absolute(pressure_df_mat)
        # pressure_diff_sum = pressure_df_abs_mat.sum(axis=1)

        # nonzero_idxs = pressure_diff_sum >= 1e-5

        # pressure_diff_sum_inv = np.reciprocal(pressure_diff_sum, where=nonzero_idxs)

        # norm_mat = scipy.sparse.csr_matrix((pressure_diff_sum_inv, (range(n_rows), range(n_rows))), shape=(n_rows, n_rows))
        # pressure_df_mat = norm_mat.dot(pressure_df_mat)

        # return pressure_df_mat
        return linalg.normalize_rows_l1(pressure_df_mat)

    def preprocess_abs_pressure(self, pressure_mat):
        return pressure_mat


    def get_random_data_split_by_node(self):
        return self.prepare_training_and_test_data(mode="RANDOM")

    def get_sequential_subsample_data_split_by_leaks(self, subsample_size=0.01):
        return self.prepare_training_and_test_data(mode="SEQUENTIAL-SUBSAMPLING", subsample_size=subsample_size)
