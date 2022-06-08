import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


# TODO add documentation
# TODO test implemented methods
# TODO implement proper training method and test it
# TODO implement evaluation
# TODO tune hyper parameters for each model
# TODO split into DataLoader and Model classes
class GroupsModel:
    """
    GroupsModel class
    """
    SUPPORTED_MODELS = ["GMM", "SVM"]
    ENC_NODE_COL = "encoded_node_with_leak"
    NODE_COL = "node_with_leak"
    SENSOR_COLS = ["Sensor1", "Sensor2", "Sensor3", "Sensor4", "J-Apollo", "J-RN2", "J-RN1"]

    def __init__(self, model_type, prepared_data_df):
        """
        TODO
        :param model_type:
        :param prepared_data_df:
        """
        self.check_input_parameters(model_type, prepared_data_df)
        self.model_type = model_type
        self.train_size = 0.7
        self.model = None

        # Processing data frame and preparing data for training and predicting
        prepared_data_tup = self.prepare_training_and_test_data(prepared_data_df)

        self.x_train = prepared_data_tup[0]
        self.x_test = prepared_data_tup[1]
        self.y_train = prepared_data_tup[2]
        self.y_test = prepared_data_tup[3]
        self.node_to_enc_node_dict = prepared_data_tup[4]
        self.node_count = len(self.node_to_enc_node_dict.keys())

    def check_input_parameters(self, model_type, prepared_data_df):
        if model_type not in self.SUPPORTED_MODELS:
            raise Exception(f"Model type {model_type} is not supported!")

        # Check if prepared_data_df is a dataframe
        if not isinstance(prepared_data_df, pd.DataFrame):
            raise Exception(f"Input parameter 'prepared_data_df' must be of type pd.DataFrame! "
                            f"Got {type(prepared_data_df)} instead!")

        # Check if prepared_data_df has all required columns
        if set(prepared_data_df.columns).issubset(set(self.SENSOR_COLS + [self.NODE_COL])):
            raise Exception(f"Prepared data does not have expected columns! It should contain the "
                            f"following columns: {self.SENSOR_COLS + [self.NODE_COL]}, but it has only these: "
                            f"{list(prepared_data_df.columns)}")

    def prepare_training_and_test_data(self, prepared_data_df):
        """
        TODO
        :param prepared_data_df:
        :return:
        """
        node_col_series = prepared_data_df[self.NODE_COL]

        # encoding node names to numbers
        label_enc = LabelEncoder()
        prepared_data_df[self.ENC_NODE_COL] = label_enc.fit_transform(node_col_series)
        node_to_enc_node_dict = dict(zip(label_enc.classes_, label_enc.transform(label_enc.classes_)))

        # Splitting data into train and test dataframes by index
        train_df, test_df = self.data_split_by_enc_node(prepared_data_df)
        print(f"Train data shape: {train_df.shape}, test data shape: {test_df.shape}")
        # Setting original dataframe to None to free up memory
        prepared_data_df = None

        # Prepare X matrices
        x_train = train_df[self.SENSOR_COLS].to_numpy()
        x_test = test_df[self.SENSOR_COLS].to_numpy()

        # Prepare output vectors
        y_train = train_df[self.ENC_NODE_COL].to_numpy()
        y_test = test_df[self.ENC_NODE_COL].to_numpy()

        print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
        return x_train, x_test, y_train, y_test, node_to_enc_node_dict

    def data_split_by_enc_node(self, df):
        """
        TODO
        :param df:
        :return:
        """
        train_arr, test_arr = np.array([]), np.array([])
        grouped_node_indexes = df.reset_index().groupby(self.ENC_NODE_COL).agg(list)["index"]

        for node_name, index_arr in grouped_node_indexes.items():
            node_train_indexes, node_test_indexes = train_test_split(index_arr, train_size=self.train_size)
            # combine to main arrays
            train_arr = np.append(train_arr, node_train_indexes)
            test_arr = np.append(test_arr, node_test_indexes)

        # TODO test if this works as it should
        return df.loc[train_arr], df.loc[test_arr]

    def train(self):
        if self.model_type == "GMM":
            self.model = GaussianMixture(n_components=self.node_count, random_state=0, n_init=10, init_params="kmeans")
            self.model.fit(self.x_train)
        elif self.model_type == "SVM":
            self.model = SVC(kernel='sigmoid', C=1, decision_function_shape='ovo')
            self.model.fit(self.x_train, self.y_train)
        else:
            raise Exception(f"Model type {self.model_type} is not supported!")

    def predict(self):
        pass

    def evaluate_model(self, test_set):
        pass

    def save_model(self, dir_path):
        if self.model is None:
            raise Exception(f"Model with type {self.model_type} is not trained yet!")
        if not os.path.exists(dir_path):
            raise Exception(f"Directory {dir_path} does not exist!")

        saved_at_str = datetime.now().strftime("%m-%d-%Y_%H:%M")
        model_name = f"model_{self.model_type}_{saved_at_str}.pickle"
        path_to_model_pkl = os.path.join(dir_path, model_name)

        print(f"Saving model to {path_to_model_pkl} ...")
        with open(path_to_model_pkl, "wb") as model_file:
            pickle.dump(self.model, model_file)
        print("Successfully saved model!")

