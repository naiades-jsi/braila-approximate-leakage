import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


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
        Initializes the GroupsModel class, which is meant to train and evaluate the model before it is saved and
        used in our service. Class attributes are set here based on the input parameters, node_to_enc_node_dict contains
        the mapping between node names and encoded node names and node count is the number of nodes in the network.

        :param model_type: String. The type of model to be used. Currently supported: "GMM" and "SVM".
        :param prepared_data_df: DataFrame. The prepared dataframe containing the data to be used for training and
        evaluation. It SHOULD NOT include duplicates!
        """
        self.check_input_parameters(model_type, prepared_data_df)
        self.model_type = model_type
        # Flag to indicate if the model has been trained
        self.trained_bool = False
        self.train_size = 0.7

        # Processing data frame and preparing data for training and predicting
        prepared_data_tup = self.prepare_training_and_test_data(prepared_data_df)

        self.x_train = prepared_data_tup[0]
        self.x_test = prepared_data_tup[1]
        self.y_train = prepared_data_tup[2]
        self.y_test = prepared_data_tup[3]
        self.node_to_enc_node_dict = prepared_data_tup[4]
        self.node_count = len(self.node_to_enc_node_dict.keys())

        # Set model
        self.model = self.set_model()

    def check_input_parameters(self, model_type, prepared_data_df):
        """
        Checks if the input parameters are valid, else it raises and exception. Information about the parameters is
        located in the init method.
        """
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

    def set_model(self):
        """
        Sets the model to be used for training and predicting with all the necessary hyper parameters.

        :return: Sklearn model. The model to be used for training and predicting.
        """
        model = None
        if self.model_type is None:
            raise Exception("Model type is not set!")

        if self.model_type == "GMM":
            model = GaussianMixture(n_components=self.node_count, random_state=0, n_init=10, init_params="kmeans")
        elif self.model_type == "SVM":
            model = SVC(kernel="sigmoid", C=1, decision_function_shape="ovo")
        else:
            raise Exception(f"Unexpected error for model type '{self.model_type}'")
        return model

    def prepare_training_and_test_data(self, prepared_data_df):
        """
        Method prepares the data for training and testing. First it encodes the node names with LabelEncoder, then it
        splits the data into training and testing data, and finally it splits the data into x_train, x_test,
        y_train, y_test numpy arrays depending on the columns specified in global class attributes.

        :param prepared_data_df: DataFrame. The prepared dataframe containing the data to be used for training and
        evaluation. It SHOULD NOT include duplicates!
        :return: Tuple of five elements. The tuple contains the following:
            - x_train: Numpy array. The training data.
            - x_test: Numpy array. The testing data.
            - y_train: Numpy array. The training labels.
            - y_test: Numpy array. The testing labels.
            - node_to_enc_node_dict: Dictionary. The mapping between node names and encoded node names.
        """
        node_col_series = prepared_data_df[self.NODE_COL]

        # encoding node names to numbers
        label_enc = LabelEncoder()
        prepared_data_df[self.ENC_NODE_COL] = label_enc.fit_transform(node_col_series)
        node_to_enc_node_dict = dict(zip(label_enc.classes_, label_enc.transform(label_enc.classes_)))

        # Splitting data into train and test dataframes by index
        train_df, test_df = self.data_split_by_enc_node(prepared_data_df)
        print(f"Train data shape: {train_df.shape}, test data shape: {test_df.shape}")

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
            node_train_indexes, node_test_indexes = train_test_split(index_arr, train_size=self.train_size)
            # combine to main arrays
            train_arr = np.append(train_arr, node_train_indexes)
            test_arr = np.append(test_arr, node_test_indexes)

        # TODO test if this works as it should
        return df.loc[train_arr], df.loc[test_arr]

    def train(self):
        if self.model_type == "GMM":
            self.model.fit(self.x_train)
        elif self.model_type == "SVM":
            self.model.fit(self.x_train, self.y_train)
        else:
            raise Exception(f"Unexpected error for model type '{self.model_type}'!")

        self.trained_bool = True

    def predict_node(self, x_data):
        return self.model.predict(x_data)

    def predict_groups(self, x_data):
        return self.model.predict_proba(x_data)

    def evaluate_model_on_node_basis(self):
        y_pred_arr = []
        for x_row in self.x_test:
            y_pred_arr.append(self.predict_node(x_row))

        accuracy = accuracy_score(y_true=self.y_test, y_pred=y_pred_arr)
        print(f"Accuracy: {accuracy}")
        precision, recall, fscore, support = precision_recall_fscore_support(y_true=self.y_test, y_pred=y_pred_arr)
        print(f"Precision: {precision}, recall: {recall}, fscore: {fscore}, support: {support}")

    def evaluate_model_on_group_basis(self):
        y_pred_arr = []
        for x_row in self.x_test:
            probability_arr = self.predict_groups(x_row)
            y_pred_arr.append(probability_arr)
        # TODO make the main group fixed for example with 5 nodes
        # TODO check if ground truth label is in this 5 node array

    def save_model(self, dir_path):
        """
        Method saves the model to a pickle file, to be used for later predictions. If the model is not initialized or
        trained, it will raise an exception.

        :param dir_path: String. Path to directory where the model should be saved.
        """
        if self.model is None or self.trained_bool is False:
            raise Exception(f"Model with type '{self.model_type}' is not trained yet!")
        if not os.path.exists(dir_path):
            raise Exception(f"Directory '{dir_path}' does not exist!")

        saved_at_str = datetime.now().strftime("%m-%d-%Y_%H:%M")
        model_name = f"model_{self.model_type}_{saved_at_str}.pickle"
        path_to_model_pkl = os.path.join(dir_path, model_name)

        print(f"Saving model to {path_to_model_pkl} ...")
        with open(path_to_model_pkl, "wb") as model_file:
            pickle.dump(self.model, model_file)
        print("Successfully saved model!")

