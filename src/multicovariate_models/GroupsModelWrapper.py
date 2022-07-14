import os
import pickle
from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC


# TODO test implemented methods
# TODO implement proper training method and test it
# TODO implement evaluation
# TODO tune hyper parameters for each model
class GroupsModelWrapper:
    """
    GroupsModel class
    """
    SUPPORTED_MODELS = ["GMM", "SVM"]

    def __init__(self, x_train, x_test, y_train, y_test, node_to_enc_node_dict, model_type):
        """
        Initializes the GroupsModel class, which is meant to train and evaluate the model before it is saved and
        used in our service. Class attributes are set here based on the input parameters, node_to_enc_node_dict contains
        the mapping between node names and encoded node names and node count is the number of nodes in the network.

        :param model_type: String. The type of model to be used. Currently supported: "GMM" and "SVM".
        evaluation.
        :param x_train: Numpy array. The training data.
        :param y_train: Numpy array. The training labels.
        :param x_test: Numpy array. The test data.
        :param y_test: Numpy array. The test labels.
        :param node_to_enc_node_dict: Dictionary. The mapping between node names and encoded node names.
        """
        self.check_input_parameters(model_type)
        self.model_type = model_type

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.node_to_enc_node_dict = node_to_enc_node_dict
        self.node_count = len(self.node_to_enc_node_dict.keys())

        # Set model
        self.model = self.set_model()
        # Flag to indicate if the model has been trained
        self.trained_bool = False

    def check_input_parameters(self, model_type):
        """
        Checks if the input parameters are valid, else it raises and exception. Information about the parameters is
        located in the init method.
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise Exception(f"Model type {model_type} is not supported!")

    def set_model(self):
        """
        Sets the model to be used for training and predicting with all the necessary hyper parameters.

        :return: Sklearn model. The model to be used for training and predicting.
        """
        model = None
        if self.model_type is None:
            raise Exception("Model type is not set!")

        if self.model_type == self.SUPPORTED_MODELS[0]:
            model = GaussianMixture(n_components=self.node_count, random_state=0, n_init=10, init_params="kmeans")
        elif self.model_type == self.SUPPORTED_MODELS[1]:
            model = SVC(kernel="sigmoid", C=1, decision_function_shape="ovo")
        else:
            raise Exception(f"Unexpected error for model type '{self.model_type}'")
        return model

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
        y_pred_arr = np.array([])

        for x_row in self.x_test:
            y_pred_arr = np.append(y_pred_arr, self.predict_node([x_row]))

        accuracy = accuracy_score(y_true=self.y_test, y_pred=y_pred_arr)
        recall = recall_score(y_true=self.y_test, y_pred=y_pred_arr, average="macro")
        precision = precision_score(y_true=self.y_test, y_pred=y_pred_arr, average="macro")
        print(f"\nAccuracy: {accuracy}")
        print(f"Precision: {precision}, recall: {recall}")

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

