import pandas
from detectors.tmd_base import TravelModeDetector
from sklearn import tree
from sklearn.metrics import accuracy_score


class DecisionTreeTMD(TravelModeDetector):
    """
    Wrapper that uses DecisionTree to train a travel mode detector through smartphone sensors
    """

    MODEL_FILENAME = 'model.pkl'
    CLASSES_FILENAME = 'classes.pkl'
    DEFAULT_MODEL_PATH = 'dt_tmd'
    DEFAULT_TRAIN_RATIO = 0.5
    DEFAULT_VAL_RATIO = 0.5
    DEFAULT_TEST_RATIO = 0.0

    def __init__(self, save_path=DEFAULT_MODEL_PATH, **kwargs):
        super(DecisionTreeTMD, self).__init__(save_path=save_path, **kwargs)

    def fit(
            self,
            data_frame: pandas.DataFrame,
            travel_mode_column: str = 'target',
            shuffle: bool =True,
            train_ratio: float = DEFAULT_TRAIN_RATIO,
            val_ratio: float = DEFAULT_VAL_RATIO,
            test_ratio: float = DEFAULT_TEST_RATIO,
            criterion="gini",
            splitter="best",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.,
            max_features=None,
            random_state=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.,
            min_impurity_split=None,
            class_weight=None,
            presort=False,
            **kwargs
    ):
        """
        Pre process input and train DT for travel mode detection
        :param data_frame:
        :param travel_mode_column:
        :param shuffle:
        :param train_ratio:
        :param val_ratio:
        :param test_ratio:
        :param criterion:
        :param splitter:
        :param max_depth:
        :param min_samples_split:
        :param min_samples_leaf:
        :param min_weight_fraction_leaf:
        :param max_features:
        :param random_state:
        :param max_leaf_nodes:
        :param min_impurity_decrease:
        :param min_impurity_split:
        :param class_weight:
        :param presort:
        :param kwargs:
        :return:
        """

        # partition data frame into train, validation and test
        x_test, x_train, x_val, y_test, y_train, y_val = self.get_preprocessed_partitions(
            data_frame=data_frame,
            travel_mode_column=travel_mode_column,
            convert_modes_to_numbers=True,
            shuffle=shuffle,
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            val_ratio=val_ratio
        )

        # build model
        decision_tree = tree.DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            class_weight=class_weight,
            presort=presort
        )

        # train and validate model
        decision_tree.fit(x_train, y_train)

        # get train accuracy
        y_pred = decision_tree.predict(x_train)
        print("Train accuracy: {:.4f}".format(accuracy_score(y_train, y_pred)))

        # get val accuracy
        if val_ratio > 0.0:
            y_pred = decision_tree.predict(x_val)
            print("Validation accuracy: {:.4f}".format(accuracy_score(y_val, y_pred)))

        # get test accuracy
        if test_ratio > 0.0:
            y_pred = decision_tree.predict(x_test)
            print("Test accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))

        self.model = decision_tree
        self.save(self.save_path)
