import pandas
from detectors.tmd_base import TravelModeDetector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class RandomForestTMD(TravelModeDetector):
    """
    Wrapper that uses RandomForest to train a travel mode detector through smartphone sensors
    """

    DEFAULT_MODEL_PATH = 'rf_tmd'
    DEFAULT_TRAIN_RATIO = 0.5
    DEFAULT_VAL_RATIO = 0.5
    DEFAULT_TEST_RATIO = 0.0

    def __init__(self, save_path=DEFAULT_MODEL_PATH, **kwargs):
        super(RandomForestTMD, self).__init__(save_path=save_path, **kwargs)

    def fit(
            self,
            data_frame: pandas.DataFrame,
            travel_mode_column: str = 'target',
            shuffle: bool =True,
            train_ratio: float = DEFAULT_TRAIN_RATIO,
            val_ratio: float = DEFAULT_VAL_RATIO,
            test_ratio: float = DEFAULT_TEST_RATIO,
            n_estimators=10,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.,
            max_features="auto",
            max_leaf_nodes=None,
            min_impurity_decrease=0.,
            min_impurity_split=None,
            bootstrap=True,
            oob_score=False,
            n_jobs=1,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None,
            **kwargs
    ):
        """
        Pre process input and train RF for travel mode detection
        :param data_frame:
        :param travel_mode_column:
        :param shuffle:
        :param train_ratio:
        :param val_ratio:
        :param test_ratio:
        :param n_estimators:
        :param criterion:
        :param max_depth:
        :param min_samples_split:
        :param min_samples_leaf:
        :param min_weight_fraction_leaf:
        :param max_features:
        :param max_leaf_nodes:
        :param min_impurity_decrease:
        :param min_impurity_split:
        :param bootstrap:
        :param oob_score:
        :param n_jobs:
        :param random_state:
        :param verbose:
        :param warm_start:
        :param class_weight:
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
        random_forest = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight
        )

        # train and validate model
        random_forest.fit(x_train, y_train)

        # get train accuracy
        y_pred = random_forest.predict(x_train)
        print("Train accuracy: {:.4f}".format(accuracy_score(y_train, y_pred)))

        # get val accuracy
        if val_ratio > 0.0:
            y_pred = random_forest.predict(x_val)
            print("Validation accuracy: {:.4f}".format(accuracy_score(y_val, y_pred)))

        # get test accuracy
        if test_ratio > 0.0:
            y_pred = random_forest.predict(x_test)
            print("Test accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))

        self.model = random_forest
        self.save(self.save_path)
