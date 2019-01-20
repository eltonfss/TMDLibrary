import pandas
from detectors.tmd_base import TravelModeDetector
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score


class AdaBoostTMD(TravelModeDetector):
    """
    Wrapper that uses AdaBoost to train a travel mode detector through smartphone sensors
    """

    MODEL_FILENAME = 'model.pkl'
    CLASSES_FILENAME = 'classes.pkl'
    DEFAULT_MODEL_PATH = 'ab_tmd'
    DEFAULT_TRAIN_RATIO = 0.5
    DEFAULT_VAL_RATIO = 0.5
    DEFAULT_TEST_RATIO = 0.0

    def __init__(self, save_path=DEFAULT_MODEL_PATH, **kwargs):
        super(AdaBoostTMD, self).__init__(save_path=save_path, **kwargs)

    def fit(
            self,
            data_frame: pandas.DataFrame,
            travel_mode_column: str = 'target',
            shuffle: bool =True,
            train_ratio: float = DEFAULT_TRAIN_RATIO,
            val_ratio: float = DEFAULT_VAL_RATIO,
            test_ratio: float = DEFAULT_TEST_RATIO,
            base_estimator=None,
            n_estimators=50,
            learning_rate=1.,
            algorithm='SAMME.R',
            random_state=None,
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
        :param base_estimator:
        :param n_estimators:
        :param learning_rate:
        :param algorithm:
        :param random_state:
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
        adaboost = AdaBoostClassifier(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state
        )

        # train and validate model
        adaboost.fit(x_train, y_train)

        # get train accuracy
        y_pred = adaboost.predict(x_train)
        print("Train accuracy: {:.4f}".format(accuracy_score(y_train, y_pred)))

        # get val accuracy
        if val_ratio > 0.0:
            y_pred = adaboost.predict(x_val)
            print("Validation accuracy: {:.4f}".format(accuracy_score(y_val, y_pred)))

        # get test accuracy
        if test_ratio > 0.0:
            y_pred = adaboost.predict(x_test)
            print("Test accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))

        self.model = adaboost
        self.save(self.save_path)
