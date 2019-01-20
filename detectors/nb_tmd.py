import pandas
from detectors.tmd_base import TravelModeDetector
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


class NaiveBayesTMD(TravelModeDetector):
    """
    Wrapper that uses NaiveBayes to train a travel mode detector through smartphone sensors
    """

    MODEL_FILENAME = 'model.pkl'
    CLASSES_FILENAME = 'classes.pkl'
    DEFAULT_MODEL_PATH = 'nb_tmd'
    DEFAULT_TRAIN_RATIO = 0.5
    DEFAULT_VAL_RATIO = 0.5
    DEFAULT_TEST_RATIO = 0.0

    def __init__(self, save_path=DEFAULT_MODEL_PATH, **kwargs):
        super(NaiveBayesTMD, self).__init__(save_path=save_path, **kwargs)

    def fit(
            self,
            data_frame: pandas.DataFrame,
            travel_mode_column: str = 'target',
            shuffle: bool =True,
            train_ratio: float = DEFAULT_TRAIN_RATIO,
            val_ratio: float = DEFAULT_VAL_RATIO,
            test_ratio: float = DEFAULT_TEST_RATIO,
            priors=None,
            **kwargs
    ):
        """
        Pre process input and train NB for travel mode detection
        :param data_frame:
        :param travel_mode_column:
        :param shuffle:
        :param train_ratio:
        :param val_ratio:
        :param test_ratio:
        :param priors:
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
            val_ratio=val_ratio,
            standardize=True
        )

        # build model
        nb = GaussianNB(priors=priors)

        # train and validate model
        nb.fit(x_train, y_train)

        # get train accuracy
        y_pred = nb.predict(x_train)
        print("Train accuracy: {:.4f}".format(accuracy_score(y_train, y_pred)))

        # get val accuracy
        if val_ratio > 0.0:
            y_pred = nb.predict(x_val)
            print("Validation accuracy: {:.4f}".format(accuracy_score(y_val, y_pred)))

        # get test accuracy
        if test_ratio > 0.0:
            y_pred = nb.predict(x_test)
            print("Test accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))

        self.model = nb
        self.save(self.save_path)

    def predict(self, data_frame: pandas.DataFrame, **kwargs):
        """

        :param data_frame:
        :param kwargs:
        :return:
        """
        data_frame = self.fill_nan_with_mean(data_frame)
        data_frame = self.standardize(data_frame)
        return super(NaiveBayesTMD, self).predict(data_frame, **kwargs)
