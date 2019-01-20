from sklearn.linear_model import  LogisticRegression
import pandas
from detectors.tmd_base import TravelModeDetector
from sklearn.metrics import accuracy_score


class LogisticRegressionTMD(TravelModeDetector):
    """
    Wrapper that uses LogisticRegression to train a travel mode detector through smartphone sensors
    """

    DEFAULT_MODEL_PATH = 'lr_tmd'
    DEFAULT_TRAIN_RATIO = 0.5
    DEFAULT_VAL_RATIO = 0.5
    DEFAULT_TEST_RATIO = 0.0

    def __init__(self, save_path=DEFAULT_MODEL_PATH, **kwargs):
        super(LogisticRegressionTMD, self).__init__(save_path=save_path, **kwargs)

    def fit(
            self,
            data_frame: pandas.DataFrame,
            travel_mode_column: str = 'target',
            shuffle: bool =True,
            train_ratio: float = DEFAULT_TRAIN_RATIO,
            val_ratio: float = DEFAULT_VAL_RATIO,
            test_ratio: float = DEFAULT_TEST_RATIO,
            penalty='l2',
            dual=False,
            tol=1e-4,
            error_term_penalty=1.0,
            fit_intercept=True,
            intercept_scaling=1,
            class_weight=None,
            random_state=None,
            solver='liblinear',
            max_iter=100,
            multi_class='ovr',
            verbose=0,
            warm_start=False,
            n_jobs=1,
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
        :param penalty:
        :param dual:
        :param tol:
        :param error_term_penalty:
        :param fit_intercept:
        :param intercept_scaling:
        :param class_weight:
        :param random_state:
        :param solver:
        :param max_iter:
        :param multi_class:
        :param verbose:
        :param warm_start:
        :param n_jobs:
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
        logistic_regression = LogisticRegression(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=error_term_penalty,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs
        )

        # train and validate model
        logistic_regression.fit(x_train, y_train)

        # get train accuracy
        y_pred = logistic_regression.predict(x_train)
        print("Train accuracy: {:.4f}".format(accuracy_score(y_train, y_pred)))

        # get val accuracy
        if val_ratio > 0.0:
            y_pred = logistic_regression.predict(x_val)
            print("Validation accuracy: {:.4f}".format(accuracy_score(y_val, y_pred)))

        # get test accuracy
        if test_ratio > 0.0:
            y_pred = logistic_regression.predict(x_test)
            print("Test accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))

        self.model = logistic_regression
        self.save(self.save_path)
