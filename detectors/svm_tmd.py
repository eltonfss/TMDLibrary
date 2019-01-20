import pandas
from detectors.tmd_base import TravelModeDetector
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class SupportVectorMachineTMD(TravelModeDetector):
    """
    Wrapper that uses Support Vector Machine to train a travel mode detector through smartphone sensors
    """

    DEFAULT_MODEL_PATH = 'svm_tmd'
    DEFAULT_TRAIN_RATIO = 0.5
    DEFAULT_VAL_RATIO = 0.5
    DEFAULT_TEST_RATIO = 0.0

    def __init__(self, save_path=DEFAULT_MODEL_PATH, **kwargs):
        super(SupportVectorMachineTMD, self).__init__(save_path=save_path, **kwargs)

    def fit(
            self,
            data_frame: pandas.DataFrame,
            travel_mode_column: str = 'target',
            shuffle: bool =True,
            train_ratio: float = DEFAULT_TRAIN_RATIO,
            val_ratio: float = DEFAULT_VAL_RATIO,
            test_ratio: float = DEFAULT_TEST_RATIO,
            error_term_penalty=1.0,
            kernel='rbf',
            degree=3,
            gamma='auto',
            coef0=0.0,
            shrinking=True,
            probability=False,
            tol=1e-3,
            cache_size=200,
            class_weight=None,
            verbose=False,
            max_iter=-1,
            decision_function_shape='ovr',
            random_state=None,
            **kwargs
    ):
        """
        Pre process input and train SVM for travel mode detection
        :param data_frame:
        :param travel_mode_column:
        :param shuffle:
        :param train_ratio:
        :param val_ratio:
        :param test_ratio:
        :param error_term_penalty:
        :param kernel:
        :param degree:
        :param gamma:
        :param coef0:
        :param shrinking:
        :param probability:
        :param tol:
        :param cache_size:
        :param class_weight:
        :param verbose:
        :param max_iter:
        :param decision_function_shape:
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
        support_vector_machine = SVC(
            C=error_term_penalty,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            random_state=random_state
        )

        # train and validate model
        support_vector_machine.fit(x_train, y_train)

        # get train accuracy
        y_pred = support_vector_machine.predict(x_train)
        print("Train accuracy: {:.4f}".format(accuracy_score(y_train, y_pred)))

        # get val accuracy
        if val_ratio > 0.0:
            y_pred = support_vector_machine.predict(x_val)
            print("Validation accuracy: {:.4f}".format(accuracy_score(y_val, y_pred)))

        # get test accuracy
        if test_ratio > 0.0:
            y_pred = support_vector_machine.predict(x_test)
            print("Test accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))

        self.model = support_vector_machine
        self.save(self.save_path)
