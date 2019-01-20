import pandas
from detectors.tmd_base import TravelModeDetector
import autosklearn.classification
from sklearn.metrics import accuracy_score


class AutoMLTMD(TravelModeDetector):
    """
    Wrapper that uses AutoML to identify and train
    the best classification model for travel mode detection
    through smartphone sensors
    """

    MODEL_FILENAME = 'model.pkl'
    CLASSES_FILENAME = 'classes.pkl'
    DEFAULT_MODEL_PATH = 'automl_tmd'
    DEFAULT_TRAIN_RATIO = 0.5
    DEFAULT_VAL_RATIO = 0.5
    DEFAULT_TEST_RATIO = 0.0

    def __init__(self, save_path=DEFAULT_MODEL_PATH, **kwargs):
        super(AutoMLTMD, self).__init__(save_path=save_path, **kwargs)

    def fit(
            self,
            data_frame: pandas.DataFrame,
            travel_mode_column: str = 'target',
            shuffle: bool =True,
            train_ratio: float = DEFAULT_TRAIN_RATIO,
            val_ratio: float = DEFAULT_VAL_RATIO,
            test_ratio: float = DEFAULT_TEST_RATIO,
            maximum_search_time=3600,
            maximum_training_time=360,
            initial_configurations_via_metalearning=25,
            ensemble_size=50,
            ensemble_nbest=50,
            seed=1,
            memory_limit=3072,
            include_estimators=None,
            exclude_estimators=None,
            include_preprocessors=None,
            exclude_preprocessors=None,
            resampling_strategy='holdout',
            resampling_strategy_arguments=None,
            tmp_folder=None,
            output_folder=None,
            delete_tmp_folder_after_terminate=True,
            delete_output_folder_after_terminate=True,
            shared_mode=False,
            disable_evaluator_output=False,
            get_smac_object_callback=None,
            smac_scenario_args=None,
            metric=None,
            **kwargs
    ):
        """
        Pre process input and train AutoML for travel mode detection
        :param data_frame:
        :param travel_mode_column:
        :param shuffle:
        :param train_ratio:
        :param val_ratio:
        :param test_ratio:
        :param maximum_search_time:
        :param maximum_training_time:
        :param initial_configurations_via_metalearning:
        :param ensemble_size:
        :param ensemble_nbest:
        :param seed:
        :param memory_limit:
        :param include_estimators:
        :param exclude_estimators:
        :param include_preprocessors:
        :param exclude_preprocessors:
        :param resampling_strategy:
        :param resampling_strategy_arguments:
        :param tmp_folder:
        :param output_folder:
        :param delete_tmp_folder_after_terminate:
        :param delete_output_folder_after_terminate:
        :param shared_mode:
        :param disable_evaluator_output:
        :param get_smac_object_callback:
        :param smac_scenario_args:
        :param metric:
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
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=maximum_search_time,
            per_run_time_limit=maximum_training_time,
            initial_configurations_via_metalearning=initial_configurations_via_metalearning,
            ensemble_size=ensemble_size,
            ensemble_nbest=ensemble_nbest,
            seed=seed,
            ml_memory_limit=memory_limit,
            include_estimators=include_estimators,
            exclude_estimators=exclude_estimators,
            include_preprocessors=include_preprocessors,
            exclude_preprocessors=exclude_preprocessors,
            resampling_strategy=resampling_strategy,
            resampling_strategy_arguments=resampling_strategy_arguments,
            tmp_folder=tmp_folder,
            output_folder=output_folder,
            delete_tmp_folder_after_terminate=delete_tmp_folder_after_terminate,
            delete_output_folder_after_terminate=delete_output_folder_after_terminate,
            shared_mode=shared_mode,
            disable_evaluator_output=disable_evaluator_output,
            get_smac_object_callback=get_smac_object_callback,
            smac_scenario_args=smac_scenario_args
        )

        # train and validate model
        automl.fit(
            x_train,
            y_train,
            X_test=x_val,
            y_test=y_val,
            metric=metric
        )

        # get train accuracy
        y_pred = automl.predict(x_train)
        print("Train accuracy: {:.4f}".format(accuracy_score(y_train, y_pred)))

        # get val accuracy
        if val_ratio > 0.0:
            y_pred = automl.predict(x_val)
            print("Validation accuracy: {:.4f}".format(accuracy_score(y_val, y_pred)))

        # get test accuracy
        if test_ratio > 0.0:
            y_pred = automl.predict(x_test)
            print("Test accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))

        self.model = automl
        self.save(self.save_path)

    def refit(
            self,
            data_frame: pandas.DataFrame,
            travel_mode_column: str = 'target',
            shuffle: bool = True,
            train_ratio: float = DEFAULT_TRAIN_RATIO,
            val_ratio: float = DEFAULT_VAL_RATIO,
            test_ratio: float = DEFAULT_TEST_RATIO,
            **kwargs
    ):
        """
        Retrains the model using learned ensemble strucutre on new data
        :param data_frame:
        :param travel_mode_column:
        :param shuffle:
        :param train_ratio:
        :param val_ratio:
        :param test_ratio:
        :param kwargs:
        :return:
        """

        self.check_if_model_is_trained()

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

        self.model.refit(x_train, y_train)

        # get train accuracy
        y_pred = self.model.predict(x_train)
        print("Train accuracy: {:.4f}".format(accuracy_score(y_train, y_pred)))

        # get val accuracy
        if val_ratio > 0.0:
            y_pred = self.model.predict(x_val)
            print("Validation accuracy: {:.4f}".format(accuracy_score(y_val, y_pred)))

        # get test accuracy
        if test_ratio > 0.0:
            y_pred = self.model.predict(x_test)
            print("Test accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))

        self.save(**kwargs)

    def predict(self, data_frame: pandas.DataFrame, batch_size=1, verbose=0):
        """
        Detect travel mode of samples in data_frame
        :param data_frame:
        :param batch_size:
        :param verbose:
        :return:
        """
        self.check_if_model_is_trained()
        predictions = self.model.predict(data_frame.values, batch_size=batch_size)
        predictions = self.convert_numbers_to_classes(pandas.DataFrame(predictions))
        return predictions
