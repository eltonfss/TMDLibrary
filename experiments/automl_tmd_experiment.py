from experiments.tmd_experiment_base import TMDExperiment
from detectors.automl_tmd import AutoMLTMD
import pandas as pd


class AutoTMDExperiment(TMDExperiment):
    """
    Class that implements optimizations for AutoMLTMDExperiment
    """

    def __init__(self, detector_type=AutoMLTMD, **kwargs):
        super(AutoTMDExperiment, self).__init__(detector_type=detector_type, **kwargs)
        self.fold_path = 'tmp'
        self.travel_mode_detector = None

    def evaluate_with_cross_validation(
            self,
            classes_dataframe,
            configuration_id,
            feature_columns,
            travel_mode_column,
            features_dataframe,
            metrics_filepath,
            detector_configuration,
            configuration_path,
            users_dataframe
    ):
        """
        :param classes_dataframe:
        :param configuration_id:
        :param feature_columns:
        :param travel_mode_column:
        :param features_dataframe:
        :param metrics_filepath:
        :param detector_configuration:
        :param configuration_path:
        :param users_dataframe:
        :return:
        """

        # search for best ensemble structure with full dataset
        search_dataframe = pd.concat(
            [features_dataframe, classes_dataframe],
            axis=1,
            sort=False
        )
        self.travel_mode_detector = self.detector_type(save_path=configuration_path)
        self.travel_mode_detector.fit(
            data_frame=search_dataframe.copy(),
            travel_mode_column=travel_mode_column,
            train_ratio=0.5,
            val_ratio=0.5,
            test_ratio=0.0,
            **detector_configuration
        )

        super(AutoTMDExperiment, self).evaluate_with_cross_validation(
            classes_dataframe, configuration_id, feature_columns,
            travel_mode_column,
            features_dataframe, metrics_filepath,
            detector_configuration, configuration_path, users_dataframe
        )

    def get_travel_mode_detector(self, fold_path):
        """
        Retrieves Travel Mode Detector Instance
        :param fold_path:
        :return:
        """
        self.fold_path = fold_path
        return self.travel_mode_detector

    def fit_travel_mode_detector(
            self,
            detector_configuration,
            travel_mode_detector,
            train_dataframe,
            travel_mode_column,
            **kwargs
    ):
        """
        Trains travel mode detector instance
        :param detector_configuration:
        :param travel_mode_detector:
        :param train_dataframe:
        :param travel_mode_column:
        :param kwargs:
        :return:
        """
        travel_mode_detector.refit(
            data_frame=train_dataframe.copy(),
            travel_mode_col=travel_mode_column,
            train_ratio=1.0,
            val_ratio=0.0,
            test_ratio=0.0,
            model_path=self.fold_path,
            **detector_configuration
        )
