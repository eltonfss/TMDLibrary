from experiments.tmd_experiment_base import TMDExperiment
from detectors.rf_tmd import RandomForestTMD
from os import path

experiment = TMDExperiment(
    experiment_path=path.abspath(path.dirname(__file__)),
    detector_type=RandomForestTMD
)
experiment.run()

