from experiments.tmd_experiment_base import TMDExperiment
from os import path
from detectors.ab_tmd import AdaBoostTMD

experiment = TMDExperiment(
    experiment_path=path.abspath(path.dirname(__file__)),
    detector_type=AdaBoostTMD
)
experiment.run()

