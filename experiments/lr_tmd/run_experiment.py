from experiments.tmd_experiment_base import TMDExperiment
from os import path
from detectors.lr_tmd import LogisticRegressionTMD

experiment = TMDExperiment(
    experiment_path=path.abspath(path.dirname(__file__)),
    detector_type=LogisticRegressionTMD
)
experiment.run()

