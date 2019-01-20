from experiments.automl_tmd_experiment import AutoTMDExperiment
from detectors.automl_tmd import AutoMLTMD
from os import path

experiment = AutoTMDExperiment(
    experiment_path=path.abspath(path.dirname(__file__)),
    detector_type=AutoMLTMD
)
experiment.run()

