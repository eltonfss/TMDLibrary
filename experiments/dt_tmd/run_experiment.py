from experiments.tmd_experiment_base import TMDExperiment
from os import path
from detectors.dt_tmd import DecisionTreeTMD

experiment = TMDExperiment(
    experiment_path=path.abspath(path.dirname(__file__)),
    detector_type=DecisionTreeTMD
)
experiment.run()

