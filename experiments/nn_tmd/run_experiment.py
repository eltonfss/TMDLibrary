from experiments.tmd_experiment_base import TMDExperiment
from os import path
from detectors.nn_tmd import NeuralNetworkTMD

experiment = TMDExperiment(
    experiment_path=path.abspath(path.dirname(__file__)),
    detector_type=NeuralNetworkTMD
)
experiment.run()

