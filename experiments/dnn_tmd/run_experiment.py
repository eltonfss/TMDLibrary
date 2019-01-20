from experiments.tmd_experiment_base import TMDExperiment
from os import path
from detectors.dnn_tmd import DeepNeuralNetworkTMD

experiment = TMDExperiment(
    experiment_path=path.abspath(path.dirname(__file__)),
    detector_type=DeepNeuralNetworkTMD
)
experiment.run()

