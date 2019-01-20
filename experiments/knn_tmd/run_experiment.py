from experiments.tmd_experiment_base import TMDExperiment
from os import path
from detectors.knn_tmd import KNearestNeighborsTMD

experiment = TMDExperiment(
    experiment_path=path.abspath(path.dirname(__file__)),
    detector_type=KNearestNeighborsTMD
)
experiment.run()

