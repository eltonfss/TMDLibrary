from experiments.tmd_experiment_base import TMDExperiment
from os import path
from detectors.nb_tmd import NaiveBayesTMD

experiment = TMDExperiment(
    experiment_path=path.abspath(path.dirname(__file__)),
    detector_type=NaiveBayesTMD
)
experiment.run()

