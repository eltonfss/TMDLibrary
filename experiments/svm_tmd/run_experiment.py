from experiments.tmd_experiment_base import TMDExperiment
from detectors.svm_tmd import SupportVectorMachineTMD
from os import path

experiment = TMDExperiment(
    detector_type=SupportVectorMachineTMD,
    experiment_path=path.abspath(path.dirname(__file__))
)
experiment.run()

