from detectors.nn_tmd import NeuralNetworkTMD
from examples.util import get_tmd_dataset
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    # load dataset
    df = get_tmd_dataset()
    travel_mode_column = 'target'

    # train and save model
    neural_tmd = NeuralNetworkTMD()
    neural_tmd.fit(
        data_frame=df.copy(),
        travel_mode_col=travel_mode_column,
        max_epochs=500
    )

    # evaluate accuracy
    labeled_modes = df.pop(travel_mode_column)
    detected_modes = neural_tmd.predict(df.copy())
    print('Full Data Accuracy', accuracy_score(labeled_modes, detected_modes))
