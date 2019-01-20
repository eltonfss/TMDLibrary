from examples.util import get_tmd_dataset
from detectors.nb_tmd import NaiveBayesTMD
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    # load dataset
    df = get_tmd_dataset()
    travel_mode_column = 'target'

    # train model
    nb_tmd = NaiveBayesTMD()
    nb_tmd.fit(
        data_frame=df,
        travel_mode_column=travel_mode_column,
    )

    # load model and evaluate accuracy
    labeled_modes = df.pop(travel_mode_column)
    detected_modes = nb_tmd.predict(df)
    print('Full Data Accuracy', accuracy_score(labeled_modes, detected_modes))
