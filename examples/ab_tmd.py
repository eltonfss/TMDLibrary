from examples.util import get_tmd_dataset
from detectors.ab_tmd import AdaBoostTMD
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    # load dataset
    df = get_tmd_dataset()
    travel_mode_column = 'target'

    # train model
    ab_tmd = AdaBoostTMD()
    ab_tmd.fit(
        data_frame=df,
        travel_mode_column=travel_mode_column,
    )

    # load model and evaluate accuracy
    labeled_modes = df.pop(travel_mode_column)
    detected_modes = ab_tmd.predict(df)
    print('Full Data Accuracy', accuracy_score(labeled_modes, detected_modes))
