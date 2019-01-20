from examples.util import get_tmd_dataset
from detectors.lr_tmd import LogisticRegressionTMD
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    # load dataset
    df = get_tmd_dataset()
    travel_mode_column = 'target'

    # train model
    lr_tmd = LogisticRegressionTMD()
    lr_tmd.fit(
        data_frame=df,
        travel_mode_column=travel_mode_column,
    )

    # load model and evaluate accuracy
    labeled_modes = df.pop(travel_mode_column)
    detected_modes = lr_tmd.predict(df)
    print('Full Data Accuracy', accuracy_score(labeled_modes, detected_modes))
