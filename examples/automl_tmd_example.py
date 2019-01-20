from examples.util import get_tmd_dataset
from detectors.automl_tmd import AutoMLTMD
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    # load dataset
    df = get_tmd_dataset()
    travel_mode_column = 'target'

    # train model
    automl_tmd = AutoMLTMD()
    automl_tmd.fit(
        data_frame=df,
        travel_mode_column=travel_mode_column,
        maximum_search_time=300
    )

    # load model and evaluate accuracy
    labeled_modes = df.pop(travel_mode_column)
    detected_modes = automl_tmd.predict(df, batch_size=100)
    print('Full Data Accuracy', accuracy_score(labeled_modes, detected_modes))
