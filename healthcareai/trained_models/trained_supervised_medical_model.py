"""A Trained Supervised Medical Model."""
import time
import pandas as pd
from datetime import datetime
import healthcareai.common.file_io_utilities as hcai_io

class TrainedSupervisedMedicalModel(object):
    def __init__(self, model, features, algorithm):
        self.model = model
        self.features = features
        # self.type = type
        # self.score = score
        self.algorithm = algorithm

    def save(self, filename=None, debug=True):
        """
        Save this object to a pickle file with the given file name.
        
        Args:
            filename (str): Optional filename override. Defaults to `timestamp_<MODEL_TYPE>_<ALGORITHM_NAME>.pkl`. For
                example: `2017-05-27T09-12-30_regression_LinearRegression.pkl`
            debug (bool): Print debug output to console by default
        """
        if filename is None:
            time_string = time.strftime("%Y-%m-%dT%H-%M-%S")
            filename = '{}_{}.pkl'.format(time_string, self.algorithm)

        hcai_io.save_object_as_pickle(self, filename)

        if debug:
            print('Trained {} model saved as {}'.format(self.algorithm, filename))

    def make_predictions(self, dataframe):
        # Run the raw dataframe through the preparation process
        # Run the raw dataframe through the preparation process
        print(self.model.estimator)
        print(self.model)
        y_predictions = self.model.predict(dataframe)

        # Create a new dataframe with the grain column from the original dataframe
        results = pd.DataFrame()
        results['Outcome'] = y_predictions

        return results
    