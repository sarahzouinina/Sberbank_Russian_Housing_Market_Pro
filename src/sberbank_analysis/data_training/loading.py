import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
np.random.seed(0)

class Loader:

    def load_data(self, training_data_path_str, testing_data_path_str):
        print("Loading:", training_data_path_str , "...")
        training_set_df = pd.read_csv(training_data_path_str)
        print("Loading:", testing_data_path_str, "...")
        testing_set_df = pd.read_csv(testing_data_path_str)
        return training_set_df, testing_set_df

    def display_head(self, data, num_rows=5):
        return print(data.head(num_rows))
