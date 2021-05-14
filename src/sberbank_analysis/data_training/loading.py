import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
np.random.seed(0)

class Loader:

    def load_data(self, data_path ='data_path'):
        return pd.read_csv(data_path)
    def display_head(self, data, num_rows = 5):
        return print(data.head(num_rows))
