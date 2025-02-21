import pickle
path = "data/lungcancer/lung_cancer_test.pkl"

import pandas as pd

data = pd.read_pickle(path)  # Directly load the DataFrame
