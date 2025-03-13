import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

X_train = pd.read_csv('data/processed_data/X_train.csv')
X_test = pd.read_csv('data/processed_data/X_test.csv')
output_root_filepath = 'data/processed_data'

#Transformation of X_train and X_test
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

for file, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
    output_filepath = os.path.join(output_root_filepath, f'{filename}.csv')
    file.to_csv(output_filepath, index=False)