import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
import os

##paths
input_filepath = 'data/raw_data/raw.csv'
output_root_filepath = 'data/processed_data'
file_url = 'https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv'

### getting file and saving into data/raw_data

response = requests.get(file_url)
if response.status_code == 200:
     # Process the response content as needed
    content = response.text
    text_file = open(input_filepath, "wb")
    text_file.write(content.encode('utf-8'))
    text_file.close()
else:
    print(f'Error accessing the object {file_url}:', response.status_code)

### getting raw_data dataframe and splitting 
df_raw = pd.read_csv(input_filepath)
X = df_raw.drop(['date', 'silica_concentrate'], axis=1)
y = df_raw['silica_concentrate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

### saving splitted files

for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
    output_filepath = os.path.join(output_root_filepath, f'{filename}.csv')
    file.to_csv(output_filepath, index=False)