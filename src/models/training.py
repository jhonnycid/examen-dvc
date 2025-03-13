import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
import pickle

with open("models/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

X_train_scaled = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')
y_train = np.ravel(y_train)

model = ElasticNet()
model.set_params(**best_params)
model.fit(X_train_scaled, y_train)

#--Save the trained model to a file
model_filename = './models/gbr_model.pkl'
with open(model_filename, "wb") as f:
    pickle.dump(model, f)
print("Model trained and saved successfully.")