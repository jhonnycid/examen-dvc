#Libraries
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import pickle
import json

#import best model
model_filename = './models/gbr_model.pkl'
with open(model_filename, "rb") as f:
    model = pickle.load(f)

#import test set
X_test_scaled = pd.read_csv('data/processed_data/X_test_scaled.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv')
y_test = np.ravel(y_test)

y_pred = pd.DataFrame(model.predict(X_test_scaled), columns=['Predictions'])
y_pred.to_csv('data/predictions.csv', index=False)

test_score = model.score(X_test_scaled, y_test)  # R^2 score
mean_squared_error_score = mean_squared_error(y_test, y_pred)

metrics = {"R2_score": test_score, "MSE":mean_squared_error_score}

with open('metrics/scores.json', 'w') as f:
    json.dump(metrics, f)
print(f"Test R^2 score: {test_score}")
print(f"Test Mean Squared Error: {mean_squared_error_score}")
