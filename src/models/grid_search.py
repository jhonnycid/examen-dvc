import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
import pickle

param_grid = {'alpha': [0.01, 0.1, 1, 10, 100], 'l1_ratio': [0.1, 0.25, 0.5, 0.75, 0.9]} 
model = ElasticNet()
X_train_scaled = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')
y_train = np.ravel(y_train)

grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train_scaled, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best neg MSE:", grid_search.best_score_)

best_params = grid_search.best_params_

#Save best parameters to a pickle file
with open("models/best_params.pkl", "wb") as f:
    pickle.dump(best_params, f)
