import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# sklearn import
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from constants import SPLIT_RATE
# models import
from models.decision_tree import decision_tree
from models.k_nearest_neighbors import k_nearest_neighbors
from models.linear_regression import linear_regression

import warnings
warnings.filterwarnings('ignore')

if len(sys.argv) < 2:
  print("Missing parameter. Correct usage: python main.py [model]")
  exit()

# process raw data
data = pd.read_csv("./data/diabetes.csv")

# encode categorical features(this step is moot)

# train test split
y = data['Outcome']
X = data.drop(columns=['Outcome'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=SPLIT_RATE)

match sys.argv[1]:
  case 'decision_tree':
    decision_tree(X_train, y_train, X_test, y_test)
  case 'linear_regression':
    linear_regression(X_train, y_train, X_test, y_test)
  case 'k_nearest_neighbors':
    k_nearest_neighbors(X_train, y_train, X_test, y_test)
  case _:
    decision_tree(X_train, y_train, X_test, y_test)
