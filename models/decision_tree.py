import sys
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def decision_tree(X_train, y_train, X_test, y_test):
  max_depth_list = [12,20,22,24,26,28,30,32,34,36,40]
  min_test_error = sys.maxsize
  depth_with_min_error = -1

  for index, max_depth in enumerate(max_depth_list):
    decisiontree = DecisionTreeRegressor(max_depth = max_depth, min_samples_split = 2)
    decisiontree.fit(X_train, y_train)
    error = mean_absolute_error(y_test, decisiontree.predict(X_test))
    if(error < min_test_error):
      min_test_error = error
      depth_with_min_error = max_depth_list[index]

  print("depth with min error: ", depth_with_min_error)
  print("min_error: ", min_test_error)