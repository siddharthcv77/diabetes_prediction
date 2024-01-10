from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import mean_absolute_error, mean_squared_error

def k_nearest_neighbors(X_train, y_train, X_test, y_test):
  reg=KNeighborsClassifier(n_neighbors=3)
  reg.fit(X_train, y_train)

  y_pred = reg.predict(X_test)
  print(mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred)**0.5)
  # print(y_pred)
  # fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
  # plot_decision_function( X, y, knn, axs, title=f"Decision function for \n KNN")