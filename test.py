import numpy as np

from sklearn.model_selection import LeaveOneOut 
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 2, 3])
loo = LeaveOneOut()
n_splits = loo.get_n_splits(X)
print(n_splits)
print(loo.split(X))


for train_index, test_index in loo.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
#    print(X_train, X_test, y_train, y_test)