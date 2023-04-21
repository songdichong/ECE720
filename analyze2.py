import pandas as pd
data = pd.read_csv("./params0.csv")
data = data.drop(columns=["true_labels","pred_labels"])
data = data.iloc[:, 1:]

data_x = data.iloc[:, :-1]
data_y = data.iloc[:, -1]
data_y[data_y != 0] = 1

print(data_y[data_y == 1].count())

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=123)
from sklearn import tree
dt = DecisionTreeClassifier()
#Define the Decision Tree model with a range of hyperparameters to test
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 4, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
# Set up the GridSearchCV with k-fold cross-validation
k_folds = 5  # Choose the number of folds for cross-validation
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=k_folds, scoring='accuracy', n_jobs=-1)
# Fit the GridSearchCV to our training data
grid_search.fit(X_train, y_train)
# Train the Decision Tree model with the best parameters and evaluate it on the test set
best_dt = grid_search.best_estimator_
print("Best parameters for decision tree:", grid_search.best_params_)
# Predict the test set labels and generate the truth table
y_pred = best_dt.predict(X_test)
print("decision tree")
print("accuracy  = ", best_dt.score(X_test, y_test))
text_representation = tree.export_text(best_dt)
print(text_representation)
tree.plot_tree(best_dt)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=123)
#Define the Random Forest model with a range of hyperparameters to test
param_grid = {
    'n_estimators': [10, 50, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 4, 6, 8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=k_folds, scoring='accuracy', n_jobs=-1)
# Fit the GridSearchCV to our training data
grid_search.fit(X_train, y_train)
# Train the Decision Tree model with the best parameters and evaluate it on the test set
best_dt = grid_search.best_estimator_
print("Best parameters for random forest:", grid_search.best_params_)
# Predict the test set labels and generate the truth table
y_pred = best_dt.predict(X_test)
print("random forest")
print("accuracy  = ", best_dt.score(X_test, y_test))
text_representation = tree.export_text(best_dt)
print(text_representation)
tree.plot_tree(best_dt)
from sklearn import svm
clf = svm.SVC(random_state=123)
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='balanced_accuracy', n_jobs=-1)
# Fit the GridSearchCV to our training data
grid_search.fit(X_train, y_train)
# Train the Decision Tree model with the best parameters and evaluate it on the test set
best_dt = grid_search.best_estimator_
print("Best parameters for svm:", grid_search.best_params_)
# Predict the test set labels and generate the truth table
y_pred = best_dt.predict(X_test)
print("svm")
print("accuracy  = ", best_dt.score(X_test, y_test))
# for xypercent in range(100):
#     for theta in range(100):
#         for delta in range(100):
#             for L in range(100):
#                 pred = best_dt.predict([[xypercent, xypercent, theta, delta, L]])
#                 if pred == 1:
#                     print("xypercent = ", xypercent, "theta = ", theta, "delta = ", delta, "L = ", L)
#                     break