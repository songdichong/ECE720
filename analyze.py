import pandas as pd
data = pd.read_csv("./params0.csv")
data = data.drop(columns=["true_labels","pred_labels"])
data = data.iloc[:, 1:]
print(data)

data_x = data.iloc[:, :-1]
data_x['Xpercent'] = data_x['Xpercent']
data_x['Ypercent'] = data_x['Ypercent'] 

print(data_x)
data_y = data.iloc[:, -1]
data_y[data_y != 0] = 1

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=123)

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

from sklearn import svm
clf = svm.SVC(random_state=123)
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=k_folds, scoring='balanced_accuracy', n_jobs=-1)
# Fit the GridSearchCV to our training data
grid_search.fit(X_train, y_train)
# Train the Decision Tree model with the best parameters and evaluate it on the test set
best_dt = grid_search.best_estimator_
print("Best parameters for svm:", grid_search.best_params_)
# Predict the test set labels and generate the truth table
y_pred = best_dt.predict(X_test)
print("svm")
print("accuracy  = ", best_dt.score(X_test, y_test))

# #Artificial Neural Network
# from keras.models import Sequential
# from keras.layers import Dense
# # create ANN model
# model = Sequential()

# # Defining the Input layer and FIRST hidden layer, both are same!
# model.add(Dense(units=10, input_dim=5, kernel_initializer='normal', activation='relu'))
# # Defining the Second & third layer of the model
# # after the first layer we don't have to specify input_dim as keras configure it automatically
# model.add(Dense(units=20, kernel_initializer='normal', activation='relu'))
# model.add(Dense(units=30, kernel_initializer='normal', activation='relu'))
# model.add(Dense(units=10, kernel_initializer='normal', activation='relu'))

# # The output neuron is a single fully connected node 
# # Since we will be predicting a single number
# model.add(Dense(1, kernel_initializer='normal', activation="relu"))
# # Compiling the model
# model.compile(loss='mean_squared_error', optimizer='adam')

# # Fitting the ANN to the Training set
# model.fit(X_train, y_train, batch_size = 128, epochs = 100)
# print("predict", model.predict(X_test))
# print("truth", y_test)