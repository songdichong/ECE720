import math
import numpy as np
import cv2
from numba import jit
import torch
import os
from random import randrange, uniform
import pandas as pd
def gaussianKernal(l = 5, sign = 1.):
    ax = np.linspace(-(l-1)/2., (l-1)/2., l)
    guass = np.exp(-(ax**2)/(2.0*sign**2))
    kernel = np.outer(guass, guass)
    return kernel/np.sum(kernel)

def applyFilter(original, startX, startY, endX, endY, kernel):
    result = np.array(original, dtype="float32", copy=True)  
    kernel_x, kernel_y = kernel.shape
    k_row, k_col = int(kernel_x/2), int(kernel_y/2)
    for x in range(startX, endX):
        for y in range(startY, endY):
            x0 = min(original.shape[0], max(0, x-k_row))
            x1 = min(original.shape[0], max(0, x + k_row+1))
            y0 = min(original.shape[1], max(0, y-k_col))
            y1 = min(original.shape[1], max(0, y+k_col+1))
            patch = result[x0 : x1, y0 : y1]
            result[x][y][:] = np.array(cv2.filter2D(patch, -1, kernel)[k_row, k_col][:], copy=True)
    return result


@jit
def cal_blur(imgarray, theta, delta, L, S=0):
    imgheight = imgarray.shape[0]
    imgwidth = imgarray.shape[1]
    c0 = int(imgheight / 2)
    c1 = int(imgwidth / 2)
    theta = theta / 180 * math.pi
    delta = delta / 180 * math.pi
    blurred_imgarray = np.copy(imgarray)
    for x in range(0, imgheight):
        for y in range(0, imgwidth):
            R = math.sqrt((x - c0) ** 2 + (y - c1) ** 2)
            alpha = math.atan2(y - c1, x - c0)
            X_cos = L * math.cos(delta) - S * R * math.cos(alpha)
            Y_sin = L * math.sin(delta) - S * R * math.sin(alpha)
            N = int(
                max(
                    abs(R * math.cos(alpha + theta) + X_cos + c0 - x),
                    abs(R * math.sin(alpha + theta) + Y_sin + c1 - y),
                )
            )
            if N <= 0:
                continue
            count = 0
            sum_r, sum_g, sum_b = 0, 0, 0
            for i in range(0, N + 1):
                n = i / N
                xt = int(R * math.cos(alpha + n * theta) + n * X_cos + c0)
                yt = int(R * math.sin(alpha + n * theta) + n * Y_sin + c1)
                if xt < 0 or xt >= imgheight:
                    continue
                elif yt < 0 or yt >= imgwidth:
                    continue
                else:
                    sum_r += imgarray[xt, yt][0]
                    sum_g += imgarray[xt, yt][1]
                    sum_b += imgarray[xt, yt][2]
                    count += 1
            blurred_imgarray[x, y] = np.array(
                [sum_r / count, sum_g / count, sum_b / count]
            )
    return blurred_imgarray

def blur_attack(img):
    endX = img.shape[0]
    endY = img.shape[1]
    midX = int(endX/2)
    midY = int(endY/2)
    blurX = int(midX*1)
    blurY = int(midY*1)
    startX = midX - blurX
    endX = midX + blurX
    startY = midY - blurY
    endY = midY + blurY

    theta = 1.5
    delta = 0
    L = 3
    img = img[0]
    img = torch.from_numpy(img).permute(1, 2, 0)
    img = img.numpy().astype("uint8")
    print(img.shape)
    blurred_img = cal_blur(img[startX:endX, startY:endY], theta, delta, L).astype("uint8")
    img[startX:endX, startY: endY] = blurred_img[:,:]
    out = torch.from_numpy(img).permute(2, 0, 1)
    out = out.unsqueeze(0)
    out = out.numpy()

    return out

if __name__ == "__main__":
    DIR_PATH = "./bdd100k"
    OUT_PATH = "./blurred/"

    model = torch.hub.load("ultralytics/yolov5", "yolov5s")  
    param_dict = {}
    for root, dirs, files in os.walk(DIR_PATH):
        for file in files:
            path = os.path.join(DIR_PATH, file)
            img = cv2.imread(path).astype("uint8")
            img2 = img.copy()
            endX = img.shape[0]
            endY = img.shape[1]
            midX = int(endX/2)
            midY = int(endY/2)
            true_results = model(img2)
            true_labels = true_results.pandas().xyxy[0]['class']
            fileSaveDir = OUT_PATH + file[:-4]
            true_results.save(save_dir = fileSaveDir)
            true_labels_unique, true_labels_counter = np.unique(true_labels, return_counts=True)
            true_labels_saved = np.asarray((true_labels_unique, true_labels_counter)).T.tolist()

            for i in range(5):
                blurX = randrange(midX//10, midX+1)
                blurY = randrange(midX//10, midY+1)
                # blurX = int(midX*0.34)
                # blurY = int(midY*0.42)
                fileSaveName = fileSaveDir + "_" + str(i) + ".jpg"
                blurXpercent = blurX / midX
                blurYpercent = blurY / midY
                startX = midX - blurX
                endX = midX + blurX
                startY = midY - blurY
                endY = midY + blurY

                theta = uniform(0, 21)
                delta = uniform(0, 21)
                L = uniform(0, 21)
                print(img.shape)
                blurred_img = cal_blur(img[startX:endX, startY: endY], theta, delta, L)
                img[startX:endX, startY: endY] = blurred_img[:,:]
                
                pred_results = model(img)
                pred_labels = pred_results.pandas().xyxy[0]['class']
                pred_labels_unique, pred_labels_counter = np.unique(pred_labels, return_counts=True)
                pred_labels_saved = np.asarray((pred_labels_unique, pred_labels_counter)).T.tolist()

                label = 0 # 0 = good, 1 = miss class, 2 =  more class, 3 = number not matched
                if len(true_labels_saved) > len(pred_labels_saved):
                    label = 1
                elif len(true_labels_saved) < len(pred_labels_saved):
                    label = 2
                elif len(true_labels_saved) == len(pred_labels_saved):
                    for i in range(len(true_labels_saved)):
                        if true_labels_saved[i][0] != pred_labels_saved[i][0]:
                            label = 3 
                        elif true_labels_saved[i][1] != pred_labels_saved[i][1]:
                            label = 3

                param_dict[fileSaveName] = [blurXpercent, blurYpercent, theta, delta, L, true_labels_saved ,pred_labels_saved, label]
                pred_results.save(save_dir = fileSaveDir)
    data = pd.DataFrame.from_dict(param_dict, orient='index',
                        columns=['Xpercent', 'Ypercent', 'theta', 'delta', 'L', 'true_labels', 'pred_labels', 'label'])
    #data = data.drop(columns=['true_labels', 'pred_labels'], axis=1)
    data.to_csv("./params.csv")
    # data_x = data.iloc[:, :-1]
    # data_x['Xpercent'] = data_x['Xpercent'] * 100
    # data_x['Ypercent'] = data_x['Ypercent'] * 100
    # print(data_x)
    # data_y = data.iloc[:, -1]
    # from sklearn.model_selection import train_test_split
    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn.model_selection import GridSearchCV
    # X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=123)

    # dt = DecisionTreeClassifier()
    # #Define the Decision Tree model with a range of hyperparameters to test
    # param_grid = {
    #     'criterion': ['gini', 'entropy'],
    #     'max_depth': [None, 2, 4, 6],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    # }
    # # Set up the GridSearchCV with k-fold cross-validation
    # k_folds = 5  # Choose the number of folds for cross-validation
    # grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=k_folds, scoring='accuracy', n_jobs=-1)
    # # Fit the GridSearchCV to our training data
    # grid_search.fit(X_train, y_train)
    # # Train the Decision Tree model with the best parameters and evaluate it on the test set
    # best_dt = grid_search.best_estimator_
    # print("Best parameters for decision tree:", grid_search.best_params_)
    # # Predict the test set labels and generate the truth table
    # y_pred = best_dt.predict(X_test)
    # print("decision tree")
    # print("accuracy  = ", best_dt.score(X_test, y_test))


    # from sklearn.ensemble import RandomForestClassifier
    # clf = RandomForestClassifier(random_state=123)
    # #Define the Random Forest model with a range of hyperparameters to test
    # param_grid = {
    #     'n_estimators': [10, 50, 100],
    #     'criterion': ['gini', 'entropy'],
    #     'max_depth': [None, 2, 4, 6, 8],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    # }
    # grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=k_folds, scoring='accuracy', n_jobs=-1)
    # # Fit the GridSearchCV to our training data
    # grid_search.fit(X_train, y_train)
    # # Train the Decision Tree model with the best parameters and evaluate it on the test set
    # best_dt = grid_search.best_estimator_
    # print("Best parameters for random forest:", grid_search.best_params_)
    # # Predict the test set labels and generate the truth table
    # y_pred = best_dt.predict(X_test)
    # print("random forest")
    # print("accuracy  = ", best_dt.score(X_test, y_test))

    # from sklearn import svm
    # clf = svm.SVC(random_state=123)
    # param_grid = {
    #     'C': [0.1, 1, 10],
    #     'kernel': ['linear', 'rbf'],
    #     'gamma': ['scale', 'auto']
    # }
    # grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=k_folds, scoring='balanced_accuracy', n_jobs=-1)
    # # Fit the GridSearchCV to our training data
    # grid_search.fit(X_train, y_train)
    # # Train the Decision Tree model with the best parameters and evaluate it on the test set
    # best_dt = grid_search.best_estimator_
    # print("Best parameters for svm:", grid_search.best_params_)
    # # Predict the test set labels and generate the truth table
    # y_pred = best_dt.predict(X_test)
    # print("svm")
    # print("accuracy  = ", best_dt.score(X_test, y_test))

    # #Artificial Neural Network
    # from keras.models import Sequential
    # from keras.layers import Dense
    # # create ANN model
    # model = Sequential()

    # # Defining the Input layer and FIRST hidden layer, both are same!
    # model.add(Dense(units=10, input_dim=5, kernel_initializer='normal', activation='relu'))
    # # Defining the Second & third layer of the model
    # # after the first layer we don't have to specify input_dim as keras configure it automatically
    # model.add(Dense(units=30, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(units=10, kernel_initializer='normal', activation='relu'))
    # # The output neuron is a single fully connected node 
    # # Since we will be predicting a single number
    # model.add(Dense(1, kernel_initializer='normal', activation="relu"))
    # # Compiling the model
    # model.compile(loss='mean_squared_error', optimizer='adam')

    # # Fitting the ANN to the Training set
    # model.fit(X_train, y_train, batch_size = 64, epochs = 100)
    # print("accuracy  = ", model.evaluate(X_test, y_test))