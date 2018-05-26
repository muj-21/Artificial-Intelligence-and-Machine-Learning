# -*- coding: utf-8 -*-
"""
@author: Sahil
"""
# import all of a libraries required
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.patches as mpatches
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import ModelEvaluation

def CreateModel():
    # variable for filepath of dataset
    address = 'student-maths.csv'
    # convert file into pd data frame for to allow data to be processed
    data = pd.read_csv(address)
    data.head()
    # assign X to feature data from csv
    X = (data.iloc[:, 0:6].values).astype(float)
    
    # assign y to output data from csv
    y = (data.iloc[:, 6].values).astype(float)
    
    # split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.25)
    
    # scale the data with StandardScalar and choose SVR kernel
    pipeline = Pipeline([('scaler', StandardScaler()), ('estimator', SVR(kernel="linear"))])
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1,1))
    # train the model using the training data
    pipeline.fit(X_train, y_train.ravel())
    # male predictions on the test data
    y_pred = y_scaler.inverse_transform(pipeline.predict(X_test))
    
    
    # plot styling parameters
    plt.style.use('ggplot')
    plt.figure(figsize=(16, 12))
    ageColour = mpatches.Patch(color='red', label='Student Age')
    g2Colour = mpatches.Patch(color='yellowgreen', label='Student G2')
    g1Colour = mpatches.Patch(color='orange', label='Student G1')
    absColour = mpatches.Patch(color='silver', label='Student Absences')
    failColour = mpatches.Patch(color='purple', label='Student Past Failures')
    stColour = mpatches.Patch(color='cornflowerblue', label='Student Study Time')
    # plot a chart of actual values vs predicted values
    plt.plot(X_test, y_pred, 'x', markersize=10)
    
    # define label for x axis
    plt.xlabel('Features', fontsize=20)
    # define label for y axis
    plt.ylabel('Final Grade', fontsize=20)
    # insert title of the chart
    plt.title('Chart of Features vs Final Grade', fontsize=20)
    # add a legend to show labels of plot
    plt.legend(handles=[ageColour, g2Colour, absColour, stColour, failColour, g1Colour])
    # show model
    plt.show()
    # calculate the k-fold accuracy with 10 folds
    cv = cross_val_score(SVR(kernel="linear"), X_train, y_train.ravel(), cv=10)
    # run the model evaluation program from ModelEvaluation file
    ModelEvaluation.ModelEvaluation(y_test, y_train, y_pred, "SVM REGRESSION", cv)