# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:29:32 2018

@author: mujta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.patches as mpatches
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import ModelEvaluation

def CreateModel():
    #Using panda dataframes to import the dataset  
    path = 'student-maths.csv'
    data = pd.read_csv(path)
    data.head()
     
    #Picking out the features and converting the values into float 
    X = (data.iloc[:, 0:6].values).astype(float)
    y = (np.array(data.iloc[:,6].values)).astype(float)
    
    #split the data into X_train, X_test, y_train and y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.25)
    
    #Create linear regression object
    regr = linear_model.LinearRegression()
    
    #Assembles the parameters and scales the data
    pipeline = Pipeline([('scaler', StandardScaler()), ('estimator', regr)])
    
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1,1))
    #Using the trained data to train the model
    pipeline.fit(X_train, y_train.ravel())
    #makes the predictions based on the test data
    y_pred = y_scaler.inverse_transform(pipeline.predict(X_test))
    
    

    #Plot outputs
    #differentiate the plots by styling them
    plt.style.use('ggplot')
    ageColour = mpatches.Patch(color='red', label='Student Age')
    g2Colour = mpatches.Patch(color='yellowgreen', label='Student G2')
    g1Colour = mpatches.Patch(color='orange', label='Student G1')
    absColour = mpatches.Patch(color='silver', label='Student Absences')
    failColour = mpatches.Patch(color='purple', label='Student Past Failures')
    stColour = mpatches.Patch(color='cornflowerblue', label='Student Study Time')
    
    #change size of the figure
    plt.figure(figsize=(6,6))
    #plot the attributes
    plt.plot(X_test, y_pred, 'x')
    #X label
    plt.xlabel('Features')
    #y label
    plt.ylabel('Predicted Final Year Grade')
    #legend to show which plot iswhich
    plt.legend(handles=[ageColour, g1Colour, g2Colour, absColour, failColour, stColour])
    #configure graph
    plt.show()
    #cross validation scores
    scores = (cross_val_score(regr, X_train, y_train, cv=10))
    #Code is on Model Evaluation file
    ModelEvaluation.ModelEvaluation(y_test, y_train, y_pred, "LINEAR REGRESSION", scores)
    
