# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 19:13:15 2018

@author: Sahil
"""
# import required libraries
import numpy as np
import math

# function to root relative squared error
def RRSE(y_train, y_pred, y_test):
    a = np.mean(y_train) # mean of the training data
    sum_top = 0 # init value for the numerator in rrse equation
    sum_bottom = 0 # init value for the denominator in rrse equation
    list = range(99) # list to loop through, same length as y_lin and y_test
    for i in list:# begin loop
        top = (y_pred[i] - y_test[i])**2  # numerator [pred(i) - real(i)]^2
        sum_top += top # complete numerator for rrse equation sum of  [pred(i) - real(i)]^2 for all values of i
        bottom = (y_test[i] - a)**2 # denominator [real(i) - mean of training data values]^2
        # complete denominator for rae equation sum of [real(i) - mean of training data values]^2 for all values of i
        sum_bottom += bottom
    
    rse = sum_top / sum_bottom # relative squared error
    rrse = math.sqrt(rse) # root relative squared error
    return rrse # return rrse to ModelEvaluation function