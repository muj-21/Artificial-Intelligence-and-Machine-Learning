# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 19:15:06 2018

@author: Sahil
"""
# import required libraries
import numpy as np
import math

# function to calculate root absolute error
def rel_abs_error(y_train, y_pred, y_test):
    a = np.mean(y_train) # mean of the training data
    sum_top = 0 # init value for the numerator in rae equation
    sum_bottom = 0 # init value for denominator in rae equation
    list = range(99) # list to loop through, same length as y_lin and y_test
    for i in list: # begin loop
            top = abs(y_pred[i] - y_test[i]) # numerator |pred(i) - real(i)|
            sum_top += top # complete numerator for rae equation sum of  |pred(i) - real(i)| for all values of i
            bottom = abs(y_test[i] - a) # denominator |real(i) - mean of training data values|
            # complete denominator for rae equation sum of |real(i) - mean of training data values| for all values of i            
            sum_bottom += bottom 
            
    rae = sum_top / sum_bottom # value for rae
    return rae # return rae to ModelEvaluation function