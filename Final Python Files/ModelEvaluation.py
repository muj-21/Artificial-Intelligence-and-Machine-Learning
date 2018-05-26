# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 19:17:44 2018

@author: Sahil
"""
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import RRSE, rel_abs_error
from scipy.stats import spearmanr

def ModelEvaluation(y_test, y_train, y_pred, modelname, CV):
    # Model evaluation using accuracy tests
    rmse = (math.sqrt(mean_squared_error(y_test, y_pred)))
    print("*** " + modelname + " MODEL EVALUATION ***\nRoot mean-squared error: %.2f " % rmse)
    mae = (mean_absolute_error(y_test, y_pred))
    print('K-fold mean accuracy: %.2f' % (CV.mean()) )
    print('K-fold stdev accuracy: %.2f' % (CV.std()) )
    print("Mean absolute error: %.2f" % mae)
    spearmanr_coefficient, p_value = spearmanr(y_test, y_pred)
    print("Correlation coefficient: %.2f" % spearmanr_coefficient)
    print("R-Squared: %.2f" % r2_score(y_test, y_pred))
    # use the created rrse and rae functions to print the model testing values
    rrse= 100 * (RRSE.RRSE(y_train, y_pred, y_test))
    rae = 100 * (rel_abs_error.rel_abs_error(y_train, y_pred, y_test))
    print("Root relative squared error: %.1f" % rrse + "%")
    print("Relative absolute error: %.1f" % rae + "%")