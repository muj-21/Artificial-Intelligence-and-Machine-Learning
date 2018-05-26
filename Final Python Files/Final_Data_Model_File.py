# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 19:27:29 2018

@author: Sahil
"""
# import our 3 model files
import SVMRegressionSahil, DTRegressionKevin, LinearRegressionMujtaba, Pearson
# run data pre processing, pearsonr between all features, output:table
Pearson.Run()
print('\n')
# run SVR model by Sahil, output:chart and performance metrics
SVMRegressionSahil.CreateModel()
# run DT model by Kevin, output:performance metrics, decision tree...
# must be opened in WebGraphviz
print('\nExport "features.dot" to WebGraphviz to see Decision Tree\n')
DTRegressionKevin.CreateModel()
print('\n')
# run Linear Reg model by mujtaba, output:chart and performance metrics
LinearRegressionMujtaba.CreateModel()