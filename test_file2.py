#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import model_vistraining
import numpy as np
import sklearn.datasets as datasets
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor




"""
this file gives a example of how to use this project to preform training 
visualization.
This is a test of when using feature and target
"""

        
wine = datasets.load_wine()
feature,target = wine.data,wine.target

base = model_vistraining.Base(model = DecisionTreeRegressor,
                              parameter_dict = {"criterion":["absolute_error","poisson","squared_error"],
                                                "max_depth":[2,4,6,8,10,12],
                                                #"min_samples_split":[2,5,7],
                                                "max_features":["sqrt","log2",0.83]},
                                        feature = feature,
                                        target = target,
                                        prediction_type="regression")




base.GridSearch(time_for_each_param=1)
base.open_html_report()


        

