#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import model_vistraining
import numpy as np
import sklearn.datasets as datasets
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import pandas as pd



"""
this file gives a example of how to use this project to preform training 
visualization.
This is a test of when using feature and target
"""

        
cancer = datasets.load_breast_cancer()
feature,target = cancer.data,cancer.target

base = model_vistraining.Base(model = DecisionTreeClassifier,
                              parameter_dict = {"criterion":["gini", "entropy", "log_loss"],
                                                "max_depth":[2,4,6,8,10,12],
                                                "min_samples_split":[2],
                                                "max_features":["sqrt","log2",0.83]},
                                        feature = pd.DataFrame(feature),
                                        target = target,
                                        prediction_type="classification",
                                        feature_names = cancer.feature_names
                                        )




base.GridSearch(time_for_each_param=1)
#base.open_html_report()


        

