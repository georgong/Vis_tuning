#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:24:55 2024

@author: gongzhenghao
"""

import model_vistraining
import numpy as np
import sklearn.datasets as datasets
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression




"""
this file gives a example of how to use this project to preform training 
visualization.
This is a test of when using feature and target
"""

        
cancer = datasets.load_breast_cancer()
feature,target = cancer.data,cancer.target


class LogisticRegressionWarpper:
    def __init__(self,C = 1,penalty = "l2"):
        self.LogisticRegression = LogisticRegression(C=C,penalty = penalty)
    def fit(self,X,y):
        self.LogisticRegression.fit(X,y)
        return self
    def predict_proba(self,X):
        return self.LogisticRegression.predict_proba(X)[:,1]
    
    


base = model_vistraining.Base(model = LogisticRegressionWarpper,
                              parameter_dict = {"C":[0.1,0.3,0.5,0.7,0.9],
                                                "penalty":["l2"],
                                               },
                                        feature = feature,
                                        target = target,
                                        prediction_type="classification_proba",
                                        predict_method="predict_proba")




base.GridSearch(time_for_each_param=5)
base.open_html_report()


        

