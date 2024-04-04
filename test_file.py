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
This is a test of when using data generator.
"""

class Counter:
    def __init__(self, feature,target):
        self.feature = feature
        self.target = target

    def __iter__(self):
        return self

    def __next__(self): # Python 2: def next(self)
        index = np.random.randint(len(self.target), size=130)
        index2 = np.random.randint(len(self.target), size=30)
        return (self.feature[index,:],self.target[index]),(self.feature[index2,:],self.target[index2])
        raise StopIteration
        
wine = datasets.load_wine()
feature,target = wine.data,wine.target
counter = Counter(feature,target)


base = model_vistraining.Base(model = DecisionTreeClassifier,
                              parameter_dict = {"criterion":["gini", "entropy", "log_loss"],
                                                "max_depth":[2,4,6,8,10,12],
                                                "min_samples_split":[2,5,7],
                                                "max_features":["sqrt","log2",0.83]},
                                        data_generator=counter,
                                        prediction_type="multi_classification")




base.GridSearch(time_for_each_param=1)
base.open_html_report()


        

