#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:09:52 2024

@author: gongzhenghao
"""
from flask import Flask, render_template, request
import img_generation
import numpy as np
import pandas as pd
from operator import methodcaller
import itertools
from sklearn.metrics import accuracy_score,f1_score,hinge_loss,recall_score,precision_score
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,max_error
from sklearn.metrics import roc_auc_score,average_precision_score,top_k_accuracy_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,roc_curve,brier_score_loss
import time
import json as js

#TODO add a attribute called total_history
#total_history keep record of each k-fold
#[hyperparam],[feature],[predicted_val],[actual_val]
# max_length,


class Base:
    def __init__(self,model,parameter_dict,feature=None,target=None,data_generator = None,prediction_type:str = "classification",fit_method:str = "fit",predict_method:str = "predict",total_history = True):
        """
        

        Parameters
        ----------
        model : TYPE
            DESCRIPTION.
        feature : TYPE, optional
            DESCRIPTION. The default is None.
        target : TYPE, optional
            DESCRIPTION. The default is None.
        data_generator : iterator object optional
            This generator is used when the feature, target parameter are set to None
            data_generator need to be an iterator which can provide a tuple contains (train_set,test_set)
            this is useful when sometimes the feature cannot do simple train_test split or k-fold. 
            For instance, the time-series data which need special train_test split method.
            you may need to construct your own data_generator
            DESCRIPTION. The default is None.
        prediction_type : str, optional
            DESCRIPTION. The default is "classification".
        fit_method : str, optional
            DESCRIPTION. The default is "fit".
        predict_method : str, optional
            DESCRIPTION. The default is "predict".

        Returns
        -------
        None.

        """
        # check if the model has fit-predicted method
        assert hasattr(model,fit_method),f"Model does not has the assigned fit method \"{fit_method}\""
        assert hasattr(model,predict_method),f"Model does not has the assigned fit method \"{predict_method}\""
        
        assert not all([type(data) == type(None) for data in [feature,target,data_generator]]),"feature,target,and data_generator cannot all be None"
        if data_generator == None:
            assert isinstance(feature, np.ndarray) or isinstance(feature, pd.DataFrame),"we currently only accept these two type of feature, maybe accpet more type in future"
            assert isinstance(feature, np.ndarray) or isinstance(feature, pd.Series),"we currently only accept these two type of target, maybe accpet more type in future"
            assert not any([type(data) == type(None) for data in [feature,target]]),"if using feature and target, you should put both parameters when initiatation"
            assert len(feature) == len(target),f"feature and target has different length,feature:{len(feature)},target:{len(target)}"
            
        else:
            assert all([type(data) == type(None) for data in [feature,target]]),"when using data_generator, feature and target should set to None"
            assert hasattr(data_generator,"__iter__"),"f data_generator does not has __iter__ method"
        # check if the feature and target are valid
        assert prediction_type in ["classification","multi_classification","regression","classification_proba"],"Find a prediction type not in valid!"
        self.model =  model
        self.parameter_dict = parameter_dict
        keys, values = zip(*parameter_dict.items())
        self.parameter_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
        self.selected_parameter = self.parameter_dict_list[0]
        self.search_history = "No Search History now!"; # this is a dataframe recording the result of hypeparameter
        self.prediction_type = prediction_type
        self.fit_method = fit_method
        self.predict_method = predict_method
        self.total_history = "No Total History now!"
        if data_generator == None:
            if isinstance(feature,np.ndarray):
                self.feature = pd.DataFrame(feature,columns = [f"variable{i}" for i in range(len(feature[0]))])
            else:
                self.feature = feature
            if isinstance(target,np.ndarray):
                self.target = target
            else:
                self.target = target.to_numpy
            self.data_generator = None
        else:
            self.data_generator = data_generator
            self.feature = None
            self.target = None;    
    
    def k_fold(self,k=4,parameter = None):
        """
        k_fold_size would be used when using data_generator, it will generate k_fold_size * k data  
        from data_generator in total
        """
        if parameter == None:
            parameter = self.selected_parameter;
        predict_value = None
        actual_value = None
        
        
        if self.data_generator != None:
            predict_value = []
            actual_value = []
            k_fold_feature = []
            iterator = iter(self.data_generator)
            
            for i in range(k):
                train_set,test_set = next(iterator)
                train_feature,train_target = train_set[0],train_set[1]
                test_feature,test_target = test_set[0],test_set[1]
                #model = self.model(**parameter).fit(train_feature,train_target)
                #predict_value.append(model.predict(test_feature))
                mc = methodcaller(self.fit_method,train_feature,train_target)
                mp = methodcaller(self.predict_method,test_feature)
                model = mc(self.model(**parameter))
                predict_value.append(mp(model))
                actual_value.append(test_target)
                k_fold_feature.append(test_feature)
            predict_value = np.hstack(predict_value)
            actual_value = np.hstack(actual_value)
            k_fold_feature = np.vstack(k_fold_feature)
            k_fold_feature = pd.DataFrame(k_fold_feature,columns = [f"variable{i}" for i in range(len(k_fold_feature[0]))])
            
        
        if self.data_generator == None:
            predict_value = np.zeros(len(self.target))
            actual_value = np.zeros(len(self.target))
            k_fold_feature = self.feature
            feature,target = self.feature,self.target
            k_fold_mask = np.random.choice(range(k),len(target))
            for k_index in range(k):
                train_feature,train_target = self.feature.loc[k_index != k_fold_mask,:].to_numpy(),self.target[k_index != k_fold_mask]
                test_feature,test_target = self.feature.loc[k_index == k_fold_mask,:].to_numpy(),self.target[k_index == k_fold_mask]
                #model = self.model(**parameter).fit(train_feature,train_target)
                mc = methodcaller(self.fit_method,train_feature,train_target)
                mp = methodcaller(self.predict_method,test_feature)
                model = mc(self.model(**parameter))
                
                predict_value[k_index == k_fold_mask] = mp(model)
                actual_value[k_index == k_fold_mask] = test_target
                
        if hasattr(self,"total_history"): 
            k_fold_result = pd.concat((k_fold_feature,pd.DataFrame({"predict_value":predict_value,"actual_value":actual_value})),axis = 1)
            for param_value in parameter:
                k_fold_result[param_value] = parameter[param_value]
            if isinstance(self.total_history,str):
                self.total_history = k_fold_result
            else:
                self.total_history = pd.concat((self.total_history,k_fold_result))
        return pd.concat((k_fold_feature,pd.DataFrame({"predict_value":predict_value,"actual_value":actual_value})),axis = 1)
    
    def evaluation_result(self,df):
        evaluation_dict = {}
        if self.prediction_type == "classification":
            for func in [accuracy_score,f1_score,hinge_loss,recall_score,precision_score]:
                evaluation_dict[func.__name__] = np.round(func(df["actual_value"],df["predict_value"]),4)
            
        if self.prediction_type == "regression":
            for func in [mean_absolute_error,mean_squared_error,r2_score,max_error]:
                evaluation_dict[func.__name__] = np.round(func(df["actual_value"],df["predict_value"]),4)
        
        if self.prediction_type == "multi_classification":
            evaluation_dict["macro_f1_score"] = np.round(f1_score(df["actual_value"],df["predict_value"],average = "macro"),4)
            evaluation_dict["micro_f1_score"] = np.round(f1_score(df["actual_value"],df["predict_value"],average = "micro"),4)
            evaluation_dict["macro_precision_score"] = np.round(precision_score(df["actual_value"],df["predict_value"],average = "macro"),4)
            evaluation_dict["micro_precision_score"] = np.round(precision_score(df["actual_value"],df["predict_value"],average = "micro"),4)
            evaluation_dict["macro_recall_score"] = np.round(recall_score(df["actual_value"],df["predict_value"],average = "macro"),4)
            evaluation_dict["micro_recall_score"] = np.round(recall_score(df["actual_value"],df["predict_value"],average = "micro"),4)
        if self.prediction_type == "classification_proba":
            for func in [roc_auc_score,average_precision_score,brier_score_loss]:
                evaluation_dict[func.__name__] = np.round(func(df["actual_value"],df["predict_value"]),4)
        
            
        
        return evaluation_dict
    
    def GridSearch(self,search_area = "all",time_for_each_param = 1,k = 4):
        if search_area == "all":
            search_area = range(0,len(self.parameter_dict_list),1)
        if isinstance(search_area,tuple):
            search_area = range(search_area[0],search_area[1],1)
        for i in search_area:
            param = self.parameter_dict_list[i]
            for j in range(time_for_each_param):
                k_fold_result = self.k_fold(k,parameter = param)
                result = self.evaluation_result(k_fold_result)
                result.update(param)
                if type(self.search_history) == str:
                    self.search_history = pd.DataFrame(data = result.values(),index = result.keys()).T
                else:
                    self.search_history = pd.concat(((pd.DataFrame(data = result.values(),index = result.keys()).T),self.search_history),axis= 0)
        
        self.search_history = self.search_history.reset_index(drop = True).fillna("None")
        self.evaluation_list = list(self.search_history.columns[:-len(self.selected_parameter)])
        self.param_list = [key for key,value in self.parameter_dict.items() if len(value) > 1]
    
    def RandomSearch(self,search_time = "auto",time_for_each_param = 1,k = 4):
        if search_time == "auto":
            search_time = int(len(self.parameter_dict_list)/10) + 1
        assert search_time <= len(self.parameter_dict_list),f"The Search Time exceed the total Combination: {len(self.parameter_dict_list)}"
        chosen_param = np.random.choice(self.parameter_dict_list,search_time,replace  = False)
        for param in chosen_param:
            for j in range(time_for_each_param):
                k_fold_result = self.k_fold(k,parameter = param)
                result = self.evaluation_result(k_fold_result)
                result.update(param)
                if type(self.search_history) == str:
                    self.search_history = pd.DataFrame(data = result.values(),index = result.keys()).T
                else:
                    self.search_history = pd.concat(((pd.DataFrame(data = result.values(),index = result.keys()).T),self.search_history),axis= 0)
        
        self.search_history = self.search_history.reset_index(drop = True).fillna("None")
        self.evaluation_list = list(self.search_history.columns[:-len(self.selected_parameter)])
        self.param_list = [key for key,value in self.parameter_dict.items() if len(value) > 1]
    
    
    def open_html_report(self,initiation = True):
        # define the id,class of the iframe
        graph_dict = {"paramSurface":"Hyperparameter",
                      "paramhistogram":"Hyperparameter",
                      "paramViolin":"Hyperparameter",
                      "scorrelationMap":"feature",
                      "pcorrelationMap":"feature",
                      "performance_measure":"performance"}
        #method_name,class_name
        
        
        html_param = {"evaluation_list":self.evaluation_list,
                      "param_dict_list":self.parameter_dict_list,
                      "param_list":self.param_list,
                      "graph_dict":graph_dict}
        
        
        if initiation:
            self.GridSearch()
        app = Flask(__name__)
        app.jinja_env.auto_reload = True
        app.config['TEMPLATES_AUTO_RELOAD'] = True
        @app.route('/index')
        def user():
            return render_template('index.html',**html_param)
        @app.route("/graphResult",methods = ["GET","POST"])
        def graph_process():
            json = request.json
            print(json)
            respond_dict = {}
            param_list = json["parameter"] + [json["evaluation"]]
            param_combination = eval(json["param_combination"])
            #fix: eval() is dangerous
            print(type(param_combination))
            for method_name in graph_dict:
                if graph_dict[method_name] == "Hyperparameter":
                    fig = methodcaller(method_name,self.search_history,param_list)(img_generation)
                    print(type(fig))
                    respond_dict[method_name] = fig.to_html(full_html=False)
                elif graph_dict[method_name] == "feature":
                    fig = methodcaller(method_name,self.total_history,len(self.selected_parameter))(img_generation)
                    respond_dict[method_name] = fig.to_html(full_html=False)
                elif graph_dict[method_name] == "performance":
                    fig = methodcaller(method_name,self.total_history,self.prediction_type,param_list,param_combination)(img_generation)
                    respond_dict[method_name] = fig.to_html(full_html=False)
                else:
                    raise Exception("Invalid type from html")
            #fig = img_generation.param_3dsurface(self.search_history, "max_depth", "max_features", json["evaluation"])
            #fig2 = img_generation.histogram_on_param(self.search_history, "max_depth", "max_features", json["evaluation"])
            return respond_dict
        @app.route("/rsCommand",methods = ["GET","POST"])
        def rsearchRspond():
            json = request.json
            
            print("Random Search")
            cur_time = time.time()
            self.RandomSearch()
            print("Take {} ms".format(time.time()-cur_time))
            
            
            return "ok"
        @app.route("/gsCommand",methods = ["GET","POST"])
        def gsearchRspond():
            json = request.json
            print("Grid Search")
            cur_time = time.time()
            self.GridSearch()
            print("Take {} ms".format(time.time()-cur_time))
            return "ok"
        @app.route("/displayCommand",methods = ["GET","POST"])
        def displayTable():
            json = request.json
            if self.prediction_type == "regression":
                return img_generation.visualize_table(self.search_history.sort_values(
                by=self.search_history.columns[0],ascending = True)).to_html()
            else:
                return img_generation.visualize_table(self.search_history.sort_values(
                by=self.search_history.columns[0],ascending = False)).to_html()
        
        app.run()
        
        
    
    
    
        
    
    
    
    
    
    
        
        
        
    
    
    
        
    
    
            
                
                
        
        
        
        
        
     
    
        



    
    
    
    