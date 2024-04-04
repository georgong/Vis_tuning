#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:58:57 2024

@author: gongzhenghao
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.io as pio
from pandas.api.types import is_string_dtype,is_numeric_dtype,is_object_dtype
from sklearn.metrics import confusion_matrix,precision_recall_curve,roc_curve
pio.renderers.default = "browser"

"""
this py.file provide a series of function for generate the 

hyperparameter relavant method
parameter: search_history param_list(changed param num)
all of the function should expect to get two types of parameters
1. df, [param1,evaluation]
2. df, [param1,param2,evaluation]
diffferent parameters style change the output


performance relavant method 
parameter: total_history,prediction_type,param_dict
return fig
"""

def visualize_table(df):
    column_list = list(df.columns)
    df = df.T
    print(column_list)
    fig = go.Figure(data=[go.Table(header=dict(values = column_list,
                                               align=['left','center'],
                                               font=dict(color='white', size=12),
                                               #height=40
                                               ),
                 cells=dict(values=df.to_numpy(),
                            align=['left', 'center'],
                            font_size=12,
                            #height=30
                            ))
                     ])
    fig.show()
    return fig



def paramSurface(param_df,param_list):
    """
    
    """
    param_df = param_df.fillna("None")
    if len(param_list) == 3:
        param1,param2,evaluation = param_list
        new_df = param_df.pivot_table(index = param1,columns = param2,values = evaluation)
        y_value = [str(i) for i in new_df.index.tolist()]
        x_value = [str(i) for i in new_df.columns.tolist()]
        z = new_df.to_numpy()
        plate_format = param1 + ": %{y}" + "<br>" + param2 + ": %{x}" + "<br>" + evaluation + ": %{z}"
    
        
        
        fig = go.Figure(data=[go.Surface(z=z,y = y_value,x= x_value,
                                        hovertemplate = plate_format
                                         )])
        fig.update_layout(title=f'{evaluation.capitalize()} 3D Surface with {param1.capitalize()} and {param2.capitalize()}',
                        autosize = True,
                        )
        fig.update_yaxes(type='category')
        fig.update_scenes(xaxis_title_text=f'{param2.capitalize()}',  
                      yaxis_title_text=f'{param1.capitalize()}',  
                      zaxis_title_text=f'{evaluation.capitalize()}')
        return fig
    else:
        param1,evaluation = param_list
        fig = px.density_heatmap(param_df,x = param1,y = evaluation,
                         marginal_y="histogram")
        return fig
        
        


def paramhistogram(param_df,param_list):
    if len(param_list) == 3:
        param1,param2,evaluation = param_list
        param_df  = param_df.fillna("param:None")
        new_df = param_df.melt(id_vars = [param1,param2],value_vars = [evaluation])
        fig = px.histogram(new_df,x = "value",histnorm='probability',facet_row=param1,facet_col = param2)
        return fig
    else:
        param1,evaluation = param_list
        param_df  = param_df.fillna("param:None")
        new_df = param_df.melt(id_vars = [param1],value_vars = [evaluation])
        fig = px.histogram(new_df,x = "value",histnorm='probability',color = param1)
        fig.update_layout(barmode='overlay')
        fig.update_traces(opacity=0.75)
        return fig
    
    
def paramViolin(param_df,param_list):
    if len(param_list) == 3:
        param1,param2,evaluation = param_list
        fig = go.Figure()
        param_iter1 = param_df[param1].unique()
        param_iter2 = param_df[param2].unique()
        
        for param_value1 in param_iter1:
            for param_value2 in param_iter2:
                fig.add_trace(go.Violin(x=param_df[param2][(param_df[param2] == param_value2)&(param_df[param1] == param_value1)],
                        y=param_df[evaluation][(param_df[param2] == param_value2)&(param_df[param1] == param_value1)],
                        legendgroup=param_value1,scalegroup = param_value1,name="{}:{}".format(param2,param_value1),
                        )
             )
        return fig
            
    else:
        param1,evaluation = param_list
        fig = go.Figure()
        param_iter = param_df[param1].unique()
        for param_value in param_iter:
            fig.add_trace(go.Violin(x=param_df[param1][param_df[param1] == param_value],
                            y=param_df[evaluation][param_df[param1] == param_value],
                            name= "{}:{}".format(param1,param_value),
                            box_visible=True,
                            meanline_visible=True))
            

        return fig
    
def pcorrelationMap(total_df,feature_split_index):
    fig = px.imshow(total_df.drop("predict_value",axis = 1).iloc[:,:-feature_split_index].corr(),
                    color_continuous_scale=px.colors.sequential.Viridis)
    
    fig.update_layout(title=f'Pearson Correlation Coefficient',
                    autosize = True,
                    )
    return fig


def scorrelationMap(total_df,feature_split_index):
    
    fig = px.imshow(total_df.drop("predict_value",axis = 1).iloc[:,:-feature_split_index].corr("spearman"),
                    color_continuous_scale=px.colors.sequential.Viridis)
    
    fig.update_layout(title=f'Spearman Correlation Coefficient',
                    autosize = True,
                    )
    return fig


def performance_measure(total_df,prediction_type,param_list,param_combination):
    target_df = total_df.loc[total_df[list(param_combination.keys())].isin(list(param_combination.values())).all(axis=1), :]
    if prediction_type == "classification":
        cm = confusion_matrix(target_df["actual_value"],target_df["predict_value"])
        fig = px.imshow(cm,x = sorted(target_df["predict_value"].unique()),y = sorted(target_df["actual_value"].unique()),
                        labels=dict(x="Predict_Value", y="Actual_Value"))
        return fig
    elif prediction_type == "multi_classification":
        cm = confusion_matrix(target_df["actual_value"],target_df["predict_value"])
        fig = px.imshow(cm,x = sorted(target_df["predict_value"].unique()),y = sorted(target_df["actual_value"].unique()))
        return fig
    elif prediction_type == "regression":
        x = np.arange(len(target_df))
        fig = go.Figure([go.Scatter(x=x, y=target_df["actual_value"], 
                   name='actual_value', mode='markers'),
                         go.Scatter(x=x, y=target_df["predict_value"], 
                   name='prediction')
                         ])
        return fig
    elif prediction_type == "classification_proba":
        p,r,t = precision_recall_curve(target_df["actual_value"],target_df["predict_value"])
        fig = go.Figure([go.Scatter(x=r, y=p)
                         ])
        fig.update_layout(title=f'Precision-Recall Curve',
                        autosize = True,
                        )
        
        return fig
        
        
    
    

    






    


    
    






    