#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 10:19:17 2020

@author: csg
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")

# License: BSD 3-Clause

# how to set an apikey
#https://openml.github.io/openml-python/master/examples/20_basic/introduction_tutorial.html#sphx-glr-examples-20-basic-introduction-tutorial-py

# how to load datasets
#https://openml.github.io/openml-python/master/examples/20_basic/simple_datasets_tutorial.html#sphx-glr-examples-20-basic-simple-datasets-tutorial-py


import openml
CSG = "4795f6a3f9aac7cfbe4d1d8b66d438a4"
openml.config.apikey = CSG

dic_task_fin = {
    	'Australian': '146818', 
    	'adult': '7592',  
     	'bank-marketing': '14965',  
      	'credit-g': '31', 
     	'numerai28.6': '167120', 
        }

dic_did_fin = {
        'Australian': 40981,
        'adult': 1590,
        'bank-marketing': 1558,
        'credit-g': 31,
        'numerai28.6': 23517
        }


###############################################################################

def get_openml_all():
    #datasets_df = openml.datasets.list_datasets(output_format='dataframe')
    #print(datasets_df.head(n=10))
    dic_datasets = openml.datasets.list_datasets()
    
    dic_all= {}
    for key in dic_datasets.keys():
        value = dic_datasets[key]
        name = value['name']
        did = value['did']
        if name in dic_did_fin.keys():
            dic_info = get_openml_info_by_did(did, True)
            dic_all[name] = dic_info # key:value -> name:info

    return dic_all

def get_openml_info_by_did(did, is_save=False):   
    dic_info = {}
    dic_info['did'] = did
    
    #did = 31
    dataset = openml.datasets.get_dataset(did)
    dic_info['name'] = dataset.name
    dic_info['target'] = dataset.default_target_attribute
    print("name:", dic_info['name'])
    print("target:", dic_info['target'])
    
    features = dataset.features
    list_features = [features[idx].name for idx in features]
    dic_info['features'] = list_features
    print("target:", dic_info['features'])
        
    data = dataset.get_data()
    df_data = pd.DataFrame(data, columns=list_features)
    dic_info['df_clean'] = df_data

    if is_save:
        path = "./" +  dic_info['name'] + "_prepro.csv"
        df_data.to_csv(path, index=False)
    
    return dic_info

def test():
    dic_fin = get_openml_all()
   
    for key in dic_fin.keys():
        df = dic_fin[key]['df_clean']
        target = dic_fin[key]['target']
        plot = sns.pairplot(df, hue=target)
        plot.map_upper(hide_current_axis)
        plt.show()
        
        
###############################################################################

# Iris dataset https://www.openml.org/d/61
dataset = openml.datasets.get_dataset(61)

# Print a summary
print("This is dataset '{dataset.name}', the target feature is "
      "'{dataset.default_target_attribute}'")
print("URL: {dataset.url}")
print(dataset.description[:500])

# X - An array/dataframe where each row represents one example with
# the corresponding feature values.
# y - the classes for each example
# categorical_indicator - an array that indicates which feature is categorical
# attribute_names - the names of the features for the examples (X) and
# target feature (y)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format='dataframe',
    target=dataset.default_target_attribute
)

###############################################################################

def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)


# We combine all the data so that we can map the different
# examples to different colors according to the classes.
combined_data = pd.concat([X, y], axis=1)
iris_plot = sns.pairplot(combined_data, hue="class")
iris_plot.map_upper(hide_current_axis)
plt.show()
