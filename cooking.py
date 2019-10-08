#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 23:39:47 2019

@author: wenyi
"""

#import json
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#df_train = pd.read_json('./whats-cooking/train.json')
df_train = pd.read_json('./whats-cooking/small.json')

# examine the data
#print(df_train.head())
#print(df_train.info())
#print(df_train['cuisine'].value_counts())
#print(df_train['cuisine'].nunique())


# keep all the unique ingredients
recipes = np.array(df_train['ingredients'].values)
ingred_dict = {}

for i in range(len(recipes)):
    for word in recipes[i][:]:
        try:
            ingred_dict[word] += 1
        except:
            ingred_dict[word] = 1
                    
ingreds = np.array(ingred_dict.keys())

def cuisine_to_int(cuisine):
    result = np.where(df_train['cuisine'].unique() == cuisine)[0][0]
    return result
    

# feature: create an ingredient feature vector (1*6714)
# train_data: create ingredient feature vetors (39774*6714)
    
train_data  = [] # features
train_label = [] # labels

for index, row in df_train.iterrows():
        
    feature = [0] * len(ingred_dict)    # number of ingredients

    for i in row['ingredients']: 
        if i in ingreds:
            result = np.where(ingreds == i)[0][0]
            feature[result] = 1         # get index and change to 1 in the feature vector

    train_data.append(feature) 
    train_label.append(cuisine_to_int(row['cuisine']))

train_data, train_label = np.array(train_data), np.array(train_label)
#print(train_data)
#print(train_label)


# Gaussian naive bayes
accuracy = []

gnb = GaussianNB()

kf = KFold(n_splits=3) # 3-fold 

for train_index, test_index in kf.split(train_data):
#    print("TRAIN:", train_index, "TEST:", test_index)

    gnb.fit( train_data[train_index] , train_label[train_index] )
    pred = gnb.predict( train_data[test_index] )
    acc = accuracy_score( pred , train_label[test_index] )
    
    accuracy.append(acc)
   
print(accuracy)
print(sum(accuracy)/len(accuracy))