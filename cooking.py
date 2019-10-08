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
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

df_train = pd.read_json('./whats-cooking/train.json')

# examine the data
#print(df_train.head())
#print(df_train.info())
#print(df_train['cuisine'].value_counts())
#print(df_train['cuisine'].nunique())
#print(df_train['cuisine'].unique())


# keep all the unique cuisines 
cuisine_dict = {}
c = df_train['cuisine'].unique()
for i in range(len(c)):
    word = c[i]
    try:
        cuisine_dict[word] = i
    except:
        cuisine_dict[word] = i


# two helper functions
def cuisine_to_int(cuisine):
#    result = np.where(df_train['cuisine'].unique() == cuisine)[0][0]
    result = cuisine_dict.get(cuisine)
    return result


def int_to_cuisine(i):
    result = cuisine_dict.keys()[cuisine_dict.values().index(i)]
    return result


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

# feature: create an ingredient feature vector (1*6714)
# train_data: create ingredient feature vetors (39774*6714)
    
train_data  = [] # features
train_label = [] # labels

for index, row in df_train.iterrows():
        
    feature = [0] * len(ingred_dict)    # number of ingredients

    for i in row['ingredients']: 
        if i in ingreds:
            result = np.where(ingreds == i)[0][0]
            feature[result] = 1         # get index, change to 1 in feature vector

    train_data.append(feature) 
    train_label.append(cuisine_to_int(row['cuisine']))

train_data, train_label = np.array(train_data), np.array(train_label)



# ******** Cross-validation on train set ********

gnb = GaussianNB()
bnb = BernoulliNB()
logreg = LogisticRegression()

kf = KFold(n_splits=3) # 3-fold 

## Gaussian Naive Bayes
#accuracy = []
#for train_index, test_index in kf.split(train_data):
#
#    gnb.fit( train_data[train_index] , train_label[train_index] )
#    pred = gnb.predict( train_data[test_index] )
#    acc = accuracy_score( pred , train_label[test_index] )
#    
#    accuracy.append(acc)
#   
#print("Gaussian Naive Bayes")
#print(accuracy)
#print(sum(accuracy)/len(accuracy))
#print("")
#
#
## Bernoulli Naive Bayes
#accuracy = []
#for train_index, test_index in kf.split(train_data):
#
#    bnb.fit( train_data[train_index] , train_label[train_index] )
#    pred = bnb.predict( train_data[test_index] )
#    acc = accuracy_score( pred , train_label[test_index] )
#    
#    accuracy.append(acc)
#   
#print("Bernoulli Naive Bayes")   
#print(accuracy)
#print(sum(accuracy)/len(accuracy))
#
#
#
## LogisticRegression
#accuracy = []
#for train_index, test_index in kf.split(train_data):
#    
#    logreg.fit( train_data[train_index] , train_label[train_index] )
#    pred = logreg.predict( train_data[test_index] )
#    acc = accuracy_score( pred , train_label[test_index] )  
#    accuracy.append(acc)
#   
#print("Logistic Regression")   
#print(accuracy)
#print(sum(accuracy)/len(accuracy))


# ******** Run on test set ********
df_test = pd.read_json('./whats-cooking/test.json')
 
test_data  = [] # features
test_label = [] # labels

for index, row in df_test.iterrows():

    feature = [0] * len(ingred_dict)    

    for i in row['ingredients']: 
        if i in ingreds:
            result = np.where(ingreds == i)[0][0]
            feature[result] = 1         

    test_data.append(feature) 

test_data = np.array(test_data)

logreg.fit(train_data, train_label)
pred = logreg.predict(test_data)    # pred generates int, need to convert back to str ('chinese', etc.)

pred_str = []
for i in pred:
    pred_str.append(int_to_cuisine(i))
pred_str = np.array(pred_str)

# output to csv
out = pd.DataFrame(columns=['id', 'cuisine'])
out['id'] = df_test['id']
out['cuisine'] = pred_str
out.to_csv('output.csv', index=False)
