"""
@author:Harsha Rauniyar and Austin Alcancia
implementing the id3 algorithm
"""

from enum import unique
import sys
import random
import math
import numpy as np
import pandas as pd
import csv
import pptree 

pd.options.mode.chained_assignment = None 

#handling command line arguments to be entered in the terminal
training_percentage=float(sys.argv[2])
random_seed= int(sys.argv[3])
is_numeric=bool(sys.argv[4])
dataset=sys.argv[1]

random.seed(random_seed)

#initialize dataframe
dataframe=pd.read_csv(sys.argv[1])

dataframe=dataframe.sample(random_state=random_seed, frac=1)


#spliting the dataset to training, and test depending on the percentage given by the user
dataframe_length = len(dataframe.index)
training_set_length=round(training_percentage*dataframe_length)

training_df = dataframe.iloc[0:training_set_length]
test_df = dataframe.iloc[training_set_length:]

columnnames = list(training_df.columns.values)
attributes = columnnames[1:]

class node:
    def __init__(self, val): 
        self.val = val
        self.children = {}

#calculate entropy of a dataframe
def calculate_entropy(dataframe):
    entropy_count = {}
    total=0
    for i in range(0,len(dataframe)):
        row=dataframe.iloc[i].to_numpy()
        if row[0] not in entropy_count:
            entropy_count[row[0]]=1
        else:
            entropy_count[row[0]]+=1
        total+=1
    entropy = 0
    for values in entropy_count.values():
        entropy = entropy + ((values/total) * math.log2(values/total))
    entropy = -1*entropy
    return entropy

#calculating entropy given a certain attribute
def calculate_attribute_entropy(dataframe, attribute, index_attribute):
    entropy_count={}
    total=0
    for i in range(0,len(dataframe)):
        row=dataframe.iloc[i].to_numpy()
        if row[index_attribute]==attribute:
            if row[0] not in entropy_count:
                entropy_count[row[0]]=1
            else:
                entropy_count[row[0]]+=1
            total+=1
    entropy = 0
    for values in entropy_count.values():
        entropy = entropy + ((values/total) * math.log2(values/total))
    entropy = -1*entropy
    return entropy

def gain(S, a):
    # print("this is gain")
    # print(S)
    # print('this is attribute')
    # print(a)
    gain_attribute = {}
    attribute_index = columnnames.index(a)
    length = len(S)
    for i in range(0,len(S)):
        row=S.iloc[i].to_numpy()
        if row[attribute_index] not in gain_attribute:
            gain_attribute[row[attribute_index]]=1
        else:
            gain_attribute[row[attribute_index]]+=1
    second_exp = 0
    for attribute_value in gain_attribute.keys():
        second_exp = second_exp + ((gain_attribute[attribute_value])/length) *  calculate_attribute_entropy(S, attribute_value, attribute_index)
    gain = calculate_entropy(S) - second_exp
    return gain

def numeric_gain(S, a):
    # print("this is gain")
    # print(S)
    # print('this is attribute')
    # print(a)
    gain_attribute = {}
    attribute_index = columnnames.index(a)
    length = len(S)
    S.sort_values(by=[a])
    for i in range(0,len(S)):
        row=S.iloc[i].to_numpy()
        if row[attribute_index] not in gain_attribute:
            gain_attribute[row[attribute_index]]=1
        else:
            gain_attribute[row[attribute_index]]+=1
    second_exp = 0
    for attribute_value in gain_attribute.keys():
        second_exp = second_exp + ((gain_attribute[attribute_value])/length) *  calculate_attribute_entropy(S, attribute_value, attribute_index)
    gain = calculate_entropy(S) - second_exp
    return gain

def best_attribute(attributes, S):
    best_dict = {}
    best_gain = 0
    best_attribute = ""
    for attribute in attributes:
        attribute_gain = gain(S, attribute)
        best_dict[attribute] = attribute_gain
        best_attribute = max(best_dict, key=best_dict.get)
    # print(best_dict)
    return best_attribute

def best_attribute_numeric(attributes, S):
    best_dict = {}
    best_gain = 0
    best_attribute = ""
    for attribute in attributes:
        attribute_gain = gain(S, attribute)
        best_dict[attribute] = attribute_gain
        best_attribute = max(best_dict, key=best_dict.get)
    # print(best_dict)
    return best_attribute

def possible_values(attribute, subset, index):
    new_df = subset.loc[subset[index]==attribute]
    print(attribute)
    print(new_df)
    return new_df

def ID3(attributes, subset):
    if is_numeric == False:
        label_count = {}
        for i in range(len(subset)):
            row=subset.iloc[i].to_numpy()
            if row[0] not in label_count:
                label_count[row[0]]=1
            else:
                label_count[row[0]]+=1
        max_label = max(label_count, key=label_count.get)

        if len(attributes)==0:
            # print("length is 0")
            N = node(max_label)
        elif len(label_count)==1:
            N = node(max_label)
        else:
            best = best_attribute(attributes, subset)
            attribute_index = columnnames.index(best)
            N = node(best)
            unique = subset[best].unique()
            attributes.remove(best)
            for value in unique:
                new_df = possible_values(value, subset, best)
                if len(new_df)==0:
                    N.children[value] = node(max_label)
                else:
                    N.children[value]=ID3(attributes, new_df)
        return N
        
    if is_numeric == True:
        label_count = {}
        for i in range(len(subset)):
            row=subset.iloc[i].to_numpy()
            if row[0] not in label_count:
                label_count[row[0]]=1
            else:
                label_count[row[0]]+=1
        max_label = max(label_count, key=label_count.get)

        if len(attributes)==0:
            # print("length is 0")
            N = node(max_label)
        elif len(label_count)==1:
            N = node(max_label)
        else:
            best = best_attribute(attributes, subset)
            attribute_index = columnnames.index(best)
            N = node(best)
            unique = subset[best].unique()
            attributes.remove(best)
            for value in unique:
                new_df = possible_values(value, subset, best)
                if len(new_df)==0:
                    N.children[value] = node(max_label)
                else:
                    N.children[value]=ID3(attributes, new_df)
        return N
                
tree = ID3(attributes, training_df)
print("this is sunny val", tree.children['Rain'].children['Strong'].val)