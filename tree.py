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
import copy


pd.options.mode.chained_assignment = None 

#handling command line arguments to be entered in the terminal
training_percentage=float(sys.argv[2])
random_seed= int(sys.argv[3])
is_numeric=str(sys.argv[4])
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

attribute_uniquevalues={}
for attribute in attributes:
    attribute_uniquevalues[attribute]=training_df[attribute].unique()

class node:
    def __init__(self, val): 
        self.val = val
        self.children = {}
        self.threshold = None

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

def threshold(S, a):
    S = S.sort_values(a)
    # print(S)
    label_index = 0
    attribute_index = columnnames.index(a)
    thresholds = []
    previous = row=S.iloc[0].to_numpy()
    for i in range(1,len(S)):
        row=S.iloc[i].to_numpy()
        if previous[label_index] != row[label_index]:
            # print("previous is: ", previous[label_index], previous[attribute_index])
            # print("row is: ", row[label_index], row[attribute_index])
            new_threshold = (previous[attribute_index]+row[attribute_index])/2
            thresholds.append(new_threshold)
            previous = row
    return thresholds

def best_threshold(S, a, thresholds):
    best_gain = 0 
    best_thresh = 0
    for threshold in thresholds:
        gain = numeric_gain(S, a, threshold)
        if gain > best_gain:
            best_gain = gain
            best_thresh = threshold
    return best_thresh

def numeric_entropy(S, a, threshold):
    attribute_index = columnnames.index(a)
    less = {}
    greater ={}
    label_index = 0
    less_total = 0
    greater_total = 0
    for i in range(0,len(S)):
        row=S.iloc[i].to_numpy()
        if row[attribute_index] <= threshold:
            if row[label_index] not in less:
                less[row[label_index]]=1
            else:
                less[row[label_index]]+=1
            less_total+=1
        else:
            if row[label_index] not in greater:
                greater[row[label_index]]=1
            else:
                greater[row[label_index]]+=1
            greater_total+=1
    entropy_less = 0
    entropy_greater = 0
    for values in less.values():
        entropy_less = entropy_less + ((values/less_total) * math.log2(values/less_total))
    entropy_less = -1*entropy_less
    for values in greater.values():
        entropy_greater = entropy_greater + ((values/greater_total) * math.log2(values/greater_total))
    entropy_greater = -1*entropy_greater
    return entropy_less, entropy_greater, less_total, greater_total

def numeric_gain(S, a, threshold):
    entropy_less, entropy_greater, less_total, greater_total = numeric_entropy(S, a, threshold)
    length = len(S)
    gain = 1 - ((less_total/length) * entropy_less) + ((greater_total/length) * entropy_greater) 
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
    # print(attribute)
    # print(new_df)
    return new_df

def new_df_numeric(attribute, subset, threshold):
    less_df = subset[subset[attribute] <= threshold]
    greater_df = subset[subset[attribute] > threshold]
    return less_df, greater_df

def ID3(attributes, subset):

    if is_numeric=="True":
        print("this is true")
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
            list_thresholds = threshold(subset, best)
            threshold_best = best_threshold(subset, best, list_thresholds)
            attribute_index = columnnames.index(best)
            N = node(best)
            N.threshold = threshold_best
            # unique = attribute_uniquevalues[best]
            less_df, greater_df = new_df_numeric(best, subset, threshold_best)
            if len(less_df)==0:
                N.children['less'] = node(max_label)
            else:
                N.children['less']=ID3(attributes, less_df)
            if len(greater_df)==0:
                N.children['greater'] = node(max_label)
            else:
                N.children['greater']=ID3(attributes, greater_df)
        return N

    if is_numeric=="False":
        print('This is false')
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
            unique = attribute_uniquevalues[best]
            pass_attribute=copy.deepcopy(attributes)
            pass_attribute.remove(best)
            for value in unique:
                
                new_df = possible_values(value, subset, best)
                if len(new_df)==0:
                    N.children[value] = node(max_label)
                else:
                    N.children[value]=ID3(pass_attribute, new_df)
        return N

def prediction(test_df,tree,columnnames):
    accuracy=0
    numerator=0
    length_testdf=len(test_df)
    for i in range(0,length_testdf):
        row=test_df.iloc[i].to_numpy()
        attribute_value={}
        for x in range(len(columnnames)):
            attribute_value[columnnames[x]]=row[x]
        predicted_label=predict_label(attribute_value,tree)
        if predicted_label==attribute_value["label"]:
            numerator+=1
    accuracy=numerator/length_testdf
    print("The accuracy is",accuracy)

def predict_label(attribute_value,tree):
    if len(tree.children)==0:
        return tree.val
    else: 
        return predict_label(attribute_value,tree.children[attribute_value[tree.val]])

def numeric_prediction(test_df,tree,columnnames):
    accuracy=0
    numerator=0
    length_testdf=len(test_df)
    for i in range(0,length_testdf):
        row=test_df.iloc[i].to_numpy()
        attribute_value={}
        for x in range(len(columnnames)):
            attribute_value[columnnames[x]]=row[x]
        predicted_label=numeric_predict_label(attribute_value,tree)
        # print("this is attribute value: ", attribute_value)
        if predicted_label==row[0]:
            numerator+=1
    accuracy=numerator/length_testdf
    print("The accuracy is",accuracy)

def numeric_predict_label(attribute_value,tree):
    if len(tree.children)==0:
        return tree.val
    else: 
        if attribute_value[tree.val] <= tree.threshold:
            return numeric_predict_label(attribute_value,tree.children['less'])
        else:
            return numeric_predict_label(attribute_value,tree.children['greater'])

def printTree(tree:node, level=0,child=""):
    
    print("        " * level,child,tree.val)
    for child in tree.children.keys():
        printTree(tree.children[child], level + 1,child)


tree = ID3(attributes, training_df)
printTree(tree)
# print(tree)
# print(tree.val)
# print(tree.threshold)
# print(tree.children)
numeric_prediction(test_df,tree,columnnames)
# prediction(test_df,tree,columnnames)

# thresholds = threshold(training_df, "bill_length_mm")
# print(thresholds)



