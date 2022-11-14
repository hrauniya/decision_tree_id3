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
import time

start=time.time()
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

#dictionary maps attribute to its unique values
attribute_uniquevalues={}
for attribute in attributes:
    attribute_uniquevalues[attribute]= dataframe[attribute].unique().tolist()

#node class for building decision tree
class node:
    def __init__(self, val): 
        self.val = val
        self.children = {}
        self.threshold = None

#calculate entropy of a dataframe
def calculate_entropy(dataframe):
    entropy_count = {}
    total=0
    for row in dataframe.values:
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

#calculating entropy given a certain attribute in a dataframe
def calculate_attribute_entropy(dataframe, attribute, index_attribute):
    entropy_count={}
    total=0
    for row in dataframe.values:
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

#calculating the gain of attribute in dataframe S
def gain(S, a):
    gain_attribute = {}
    attribute_index = columnnames.index(a)
    length = len(S)
    for row in S.values:
        if row[attribute_index] not in gain_attribute:
            gain_attribute[row[attribute_index]]=1
        else:
            gain_attribute[row[attribute_index]]+=1
    second_exp = 0
    for attribute_value in gain_attribute.keys():
        second_exp = second_exp + ((gain_attribute[attribute_value])/length) *  calculate_attribute_entropy(S, attribute_value, attribute_index)
    gain = calculate_entropy(S) - second_exp
    return gain

#finding the thresholds of an attribute in the subset S
def threshold(S, a):
    S = S.sort_values(a)
    label_index = 0
    attribute_index = columnnames.index(a)
    thresholds = []
    previous=S.iloc[0].to_numpy()
    for row in S.values[1:]:
        if previous[label_index] != row[label_index]:
            new_threshold = (previous[attribute_index]+row[attribute_index])/2
            if new_threshold not in thresholds:
                thresholds.append(new_threshold)
        previous = row
    return thresholds

#finding the best_threshold of an attribute
def best_threshold(S, a, thresholds):
    best_gain = 0 
    best_thresh = 0
    for threshold in thresholds:
        gain = numeric_gain(S, a, threshold)
        if gain > best_gain:
            best_gain = gain
            best_thresh = threshold
    return best_thresh,best_gain

#finding the numeric entropy of the subsets divided by the threshold
def numeric_entropy(S, a, threshold):
    attribute_index = columnnames.index(a)
    less = {}
    greater ={}
    label_index = 0
    less_total = 0
    greater_total = 0
    for row in S.values:
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

#calculating the gain of the attribute given a threshold and a subset S
def numeric_gain(S, a, threshold):
    entropy_less, entropy_greater, less_total, greater_total = numeric_entropy(S, a, threshold)
    length = len(S)
    second_exp=((less_total/length) * entropy_less) + ((greater_total/length) * entropy_greater)
    gain = calculate_entropy(S) - second_exp
    return gain

#find the best attribite for categorical attributes
def best_attribute(attributes, S):
    best_dict = {}
    best_gain = 0
    best_attribute = ""
    for attribute in attributes:
        attribute_gain = gain(S, attribute)
        best_dict[attribute] = attribute_gain
        best_attribute = max(best_dict, key=best_dict.get)
    return best_attribute

#finding the best attribute for continuos attributes
def best_attribute_numeric(attributes, S):
    best_dict = {}
    attribute_threshold={}
    result_threshold=0
    best_attribute = ""
    for attribute in attributes:
        
        list_thresholds = threshold(S, attribute)
        threshold_best,best_gain = best_threshold(S, attribute, list_thresholds)
        best_dict[attribute]=best_gain
        attribute_threshold[attribute]=threshold_best
    
    best_attribute=max(best_dict,key=best_dict.get)
       
    return best_attribute,attribute_threshold[best_attribute]

def possible_values(attribute, subset, index):
    new_df = subset.loc[subset[index]==attribute]
    return new_df

#function create two dataframes according to threshold
def new_df_numeric(attribute, subset, threshold):
    less_df = subset[subset[attribute] <= threshold]
    greater_df = subset[subset[attribute] > threshold]
    return less_df, greater_df

#the id3 algorithm which also handles continuous values
def ID3(attributes, subset):

    if is_numeric=="True":
        label_count = {}
        for row in subset.values:
            if row[0] not in label_count:
                label_count[row[0]]=1
            else:
                label_count[row[0]]+=1
        max_label = max(label_count, key=label_count.get)
        if len(attributes)==0:
            N = node(max_label)
        elif len(label_count)==1:
            N = node(max_label)
        else:
            best,threshold_best = best_attribute_numeric(attributes, subset)
            attribute_index = columnnames.index(best)
            N = node(best)
            N.threshold = threshold_best
            less_df, greater_df = new_df_numeric(best, subset, threshold_best)
            pass_attribute=copy.deepcopy(attributes)
            if len(less_df)==0:
                N.children['less'] = node(max_label)
            else:
                N.children['less']=ID3(pass_attribute, less_df)
            if len(greater_df)==0:
                N.children['greater'] = node(max_label)
            else:
                N.children['greater']=ID3(pass_attribute, greater_df)
        return N

    if is_numeric=="False":
        label_count = {}
        for row in subset.values:
            if row[0] not in label_count:
                label_count[row[0]]=1
            else:
                label_count[row[0]]+=1
        
        max_label = max(label_count, key=label_count.get)
        if len(attributes)==0:
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

#function to predict labels and calculate accuracy for test set for is_numeric=false
def prediction(test_df,tree,columnnames,place_dict,twolist):
    accuracy=0
    numerator=0
    length_testdf=len(test_df)
    for row in test_df.values:
        attribute_value={}
        for x in range(len(columnnames)):
            attribute_value[columnnames[x]]=row[x]
        predicted_label=predict_label(attribute_value,tree)
        if predicted_label==attribute_value[columnnames[0]]:
            numerator+=1
    column = place_dict[predicted_label]
    row = place_dict[row[0]]
    twolist[row][column]+=1
    accuracy=numerator/length_testdf
    print("The accuracy is",accuracy)

##function to traverse the tree predicting label for is_numeric=false
def predict_label(attribute_value,tree):
    if len(tree.children)==0:
        return tree.val
    else: 
        return predict_label(attribute_value,tree.children[attribute_value[tree.val]])

#function to predict labels and calculate accuracy for test set for is_numeric=true
def numeric_prediction(test_df,tree,columnnames, place_dict, twolist):
    accuracy=0
    numerator=0
    length_testdf=len(test_df)
    for row in test_df.values:
        attribute_value={}
        for x in range(len(columnnames)):
            attribute_value[columnnames[x]]=row[x]
        predicted_label=numeric_predict_label(attribute_value,tree)
        if predicted_label==row[0]:
            numerator+=1
        column = place_dict[predicted_label]
        row = place_dict[row[0]]
        twolist[row][column]+=1
    accuracy=numerator/length_testdf
    print("The accuracy is",accuracy)

#function to traverse the tree predicting label for is_numeric=true
def numeric_predict_label(attribute_value,tree):
    if tree.threshold is None:
        return tree.val
    else: 
        if attribute_value[tree.val] <= tree.threshold:
            return numeric_predict_label(attribute_value,tree.children['less'])
        else:
            return numeric_predict_label(attribute_value,tree.children['greater'])

#function to print the tree
def printTree(tree:node, level=0,child=""):
    
    print("        " * level,child,tree.val)
    for child in tree.children.keys():
        printTree(tree.children[child], level + 1,child)

#function to create variables for confusion matrix       
def confusion_matrix():
    all_unique = dataframe[dataframe.columns[0]].unique()
    twolist = []
    place_dict = {}

    for x in range(len(all_unique)):
        newlist=[]
        for y in range(len(all_unique)):
            newlist.append(0)
        twolist.append(newlist)

    for x in range(len(all_unique)):
        place_dict[all_unique[x]] = x

    return place_dict, twolist, all_unique


#main function
def main():
    place_dict, twolist, all_unique = confusion_matrix()
    if is_numeric == "True":
        tree = ID3(attributes, training_df)
        numeric_prediction(test_df,tree,columnnames,place_dict,twolist)
    else: 
        tree = ID3(attributes, training_df)
        prediction(test_df,tree,columnnames,place_dict,twolist)

    # create file name
    length = len(dataset)
    abrev = dataset[0:length-4]
    name = "results-tree" + abrev + "-" + is_numeric+ "-" + str(random_seed) + ".csv"

    final_labels = all_unique
    final_labels = final_labels.tolist()
    final_labels.append("")

    # create new csv file
    with open(name, 'w', newline='') as newfile:
    # initialize csv
        write = csv.writer(newfile)
        write.writerow(final_labels)
        count=0
        # write each row to csv
        for row in twolist:
            row.append(all_unique[count])
            write.writerow(row)
            count+=1       


main()

end=time.time()
print(end-start)




