import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.signal import savgol_filter

#opens x values for any plot, loads them as scan_values
file = "xvals.csv"
scan_values = np.genfromtxt(file,skip_header=0,delimiter=",")

#opens the pickle containing the dictionary of data, loads it as d
with open('saved_dictionary.pkl', 'rb') as f:
    d = pickle.load(f)


#this function creates a copy of the dictionary d, and then normalises the amount of data in each key to the higher amount of data in any key
def normalise_data(d):
    #creates a copy of the dictionary d
    d_copy = d.copy()
    #finds the highest amount of data in any key
    max_data = max(len(d_copy[key]) for key in d_copy)
    #for each key in the dictionary
    for key in d_copy:
        #if the length of the key is less than the highest amount of data in any key
        if len(d_copy[key]) < max_data:
            #find the difference between the highest amount of data and the length of the key
            diff = max_data - len(d_copy[key])
            #find the amount of data to be added to the key
            add = diff//len(d_copy[key])
            #find the remainder of data to be added to the key
            remainder = diff%len(d_copy[key])
            #add the amount of data to be added to the key
            d_copy[key] = d_copy[key] + d_copy[key]*add
            #add the remainder of data to be added to the key
            d_copy[key] = d_copy[key] + d_copy[key][:remainder]
    return d_copy

#this function takes a list of lists as an input, and returns two lists back, with a randomly distributed amount of data from the input list, with the same amount of data in each list being decided by an input between 0 and 1 called ratio
def split_data(data, ratio):
    #finds the length of the input list
    length = len(data)
    #finds the amount of data to be in each list
    split = int(length*ratio)
    #creates an empty list
    data1 = []
    #creates an empty list
    data2 = []
    #for each element in the input list
    for i in range(length):
        #if the length of data1 is less than the amount of data to be in each list
        if len(data1) < split:
            #add the current element to data1
            data1.append(data[i])
        #if the length of data1 is greater than or equal to the amount of data to be in each list
        else:
            #add the current element to data2
            data2.append(data[i])
    #return the two lists
    return data1, data2

#this function splits the data in each key in the dictionary into two lists, with the amount of data in each list being decided by an input between 0 and 1 called ratio
def split_dict(d,ratio):
    train, test ={},{}
    for key,val in d.items():
        np.random.shuffle(val)
        split_by = int(len(val)*ratio)
        train[key],test[key] = val[:split_by],val[split_by:]
    return train,test



#this function prints each dictionary key and the shape of its value
def print_dict(d):
    for key in d:
        print(f"{key} has dimensions: {np.shape(d[key])} for training data")
        print("---------------------------------------------------------------------------------------------")



#this function takes a dictionary, and gaussian normalizes the data in each key
def gaussian_normalization(d):
    for key in d:
        #finds the mean of the data in the key
        mean = np.mean(d[key])
        #finds the standard deviation of the data in the key
        std = np.std(d[key])
        #for each element in the data in the key
        for i in range(len(d[key])):
            #subtract the mean from the element
            d[key][i] = d[key][i] - mean
            #divide the element by the standard deviation
            d[key][i] = d[key][i]/std
        #return the dictionary
        return d

#this function takes a dictionary, and takes a rolling average of each key
def rolling_average(d):
    for key in d:
        #for each element in the data in the key
        for i in range(len(d[key])):
            #if the element is not the first element in the data in the key
            if i != 0:
                #add the element to the element before it
                d[key][i] = d[key][i] + d[key][i-1]
                #divide the element by two
                d[key][i] = d[key][i]/2
        #return the dictionary
        return d

#this function takes a dictionary, and applies the savgol_filter to all the lists in the data of each key
def savgol_filter_dict(d):
    #hyperparamters for savgol filter, these are fudge factors
    window_length = 100 #how far around the point are you taking into consideration
    polyorder = 3 #what size polynomial you are fitting

    for key in d:
        #for each element in the data in the key
        for i in range(len(d[key])):
            #apply the savgol_filter to the element
            d[key][i] = savgol_filter(d[key][i], window_length, polyorder)
    #return the dictionary
    return d

#this function takes a dictionary, and returns two lists. The first list is the values of the dictionary, the second list is the key of each element of the first list
def dict_to_list(d):
    #create an empty list
    data = []
    #create an empty list
    labels = []
    #for each key in the dictionary
    for key in d:
        #for each element in the value of the key
        for i in range(len(d[key])):
            #add the element to the data list
            data.append(d[key][i])
            #add the key to the labels list
            labels.append(key)
    #return the data list and the labels list
    return data, labels

#this function takes a dictionary, and a ratio. it first splits the dictionary into two dictionaries using split_dict, returning a training a testing set with amounts determined by ratio
# it then ensures that the training and testing sets have the same amount of data in each key by using normalise_data.
#  it then applies gaussian_normalization to the training and testing sets
#   it finally converts the training and testing sets into lists using dict_to_list, returning the training and testing sets as lists
#    method specifies what data processing will happen to the data
def process(d, ratio, method):
    #split the dictionary into two dictionaries, with the ratio being the input ratio
    d_train, d_test = split_dict(d, ratio)
    #normalizes the data using normalise_data
    d_train = normalise_data(d_train)
    d_test  = normalise_data(d_test)
    #Method 1
    #normalizes the data inside the dictionary with guassian normalization
    if method == 1:
        print("Applying Gaussian Normalization")
        d_train = gaussian_normalization(d_train)
        d_test = gaussian_normalization(d_test)
    #Method 2
    #normalizes the data inside the dictionary with rolling average
    elif method == 2:
        print("Applying Rolling Average")
        d_train = rolling_average(d_train)
        d_test = rolling_average(d_test)
    #Method 3
    #normalizes the data inside the dictionary with savgol_filter
    elif method == 3:
        print("Applying Savgol Filter")
        d_train = savgol_filter_dict(d_train)
        d_test = savgol_filter_dict(d_test)
    #converts the dictionary into two lists, one of the data, and one of the labels
    train_data, train_labels = dict_to_list(d_train)
    test_data, test_labels = dict_to_list(d_test)
    #return the four lists
    return train_data, train_labels, test_data, test_labels

#this function creates a run a decision tree based on a list of training and testing data and a list of training and testing labels
def decision_tree(train_data, train_labels, test_data, test_labels):
    #create a decision tree classifier
    clf = tree.DecisionTreeClassifier()
    #fit the decision tree classifier to the training data and labels
    clf.fit(train_data, train_labels)
    #print the accuracy of the decision tree classifier on the testing data and labels
    print("Accuracy of decision tree classifier on test set: {:.2f}".format(clf.score(test_data, test_labels)))

#this function runs a k nearest neighbor alorithm from sklearn
def knn(train_data, train_labels, test_data, test_labels):
    #create a k nearest neighbor classifier
    knn = KNeighborsClassifier(n_neighbors=1)
    #fit the k nearest neighbor classifier to the training data and labels
    knn.fit(train_data, train_labels)
    #print the accuracy of the k nearest neighbor classifier on the testing data and labels
    print("Accuracy of k nearest neighbor classifier on test set: {:.8f}".format(knn.score(test_data, test_labels)))

#this function uses sklearn MLPclassifer and runs it on inputed training and testing data and labels
def mlp(train_data, train_labels, test_data, test_labels):
    #makes training and testing labels into numpy arrays
    train_labels = np.array(train_labels)
    #create a MLP classifier
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    #fit the MLP classifier to the training data and labels
    mlp.fit(train_data, train_labels)
    #print the accuracy of the MLP classifier on the testing data and labels
    print("Accuracy of MLP classifier on test set: {:.2f}".format(mlp.score(test_data, test_labels)))


#this function plots 4 subplots of the first of data of the first key in the dictionary, one of the subplots is raw, the second gaussian normalized, the third rolling average, and the fourth is the difference between the rolling average and the raw data
def plot_data(d):
    #create a figure with 4 subplots
    fig, axs = plt.subplots(4)
    #plot the first list in the data with the key "polyethylene" in the dictionary
    axs[0].plot(d["polyethylene"][0], label = "Raw Data")
    #plot the first list in the data with the key "polyethylene" in the dictionary, but gaussian normalized, label it
    axs[1].plot(gaussian_normalization(d)["polyethylene"][5], label = "Gaussian Normalized")
    #plot the first list in the data with the key "polyethylene" in the dictionary, but rolling average, label it
    axs[2].plot(rolling_average(d)["polyethylene"][5], label = "Rolling Average")
    #plot the first list in then data with the key "polyethylene" in the dictionary, but apply the function savitzky_golay to it, label it
    axs[3].plot(savgol_filter(d["polyethylene"][5], 120, 3), label = "Savitzky-Golay")
    #plot the figure,show labels on all subplots
    plt.legend()
    plt.show()

    

plot_data(d)
a,b,c,d = process(d, 0.8,2)
decision_tree(a,b,c,d)
knn(a,b,c,d)
mlp(a,b,c,d)