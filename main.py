from PIL import Image
import cv2
import numpy as np
from sklearn.utils import shuffle
from random import seed
from random import randrange
from random import random
from csv import reader
import matplotlib.pyplot as plt 
from math import exp



def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

problist = load_csv("output.csv")

for i in range(len(problist[0])):
    str_column_to_float(problist, i)

a1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
a2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
a3 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        print(predicted)
        actual = [row[-1] for row in fold]
        print(actual)
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            #print("act",activation)
            neuron['output'] = transfer(activation)
            #print("this",neuron['output'])
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    #print("op",n_outputs)
    for epoch in range(n_epoch):
        for row in train:
            outputs = forward_propagate(network, row)
            #print("row",row)
            expected = [0 for i in range(n_outputs)]
            #print("exp",expected)
            expected[row[-1]] = 1
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    print("op",outputs)
    return outputs.index(max(outputs))


# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return (predictions)

def feature(img_name,r_img_name):
    temp = []
    img = Image.open(img_name) 
    pix = img.load()
    x,y = img.size
    r=0
    g=0
    b=0
    for i in range(x):
        for j in range(y):
            r += pix[i,j][0]
            g += pix[i,j][1]
            b += pix[i,j][2]
    col1=0
    col2=0
    col3=0
    ratio=(r+b+g)
    col1 = r/ratio
    col2 = g/ratio
    col3 = b/ratio
    temp.append(col1)
    temp.append(col2)
    temp.append(col3)
    
    m =  cv2.imread(img_name)
    edges = cv2.Canny(m,100,200)
    h,w = np.shape(edges)
    x = []
    y = []
    for py in range(0,h):
        for px in range(0,w):
	        if edges[py][px] == 255:
	        	x.append(px)
	        	y.append(py)

    size = len(x)
    xmin = 1000000000000
    xmax = 0
    ymin = 1000000000000
    ymax = 0
    for i in range(0,size):
        xmin = min(xmin,x[i])
        xmax = max(xmax,x[i])
        ymin = min(ymin,y[i])
        ymax = max(ymax,y[i])

    xdiff = xmax - xmin
    

    ydiff = ymax - ymin
    
    temp.append((problist[xdiff-85][0+1]))
    temp.append((problist[xdiff-85][1+1]))
    temp.append((problist[xdiff-85][2+1]))
    # temp.append(ydiff/100)
    temp.append((problist[xdiff-85][3+1]))
    temp.append((problist[xdiff-85][4+1]))
    temp.append((problist[xdiff-85][5+1]))  
    
    m =  cv2.imread(r_img_name)
    edges = cv2.Canny(m,100,200)
    h,w = np.shape(edges)
    x = []
    y = []
    for py in range(0,h):
        for px in range(0,w):
	        if edges[py][px] == 255:
	        	x.append(px)
	        	y.append(py)

    size = len(x)
    xmin = 1000000000000
    xmax = 0
    ymin = 1000000000000
    ymax = 0
    for i in range(0,size):
        xmin = min(xmin,x[i])
        xmax = max(xmax,x[i])
        ymin = min(ymin,y[i])
        ymax = max(ymax,y[i])

    xdiff = xmax - xmin
    ydiff = ymax - ymin
    temp.append((problist[xdiff-85][6+1]))
    temp.append((problist[xdiff-85][7+1]))
    temp.append((problist[xdiff-85][8+1]))
    # temp.append(ydiff/100)
    temp.append((problist[xdiff-85][9+1]))
    temp.append((problist[xdiff-85][10+1]))
    temp.append((problist[xdiff-85][11+1])) 
    
    return temp


def training():
    train_data = []

    for group in range(1,4):
        for roll in range(0,150):
            img1 = "/home/sidhdi/Desktop/5sem/nft/Fruit_classification/Final/"+str(group)+"_s1_"+str(roll)+".jpg"
            img2 = "/home/sidhdi/Desktop/5sem/nft/Fruit_classification/Final/"+str(group)+"_s2_"+str(roll)+".jpg"
            temp = feature(img1,img2)
            
            temp.append(group-1)
            train_data.append(temp)
    #neural_net(train_data)
    #print(train_data)
    
    
	  
	# # naming the x axis 
	# plt.xlabel('x - axis') 
	# # naming the y axis 
	# plt.ylabel('y - axis') 
	# # giving a title to my graph 
	# plt.title('3 lines on same graph!') 
	  
	# show a legend on the plot 
	 
	  
	# function to show the plot 
	
    return train_data

dataset = training()
print(dataset)

n_folds = 5
l_rate = 0.3
n_epoch = 500
n_hidden = 4
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: ', scores)
print('Mean Accuracy: ', (sum(scores) / float(len(scores))))

