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
import csv

# a1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# a2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# a3 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# s1_x1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# s1_x2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# s1_x3 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# s1_y1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# s1_y2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# s1_y3 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# s2_x1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# s2_x2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# s2_x3 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# s2_y1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# s2_y2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# s3_y3 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

s = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

def extract_dim(img_name,r_img_name,group):
    
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
    s[group][xdiff-85] += 1
    s[group+3][ydiff-85] += 1
    
    
       
    
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
    s[group+6][xdiff-85] += 1
    s[group+9][ydiff-85] += 1


# def extract_dim(img_name,group,r_img_name):
# 	m =  cv2.imread(img_name)
# 	edges = cv2.Canny(m,100,200)
# 	h,w = np.shape(edges)
# 	x = []
#     y = []
#     for py in range(0,h):
#         for px in range(0,w):
# 	        if edges[py][px] == 255:
# 	        	x.append(px)
# 	        	y.append(py)


#     size = len(x)
#     xmin = 1000000000000
#     xmax = 0
#     ymin = 1000000000000
#     ymax = 0
#     for i in range(0,size):
#         xmin = min(xmin,x[i])
#         xmax = max(xmax,x[i])
#         ymin = min(ymin,y[i])
#         ymax = max(ymax,y[i])

#     s[group][xdi-85] += 1
#     s[group+3][xdi-85] += 1

#     m =  cv2.imread(r_img_name)
#     edges = cv2.Canny(m,100,200)
#     h,w = np.shape(edges)
#     x = []
#     y = []
#     for py in range(0,h):
#         for px in range(0,w):
# 	        if edges[py][px] == 255:
# 	        	x.append(px)
# 	        	y.append(py)

#     size = len(x)
#     xmin = 1000000000000
#     xmax = 0
#     ymin = 1000000000000
#     ymax = 0
#     for i in range(0,size):
#         xmin = min(xmin,x[i])
#         xmax = max(xmax,x[i])
#         ymin = min(ymin,y[i])
#         ymax = max(ymax,y[i])

#     xdiff = xmax - xmin
#     ydiff = ymax - ymin
#     s[group+6][xdi-85] += 1
#     s[group+9][xdi-85] += 1
    
    
    
def add_to_list(p):
    
    xcor = []
    a1 = s[p]
    a2 = s[p+1]
    a3 = s[p+2]
    print(a1)
    print(a3)
    print(a2)
    a1p = []
    a2p = []
    a3p = []
    for i in range(85,100):
    	xcor.append(i)
    	tot = a1[i-85] + a2[i-85] + a3[i-85]
    	if tot == 0:
    		prob = 0
    	else:
    		prob = a1[i-85]/tot
    	a1p.append(prob)
    	if tot == 0:
    		prob = 0
    	else:
    		prob = a2[i-85]/tot
    	a2p.append(prob)
    	if tot == 0:
    		prob = 0
    	else:
    		prob = a3[i-85]/tot
    	a3p.append(prob)
    plt.plot(xcor, a1p, label = "line 1") 
    plt.plot(xcor, a2p, label = "line 2") 
    plt.plot(xcor, a3p, label = "line 3") 
    plt.legend()
    plt.show() 
    a = [a1p,a2p,a3p]
    return a1p,a2p,a3p


# def add_to_list(p):
# 	xcor = []
# 	a1 = s[p]
# 	a2 = s[p+1]
# 	a3 = s[p+2]
#     print(a1)
#     print(a3)
#     print(a2)
#     a1p = []
#     a2p = []
#     a3p = []
#     for i in range(85,100):
#     	xcor.append(i)
#     	tot = a1[i-85] + a2[i-85] + a3[i-85]
#     	prob = a1[i-85]/tot
#     	a1p.append(prob)
#     	prob = a2[i-85]/tot
#     	a2p.append(prob)
#     	prob = a3[i-85]/tot
#     	a3p.append(prob)
#     print(a1p)
#     print(a3p)
#     print(a2p)
#     plt.plot(xcor, a1p, label = "line 1") 
#     plt.plot(xcor, a2p, label = "line 2") 
#     plt.plot(xcor, a3p, label = "line 3") 
#     plt.legend()
#     plt.show() 
#     a = [a1p,a2p,a3p]
#     return a





def training():
    xcor = []

    for group in range(1,4):
        for roll in range(0,150):
            img1 = "/home/sidhdi/Desktop/5sem/nft/Fruit_classification/Final/"+str(group)+"_s1_"+str(roll)+".jpg"
            img2 = "/home/sidhdi/Desktop/5sem/nft/Fruit_classification/Final/"+str(group)+"_s2_"+str(roll)+".jpg"
            extract_dim(img1,img2,group-1)

    for i in range(85,100):
    	print(i)
    	xcor.append(i)

    final = []
    # final.append(xcor)

    for group in range(0,4):
    	print(group)
    	a1,a2,a3 = add_to_list(group*3)
    	final.append(a1)
    	final.append(a2)
    	final.append(a3)
    	print(a1,a2,a3)

    write_file = []

    print("xcor",xcor)
    for i in range(85,100):
    	temp = []
    	temp.append(xcor[i-85])
    	print(i,xcor[i-85])
    	for group in range(0,12):
    		# print(group)
    		temp.append(final[group][i-85])

    	write_file.append(temp)

    print("fin",xcor[2])

    with open("output.csv", "w") as f:
    	writer = csv.writer(f)
    	writer.writerows(write_file)
	    
	    
training()
