import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images
import csv
import os
import shutil, sys  
import imutils

os.mkdir('c')
os.mkdir('s')
os.mkdir('train_data')
from distutils.dir_util import copy_tree
copy_tree("combined_text", "c")
from distutils.dir_util import copy_tree
copy_tree("single_text", "s")
os.getcwd()
collection = 'c'
for i, filename in enumerate(os.listdir(collection)):
    os.rename("c/" + filename, "train_data/" + str(i) + ".jpg")
    img = cv2.imread("train_data/%d.jpg" %i)
    img=cv2.resize(img, (224,224))
    cv2.imwrite("train_data/%d.jpg" %i, img)

j=i
h=i+1
collection = 's'
for i, filename in enumerate(os.listdir(collection)):
    j=j+1
    os.rename("s/" + filename, "train_data/" + str(j) + ".jpg")
    img = cv2.imread("train_data/%d.jpg" %j)
    img=cv2.resize(img, (224,224))
    cv2.imwrite("train_data/%d.jpg" %j, img)




a1=0
a=0
b=j+1



file = open('csv.csv', 'w', newline ='') 
  
with file: 
    # identifying header   
    header = ['Image_ID', 'Class'] 
    writer = csv.DictWriter(file, fieldnames = header) 
      
    # writing data row-wise into the csv file 
    writer.writeheader()
    for i in range(a,b):
        
       

         if(a1<h):
            writer.writerow({'Image_ID' : 'train_data/'+str(i)+'.jpg',  
                     'Class': '0'})
         else:
             writer.writerow({'Image_ID' : 'train_data/'+str(i)+'.jpg',  
                     'Class': '1'})
         a1=a1+1
            
    
data = pd.read_csv('csv.csv')
X = [ ]     # creating an empty array
for img_name in data.Image_ID:
    img = plt.imread('' + img_name)
    X.append(img)  # storing each image in array X
X = np.array(X)    # converting list to array
y = data.Class
dummy_y = np_utils.to_categorical(y) 
image = []
for i in range(0,X.shape[0]):
    a = resize(X[i], preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
    image.append(a)
X = np.array(image)
from keras.applications.vgg16 import preprocess_input
#X = preprocess_input(X, mode='tf')      # preprocessing the input data
tf.keras.applications.vgg16.preprocess_input(X, data_format=None)
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.3, random_state=42)    # preparing the validation set
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)
X_train.shape, X_valid.shape
s=j+1
h=int(s*0.7)
X_train = X_train.reshape(h, 7*7*512) # 659     # converting to 1-D
X_valid = X_valid.reshape((s-h), 7*7*512)
train = X_train/X_train.max()      # centering the data
X_valid = X_valid/X_train.max()
# i. Building the model
model = Sequential()
model.add(InputLayer((7*7*512,)))    # input layer
model.add(Dense(units=1024, activation='sigmoid')) # hidden layer
model.add(Dense(2, activation='softmax'))    # output layer
model.summary()
# ii. Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# iii. Training the model
model.fit(train, y_train, epochs=10, validation_data=(X_valid, y_valid))
print("done")
print("[INFO] saving combined character model...")
model.save("combined_model.h5")
print("Model have been saved sucessfully............!!!!!!!")



shutil.rmtree('s')
shutil.rmtree('c')
shutil.rmtree('train_data')
os.remove("csv.csv")

