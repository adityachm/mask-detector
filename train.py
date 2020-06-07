import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,BatchNormalization,MaxPooling2D,Flatten,Dense,Dropout
# tf.config.set_visible_devices([], 'GPU')
# path to your dataset
path = "data/"
image_size = (150,150)

#initializations
data = []
#looping through all the folders
folders = os.listdir(path)
for folder in folders: 
    new_path = os.path.join(path,folder)
    total = len(os.listdir(new_path))
    label = folders.index(folder)
    print('folder_name = {} and no of images = {}'.format(folder,total))
    for img in os.listdir(new_path):#loop through all the images in that folder
        #read the images one by one
        try:
            img = cv2.imread( os.path.join(new_path,img),0) #grayscale
            img = cv2.resize(img,image_size) #check here once
            data.append([img,label])
        except Exception as e:
            print('Exception ',e)
            
#shuffle the data in place
import random
random.shuffle(data)

#seperating images are labels after shuffling
images = []
labels = []
for image,label in data:
    images.append(image)
    labels.append(label)
    

    
#normalizing the images 
images = np.array(images)/255
labels = np.array(labels)

#converting to 4D 
images = np.expand_dims(images,axis = 3)

# seperating train and test dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(images,labels,test_size = 0.1,random_state = 0)



# Custom model
#initializing the model
model=Sequential()

#The first CNN layer followed by Relu and MaxPooling layers
model.add(Conv2D(512,(3,3),activation = 'relu',input_shape=images.shape[1:]))
model.add(MaxPooling2D())

#The second convolution layer followed by Relu and MaxPooling layers
model.add(Conv2D(256,(3,3),activation = 'relu'))
model.add(MaxPooling2D())

#Flattening 
model.add(Flatten())

#Adding dropout to avoid overfitting
model.add(Dropout(0.5))

#Dense Layer of 128 neurons
model.add(Dense(128,activation='relu'))

#Output layer
model.add(Dense(1,activation='sigmoid'))



model.compile(optimizer='adam',loss = 'binary_crossentropy',metrics = ['accuracy'])


# BATCH_SIZE = 32
EPOCHS = 10
history=model.fit(X_train,y_train,epochs=EPOCHS, validation_data = (X_test,y_test))

model_name = 'model_updated.h5'
path = '/content/gdrive/My Drive/models/mask-detector/model' + '/' + model_name
#save the model
model.save(path)