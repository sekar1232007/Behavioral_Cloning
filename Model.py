#26.04.2017 : Changed batch size to 128
#26.04.2017 : changes cv.imread to mpimg.imread

import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Convolution2D
import csv
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import sklearn

#Read the lines from driving_log.csv file and append them to
#samples list

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#Split the samples as train (80%) & validation set (20%)
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# generator function to avoid the memory error during training
#
def generator(samples, batch_size=128, Validation_set=False):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        
	   # Loop through all the samples in the batch	
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
		# for each line in the batch, split the line to 				# identify the individual imges. add images and  the 			#corresponding steering wheel value to image and 				#angles list respectively
            for batch_sample in batch_samples:
                name1 = './IMG/'+batch_sample[0].split('\\')[-1]
                center_image = mpimg.imread(name1)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                if(Validation_set==False):
                    name2 = './IMG/'+batch_sample[1].split('\\')[-1]
                    left_image = mpimg.imread(name2)
                    left_angle = float(batch_sample[3])+0.25
                    images.append(left_image)
                    angles.append(left_angle)

                    name3 = './IMG/'+batch_sample[2].split('\\')[-1]
                    right_image = mpimg.imread(name3)
                    right_angle = float(batch_sample[3])-0.25
                    images.append(right_image)
                    angles.append(right_angle)
                    
            # x_traing & y_traing contains required data for 			 # training
            X_train = np.array(images)
            y_train = np.array(angles)
		# shuffle the training data to avoid the bias due to 			#the order of training data
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128,Validation_set=False)
validation_generator = generator(validation_samples, batch_size=128,Validation_set=True)


#Build the model for training
model = Sequential()
#trim the image so that only road section is visible
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
# Preprocess incoming data, centered around zero with small #standard deviation
model.add(Lambda(lambda x: x/127.5 - 1.))

#Convolution Layer : Filer size 5x5,strides 2x2,feature maps 24
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))

#Convolution Layer : Filer size 5x5,strides 2x2,feature maps 36
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))

#Convolution Layer : Filer size 5x5,strides 2x2,feature maps 48
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))

#Convolution Layer : Filer size 3x3,strides 1x1,feature maps 64
model.add(Convolution2D(64,3,3,activation="relu"))

#Convolution Layer : Filer size 3x3,strides 1x1,feature maps 64
model.add(Convolution2D(64,3,3,activation="relu"))

#Flatten the model at this point
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
# Use adam optimizer for training
model.compile(loss='mse', optimizer='adam')

#train the model for 10 epochs
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*3, validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=10)
#save the model
model.save('model.h5')
