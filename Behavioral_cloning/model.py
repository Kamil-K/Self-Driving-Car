### Author: Kamil Kaczmarczyk
### Code based on model version 8 - 01.05.2017

### Import of needed functions
import csv
import cv2
import numpy as np

### Read the driving_log.csv file for each frame of the dataset
lines = []
with open('D:\\Studies-MOOCs\\2017-02_Self-Driving-Car_NanoDegree_Udacity\\Self-Driving_Car_PythonCode\\CarND-Behavioral-Cloning-P3-master\\CarND-Behavioral-Cloning-P3-master\\data\\data_training_2\\driving_log.csv') as csvfile:
    
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
### Read all the lines with images and measurements of the drive_log.csv file
images = []
measurements = []
for line in lines:
	#loop through all three camera images: center, left and right
    for i in range(3):
		
		#find the directory of the image
        source_path = line[i]
        filename = source_path.split('\\')[-1]
        current_path = 'D:\\Studies-MOOCs\\2017-02_Self-Driving-Car_NanoDegree_Udacity\\Self-Driving_Car_PythonCode\\CarND-Behavioral-Cloning-P3-master\\CarND-Behavioral-Cloning-P3-master\\data\\data_training_2\\IMG\\' + filename
        
		#append the specific image to all images
        image = cv2.imread(current_path)
        images.append(image)
        
		#for the middle camera read directly the steering angle measured
        steering_center = float(line[3])
		
		#for left and right camera images apply a correction of steering angle of 0.2 degrees (assumed value)
		#this approach is based on the Nvidia "End to End Learning for Self-Driving Cars" paper
		correction = 0.2
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        
		#check and assign to center, left or right camera image the right steering angles
        if i == 0: #center camera
            measurement = steering_center
        elif i == 1: #left camera
            measurement = steering_left
        elif i == 2: #right camera
            measurement = steering_right
            
		#append the specific steering angle measurement to all measurements
        measurements.append(measurement)
     
### Augment images with flipped pictures and reversed steering angles
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1)) #flip images
    augmented_measurements.append(measurement*-1.0) #reverse steering angles
 
### Assign the training set
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

### Import of needed functions
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

### Build the model - based on Nvidia approach
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3))) #Normalize the images
model.add(Cropping2D(cropping=((70,25),(0,0)))) #Crop the images (top and bottom parts)
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu")) #Add Conv net with a stride and relu activation
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu")) #Add Conv net with a stride and relu activation
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu")) #Add Conv net with a stride and relu activation
model.add(Convolution2D(64,3,3,activation="relu")) #Add Conv net and relu activation
model.add(Convolution2D(64,3,3,activation="relu")) #Add Conv net and relu activation     
model.add(Flatten())
model.add(Dense(100)) #Add a fully connected layer
model.add(Dense(50)) #Add a fully connected layer
model.add(Dense(10)) #Add a fully connected layer
model.add(Dense(1))  

### Compile and fit the model

#Adam optimizer used and mean square error for loss
model.compile(loss='mse', optimizer='adam') 
#Training-validation split is 80%-20%, shuffling of the dataset applied with 5 Epochs
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5, verbose=1)

#Save the model
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
import matplotlib.pyplot as plt

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()