#**Behavioral Cloning** 

##Writeup Report - Kamil Kaczmarczyk

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: .\\MSE_loss.png "Model Visualization of Mean Squared Error"
[image2]: .\\examples\\center_2017_05_01_14_29_53_184.png "Recovery Image Part 1"
[image3]: .\\examples\\center_2017_05_01_14_29_53_286.png "Recovery Image Part 2"
[image4]: .\\examples\\center_2017_05_01_14_29_53_389.png "Recovery Image Part 3"
[image5]: .\\examples\\center_2017_05_01_14_29_53_489.png "Recovery Image Part 4"
[image6]: .\\examples\\center_2017_05_01_11_46_36_244.png "Normal Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

Note: the direcotries of folders with training datasets are hardcoded and perhaps need to be changed in order to load the training data.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

---Final Approach---

My model is based on the network architecture used in the Nvidia paper: "End to End Learning for Self-Driving Cars" and is also given below in Keras. First of the images are normalized and cropped. Then they are fed through 3 layers of convolution networks with strides 2 by 2 and relu activation functions. after that 2 convolution networks also with relu activation are used. Finally at the end 3 fully connected layers are implemented.

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

---Story of how the final architecture model was selected---

First I started with a basic architecture with one layer, normalized data and adam optimizer - this approach allowed the car to barely move but was very far away from driving. Secondly I implemented the LeNet architecture and I could already noticed that the car was driving much more dynamically but at all tries it fell of the track. Next step was to flip the images to increase the number of training samples. This still produced similar results as the LeNet with car driving directly of the track. Next step was to use 3 cameras' images: center, left and right to add even more data and cropping of the images. This further increased the results but was not satisfactory.

After all these approaches still failed to produce reliable results I decided to:
- collect much more data as until now I was using only few tausands of images --> at the end I had more than 28,000 of images with very diversified driving, track and steering scenarios in both directions
- at the same time I decided to implement my final architecture based on Nvidia approach which can be seen above

---Future work and improvements---

The network so far is focused on the current track one, so one idea would be to drive on the second track and generalize it a bit more.

Another improvement would be to use the generator functions if additional training set was used or extra data augmentation. This should improve the overal model performance and for the future work the generator function should be constructed and employed.

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting with 20% of the data being used for validation and 80% for training. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

Model architecture and training sets were mainly used to improve the accuracy and performance of the model.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used:
- 2 laps with center driving, smooth cornering and normal track direction --> see image6 as an example
- 2 laps with center driving, smooth cornering and reverse track direction --> see image6 as an example
- 1 lap with recovery situations equally distribted from left and right in a normal track direction --> see image2, image3, image4, image5 as an example
- 1 lap with recovery situations equally distribted from left and right in a reverse track direction --> see image2, image3, image4, image5 as an example
- few recovery situations on tricky corners --> see image2, image3, image4, image5 as an example

At the end of the process, the vehicle is able to drive autonomously around the track but the dataset needed to be supplemented with some additional recordings of driving behavior of some tricky corners.

####5. Creation of the Training Set & Training Process

Preprocessing of images included:
- cropping of the images from top and bottom to only leave the road its countours visible.
- normalizing the images
- flipping the images and reversing the steering angles for the flipped pictures
- using all 3 camera inputs with adjusted steering angles by a correction factor of 0.2
- data is also shuffled to split it into training and validation sets
- 20% of all data is used for validation and 80% for training

At the end I had more than 28000 of images collected to train on the model.

####6. Results & Performance

Final files are:
- model is stored under model.h5
- final Mean Square Loss plot is provided in [image1]: .\\MSE_loss.png "Model Visualization of Mean Squared Error"
- final movie recording of the autonomous run is stored in the video.mp4 file

Generally car behaves pretty well and it stays within the track, drives at a reasonable speeds and would be considered safe for humas traveling onboard.
