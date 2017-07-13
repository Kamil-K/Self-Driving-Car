## Writeup Report
## Kamil Kaczmarczyk 25th May 2017
## Version 1

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References):

[image1]: ./examples/car HOG image_test.png
[image2]: ./examples/car image_test.png
[image3]: ./examples/notcar HOG image_test.png
[image4]: ./examples/notcar_image_test.png

[image5]: ./examples/test_bboxes1.png
[image6]: ./examples/test_bboxes2.png
[image7]: ./examples/test_bboxes3.png
[image8]: ./examples/test_bboxes4.png
[image9]: ./examples/test_bboxes5.png
[image10]: ./examples/test_bboxes6.png

[image11]: ./examples/test1_org_with_boxes.png
[image12]: ./examples/test2_org_with_boxes.png
[image13]: ./examples/test3_org_with_boxes.png
[image14]: ./examples/test4_org_with_boxes.png
[image15]: ./examples/test5_org_with_boxes.png
[image16]: ./examples/test6_org_with_boxes.png

[image17]: ./examples/test1_just_heatmap.png
[image18]: ./examples/test2_just_heatmap.png
[image19]: ./examples/test3_just_heatmap.png
[image20]: ./examples/test4_just_heatmap.png
[image21]: ./examples/test5_just_heatmap.png
[image22]: ./examples/test6_just_heatmap.png

[video1]: ./final.mp4
[video2]: ./test.mp4 

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Answer:

HOG features are extracted using the function "get_hog_feature" defined in lines 83 through 100 in the file model.py.

This is then called either in "extract_features" (lines 139 through 187) if we want to extract features from a list of images or in a function "single_img_features" (Lines 248 through 300) if we want to extract features only from a single image.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. The parameters used below are defined in lines 374 to 381. Here the example uses the `RGB` color space and HOG parameters of `orientations=6`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, `hog_channel=0`, `spatial_size = (16, 16)` and `hist_bins = 16`:

color_space = 'RGB'
orient = 6 
pix_per_cell = 8
cell_per_block = 2
hog_channel = 0
spatial_size = (16, 16)
hist_bins = 16

Using a call for a single image "single_img_features" in lines 386 to 391 we can see actual random vehicle picture with its hog features:
[image1]: ./examples/car HOG image_test.png
[image2]: ./examples/car image_test.png

Using a call for a single image "single_img_features" in lines 392 to 397 we can see actual random non-vehicle picture with its hog features:
[image3]: ./examples/notcar HOG image_test.png
[image4]: ./examples/notcar_image_test.png


####2. Explain how you settled on your final choice of HOG parameters.

Answer:

I tried various combinations of parameters and the final selection is defined in lines 415 through 427. They are `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, `hog_channel="ALL"`, `spatial_size = (32, 32)` and `hist_bins = 16`:

color_space = 'YCrCb'
orient = 9 
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"
spatial_size = (32, 32)
hist_bins = 16

Initial choice for color space was 'RGB' but after some tests it did not seem like the best choice. After some experimentation the final choice was on 'YCrCb'. As for the number of orientations the final choice was switched from 6 to 9. It seem to be a reasonable choice due to the fact that after increasing the number of possible orientations the results did not seem to improve that much. Also hog channels were switched to activate all of them (`hog_channel="ALL"`). This is due to the fact that it seems to improve accuracy when all channels are taken into account instead of just one (despite the increase in computational cost which is reasonable). Final parameter that was modified from the initial conditions was the spatial size which increased from (16,16) to (32, 32). Again some improvements are visible at a reasonable computational cost.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Answer:

I trained a linear SVM using "svc.fit" function ('svc.fit(X_train, y_train)') in line 479 of the code. It takes on the training dataset which constitutes to 90% of the overall data used from KIIT database.

The training set contains the stacked scaled vehicle and non-vehicle features as well as labels for vehicles and non-vehicles. The features used are related to HOG of course but also additionally the histogram and spatial features are used and activated in the function extract_features in lines 441 to 446 for vehicle and 447 to 452 to non-vehicle.
 - Spatial features use function bin_spatial(feature_image, size=spatial_size) in line 165 with its definition in line 102.
 - Color histogram features use color_hist(feature_image, nbins=hist_bins) in line 169 with its definition in line 115.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Answer:

Function for defining the windows position/size is "slide_window" and it is defined in lines 193 through 232 and it is called in line 511. It takes as arguments the image positions whithin which to search for windows, then window sizes as well as the overlap between windows.

After some experimentations the most reasonable parameters seem to be the ones below (defined in lines 498 to 501). Defining y_start_stop focused the model to search within the regions where cars could possibly appear and it has a direct influence on the overall number of windows hence computational time. By selecting values y_start_stop = [400,656] the window is limited only to the bottom half of the image. This is also defined in lines 558 and 559 for the final implementation with a heatmap. Window size of 96 seems also to work reasonably well after experimenting of different sizes on test images from 16 to 128. Finally the standard overlap value of 0.5 is picked and again it behaves pretty well.

xy_window = 96
overlap = 0.5
y_start_stop = [400,656]

Additionally scale parameter is used of 1.5 as can be seen in line 560. After some experimentation of smaller and larger values this number performed okay.

scale = 1.5

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Answer:

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images with their corresponding bounding boxes:

[image5]: ./examples/test_bboxes1.png
[image6]: ./examples/test_bboxes2.png
[image7]: ./examples/test_bboxes3.png
[image8]: ./examples/test_bboxes4.png
[image9]: ./examples/test_bboxes5.png
[image10]: ./examples/test_bboxes6.png

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)

Answer:

Final and test videos are provided below:
[video1]: ./final.mp4 --> final video for submission
[video2]: ./test.mp4 --> short test video for implementation verification

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Answer:

Function filtering false positives "apply_threshold" is defined in the line 755 to 759. It takes in a heatmap of an image with car detections and thresholds it leaving only the detection exceeding the threshold itself. Threshold of 1 is used to exclude false positives. The function however is defined in such a way that the edges of heat concentrations are also cut out of the final heatmap if they do not equal or exceed the threshold. This significantly increased the wobbliness of the windows and also made not matching tightly around the countours of cars.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

### Here are six frames and their corresponding heatmaps:

[image11]: ./examples/test1_org_with_boxes.png
[image12]: ./examples/test2_org_with_boxes.png
[image13]: ./examples/test3_org_with_boxes.png
[image14]: ./examples/test4_org_with_boxes.png
[image15]: ./examples/test5_org_with_boxes.png
[image16]: ./examples/test6_org_with_boxes.png

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

[image17]: ./examples/test1_just_heatmap.png
[image18]: ./examples/test2_just_heatmap.png
[image19]: ./examples/test3_just_heatmap.png
[image20]: ./examples/test4_just_heatmap.png
[image21]: ./examples/test5_just_heatmap.png
[image22]: ./examples/test6_just_heatmap.png

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Answer:

Model is:
- prone to false positives - with multiple shadows, contours on the road etc. it detects multiple false cars and perhaps additional tuning of the model would have to be implemenented to account for that. Including thresholding is one solution but another one would be to use a time series tracking of a car to account for multiple frames in a video and not just looking one frame at a time.
- not detecting cars when they far away at a distance - this could be perhaps solved by using even smaller window size to search through the image.
- not detecting cars when only part of it is visible for example when one car is partially obstructed by another vehicle - this perhaps require a complete change in the model to not train it only to detect entire cars but to specifically train it to detect parts and countours of a car itself, such as its corners, lights, wheels etc.
- not detecting cars in urban environmnent (not only motorway) when cars are visible from a different angle such as driving towards us or completely sideways - this probably requires an additional training set with cars visible from all possible angles.
- not running in real time - improvements could be made to run in near real time (at least several frames per second).