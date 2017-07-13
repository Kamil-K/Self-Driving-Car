## Writeup - Kamil Kaczmarczyk
## 7th of May 2017
## Submission version 3

---

Modifications since version 1:
- car shift within the lane implemented
- additional color filtering to include in HLS colors not only S but addtionally L filter

Modifications since version 2:
- center of lane algorithm correction 

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./pipeline_images/test4.jpg - initial test picture
[image2]: ./pipeline_images/test4_undistorted.jpg - undistorted test picture
[image3]: ./pipeline_images/test4_gradient.jpg - binary test picture with the color and gradient applied
[image4]: ./pipeline_images/test4_warped.jpg - warped binary test picture with the color and gradient applied
[image5]: ./pipeline_images/test4_masked.jpg - masked region of interest on a warped binary test picture with the color and gradient applied
[image6]: ./pipeline_images/test4_lines_regions.jpg - lines regions visualised for the polynomial fit
[image7]: ./pipeline_images/test4_visualised_lane.jpg - final test image with the final found lane indicated on the image
[image8]: ./output_images/straight_lines1_PROCESSED.jpg - test image with the final lane drawn
[image9]: ./output_images/straight_lines2_PROCESSED.jpg - test image with the final lane drawn
[image10]: ./output_images/test1_PROCESSED.jpg - test image with the final lane drawn
[image11]: ./output_images/test2_PROCESSED.jpg - test image with the final lane drawn
[image12]: ./output_images/test3_PROCESSED.jpg - test image with the final lane drawn
[image13]: ./output_images/test4_PROCESSED.jpg - test image with the final lane drawn
[image14]: ./output_images/test5_PROCESSED.jpg - test image with the final lane drawn
[image15]: ./output_images/test6_PROCESSED.jpg - test image with the final lane drawn
[image16]: ./camera_calibration/calibration_test.jpg - test image used for camera calibration
[image17]: ./camera_calibration/calibration_test_undist.jpg - undistorted test image after camera calibration
[video1]: ./output_video/project_video_output.mp4 "Video" - final video with the lane drawn

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Answer:
The code for this step is contained in the second code cell of the IPython notebook located in "./model.ipynb" (or in lines 29 through 104 of the file called `model.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function in line 88.  I applied this distortion correction to the test image using the `cv2.undistort()` function in line 89 and obtained this result: 

[image17]: ./camera_calibration/calibration_test_undist.jpg - undistorted test image after camera calibration

For the reference the original image is:

[image16]: ./camera_calibration/calibration_test.jpg - test image used for camera calibration

One can clearly see the difference between these two pictures so it looks that camera calibration is actually necessary for this excercise.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Answer:
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images. The results are visible on the pictures below before and after calibration:

[image1]: ./pipeline_images/test4.jpg - initial test picture
[image2]: ./pipeline_images/test4_undistorted.jpg - undistorted test picture

There is a significant difference between these two pictures. Also it looks that correction works fine and later on in the pipeline already the undistorted images are used.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Answer:
I used color and gradient transform to create a binary image in the code from line 117 to 161 in the `model.py`. Example of this can be seen in the picture below:

[image3]: ./pipeline_images/test4_gradient.jpg - binary test picture with the color and gradient applied

To achive this result I used Sobel X in line 133 sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) as well as HLS color thresholding in line 143 with s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1.

NEW since version 1:
In version 2 also the L channel is used for filtering.

Overall thresholds used can be seen below:
result = pipeline(image, s_thresh=(170, 255), h_thresh=(18, 100), l_thresh=(200, 255), sx_thresh=(20, 100))

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Answer:
Perspective Transform is addressed from line 179 in the code through line 225. Example of a transformed image is presented below:

[image4]: ./pipeline_images/test4_warped.jpg - warped binary test picture with the color and gradient applied

The code for my perspective transform includes a function called `warp(image)`.

I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[743,485], [1040,680], [260,680], [546,485]])
offset = 400
dst = np.float32([[img_size[0]-offset, offset], 
		[img_size[0]-offset, img_size[1]], 
		[offset, img_size[1]],
		[offset, offset]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| [743,485]     | 320, 400      | 
| [1040,680]    | 840, 720      |
| [260,680]     | 400, 720      |
| [546,485]     | 400, 400      |

Code uses few significant lines of code:
M = cv2.getPerspectiveTransform(src, dst) --> to get the matrix M for the transform
Minv = cv2.getPerspectiveTransform(dst, src) --> to get the matrix Minv for the reverse transform

Finally also the warped image is obtained through:
warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)

I verified that my perspective transform was working as expected by visual inspection of the test images such as the one presented above and my conclusion was that the lines indeed appear relatively parallel to each other.

#### My next step was also to apply a mask onto the already warped binary image of the lane to get rid of the side lines which might confuse the detection algorithm. I did that in lines 228 to 277 of the python code. For this I used region_of_interest(img, vertices) function which takes the verticies in warped frame of reference coordinates to mask out the unnecessary lines.

Example of this code in action can be seen in the picture below:
[image5]: ./pipeline_images/test4_masked.jpg - masked region of interest on a warped binary test picture with the color and gradient applied

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Answer:

Lines 280 to 376 show a visualisation of the regions of lane lines and an output of this can be seen in the picture:

[image6]: ./pipeline_images/test4_lines_regions.jpg - lines regions visualised for the polynomial fit

This code however is for visualisation purposes only. The actual polynomial fit takes place in lines 379 to 477. The most important lines here are 452 and 453 which fit the polynomial using numpy polyfit function as below:
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Answer:
Radius curvature is calculated from line 479 through 496 using a specially built function give_curvature(left_fit_cr, right_fit_cr, ploty). The function takes advantage of the left_fit_cr, right_fit_cr which were calculated with the polyfit function of numpy in lines 458 and 459 with the applied conversion factors so that the final radius values will be in meters.
left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

NEW since version 1 - updated in version 2:
Position of the vehicle is calculated in lines 513 to 538. The actual calculations are as below. First the lane center is found through intersection points of two polynomial fits with the bottom of the image, dividing it by two and adding a shift of the lane due in x direction equal to the intersection of left line with the bottom of the image. Then the camera position is assumed to be exactly in the middle. Lane shift is a subtraction between the lane center and camera position multiplied by a conversion factor from pixels to SI units meters. Finally also the direction of shift is ecaluated.

y_eval = np.max(ploty)

left_fitx_max = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
right_fitx_max = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
 
lane_center = int((right_fitx_max - left_fitx_max)/2) + left_fitx_max
camera_position = image.shape[1]/2

if (lane_center - camera_position > 0):
		leng = 3.66/2 # Assume avg lane width of 3.66 m
		mag = ((lane_center-camera_position)/640.*leng)
		head = ("right", round(mag,2))
else:
		leng = 3.66/2.
		mag = ((lane_center-camera_position)/640.*leng)*-1
		head = ("left", round(mag,2))
			
#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Answer:

I implemented this step in lines 499 through 529 in my code in the function `visualise_lane()`.  Here is an example of my result on a test image:

[image7]: ./pipeline_images/test4_visualised_lane.jpg - final test image with the final found lane indicated on the image

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Answer:
Here's a [link to my video result]
[video1]: ./output_video/project_video_output.mp4 "Video" - final video with the lane drawn

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Answer:
In the current final video the lane is found pretty robustly. There is some degree of acceptable wobbliness but it occurs only in two places with very strong light and surface gradients. This however proves to be already problematic for the challenge videos where the results are not satisfactory and in order to perform well major modifications would have to be introduced.

Potential improvements could be:
- sanity checks between both right and left lines to check their parallelism and to use one line to verify and cross check the second one
- historical data from previous frames could also be used to improve the stability of the lane detection
- performance of computation could be also improved using more historical data rather than recalculating the lane position at each video frame
- potentially another thing that oculd be done is an actual geometrical modelling of potential solutions for the lane in front of the car to derive globally some parameters such as position of the car with respect to the center of the lane, the curvature angle of the turn and potentially another parameter related to pitch angle so that there would be a difference if car was going up- or downhill. This is potentially a cumbersome process but it could potentially make the lane finding pretty robust even in very poor visual conditions on very difficult terrain such as the mountain harder challenge video with high contrasts.
- one could imagine that future works could also correlate the video warped image in fron of the car with maps to improve the accuracy.
- potential another improvement is to track other cars and by observing their behaviour also deriving information about the general direction of movement which could give an idea of the line direction too.

