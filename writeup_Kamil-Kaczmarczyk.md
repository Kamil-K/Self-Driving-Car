#**Finding Lane Lines on the Road** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 7 steps:

Step 1: First the image is converted to grayscale.

Step 2: Secondly the canny transform is applied to the image to detect edges.

Step 3: Then a Gaussian Noise Kernel is applied to the image for image and lines noise reduction.

Step 4: 4th step consists of selecting the region of interest by applying an image mask. This is done using vertices defining the boundaries of the region of interest.

Step 5: After that Hough Transform is applied to the lines by changing the coordinate system to focus only on the real road lane lines.

Step 6: Next step is to use the weighted image function.

Step 7: Final Step is to draw lines corresponding to lane lines.

Function draw_lines() has been modified for the purpose of this project. The modification consisted of few steps:
- divide all lines into the ones with negative and non-negative slopes. It is assummed that in normal conditions the negative slope would correspond to left lane line and the non-negative to the right lane line (please note the coordinate system has its origin on the top left corner)
- find the average left line and average right line parameters (average slopes, average intercept parameters and average point on each line to fit the final lines)
- find bottom points (intercept point) and top points (extension point) for both left and right average line
- draw both left and right lines on the final image

![alt text][image1]


###2. Identify potential shortcomings with your current pipeline

Potential shortcomings for this pipeline are:

1. It does not work well for curved or bent roads. This is due to the fact that some of the slopes of certain lines on the image might be not always of one sign for a single side. For example for a right turn some of the right lines might have a positive and some might have a negative slope as the lines are curved. Additionally as the lines are becoming vertical the slope goes into infitinty which might cause some computational problems in the code.

2. The lines are not stable and keep jittering for video and they are easy to be influenced by any noise on the image of the road.

3. Sometimes in the video, the slopes of some lines were NaNs and in this case they were not taken into account for the final average slope. This might have also contributted to the jitter as explained in the second shortcoming.

4. The pipieline works only under very strick assumptions:
- algorithm works if car is already in the middle of the lane. If the car was in other place on the road then perhaps the pipieline would not work as well, for example due to the image mask settings and verticies.
- algorithm seems to work only short range of up to 50 meters forward. This is due to the quality image and due to what can be seen by the camera itself.
- if the lanes are not distinguishable on the road surface then the algorithm does not work. For example if there is no lights at night then the lanes might not be visible. Lighting in general is crucial for vision based lane recognition.
- if any other obstacle or object obstructs the view of a lane line then the algorithm might not have a perfect detection. For example if a truck takes over the car and starts obstructing one of the lane lines then the lane detection might not work.
- if lanes are not marked by lines but by some retro-reflectors then it might confuse the vision based lane detection.

In general the algorithm only works in very specific conditions on very selected environments.



###3. Suggest possible improvements to your pipeline

Possible improvements to the pipeline could be:
1. Get rid of the jitter by introducing some time based moving filter which takes historical images of data into account to stabilise the lane lines detection. In the current state only the present image is taken into account and probably it would bring better results to rely on historical data from let's say 1 second in the past.
2. Instead of using only straight lines a better retuls would be obtained if some kind of polynomial function was used. This would improve the lane detection at the corners.
3. Additional information could be fused together for better quality of lane detection. For example satellite imagery with lanes pictures of GPS positioning for middle of the lane detection.
4. Robustness of the algorithmm could be also improved by deriving and correlating information form one lane line to the other. For example, if right hand side line is obstructed by an object, say a truck taking over the car then the lane information could be retrieved from the left lane and by knowing the perspective and that that a lane is on average of constant width the info of the right lane lines could be retrieved and rebuilt.

