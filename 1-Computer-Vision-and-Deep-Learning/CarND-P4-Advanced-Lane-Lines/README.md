## Writeup

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

[//]: # "Image References"

[cal_img]: ./examples/cal.jpg "Undistorted"
[test1]: ./test_images/test1.jpg "Original"
[test1_undistorted]: ./examples/undistorted.jpg "Undistorted"
[color_channel]: ./examples/color_channel.jpg "color_channel"
[threshold]: ./examples/threshold.jpg "Threshold"
[Combined_threshold]: ./examples/combined_threshold.jpg "Combined_threshold"
[Pers_before1]: ./examples/pers_before1.jpg "Pers_before1"
[Pers_after1]: ./examples/pers_after1.jpg "Pers_after1"
[Re_pers]: ./examples/re_pers.jpg
[Pers_before2]: ./examples/pers_before2.jpg
[Pers_after2]: ./examples/pers_after2.jpg
[Line_detection]: ./examples/line_detection.jpg
[Road]: ./output_images/test1_detected.jpg
[video1]: .project_video/project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the third code cell of the IPython notebook located in "./test.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  Using `cv2.findChessboardCorners ` ,   `imgpoints` will be found and appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][cal_img]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][test1]

When using `cv2.undistort` to apply distortion correction, I got the following image:

![][test1_undistorted]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code is in the 14th cell in the jupyter notebook. 

I first visualized multiple color channels to help choose an approriate one. 

![][color_channel]

Visualizing all the test images, I found that  the S channel in HLS mode is relatively most insensitive to the illumination, shadows and etc, and then is the R channel in RGB mode. However, the S channel lost some sharp edge information in the far place, which harms the edge detection via gradient threshold approach. Thus, I chose the R channel as the basis of graident threshold filters, which can preserve the edge information. I performed x and y gradient thresholds, as well as gradient magnitude and direction thresholds. However, the R channel was sometimes still senstive to the illumination. So R-channel-based gradient method should be combined with S channel color threshold.  And the S channel cannot filter the cars so well, which disturbs the edge detection. I jointly used  another R channel color threshold to filter the cars. 

The visualization of thresholds are as follows. Each line is a series of transforms to one of the test image. The first column is the original image. The second is the undistorted image. The third is the R channel in RGB mode. The fourth is the x gradient threshold on R channel. The fifth is the y gradient threshold on R channel. The sixth is the gradient magnitude threshold on R channel. The seventh is the gradient direction threshold on R channel. The eighth is the S channel color threshold. The ninth is the R channel color threshold. The tenth is the combined threshold.

![][threshold]

For example, the binary image of the combined threshold of the above test image is 

![][Combined_threshold]



#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code is in the 10th, 11st, 12nd and 13rd cells in the jupyter notebook. 

  I chose the hardcode the source and destination points in the following manner:

```python
src_region = np.float32([[441,560], [849,560], [215,719], [1090,719]])
dst_region = np.float32([[250,630], [900,630], [250,719], [900,719]])
```

I found that the two upper points should not be chosen too far in the road, which made the two points too close in a trapezoid and may result in the singularity in calculating the perspective matrix. 

I verified that my perspective transform was working as expected by verifing that the lines appear parallel in the warped image.

![alt text][Pers_before1]

![][Pers_after1]

And perfom inverse perspective, I got

![][re_pers]

Another example was

![][Pers_before2]

![][Pers_after2]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code is also in the 14th cell in the jupyter notebook, where I built a pipeline to work on all images in the 'test_images' folder. 

I used 9 sliding windows for each line in each image. First I found all the lane-line pixels and then fit them with a second-order polynomial. 



![alt text][Line_detection]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code is also in the 14th cell in the jupyter notebook, where I built a pipeline to work on all images in the 'test_images' folder. 

As I fit the lane lines with a second-order polynomial, I can calculate the radius of curvature using an analytical formula, which was derived in the course material. 

To calculate the position of the vehicle, I used the assumption that the camera was mounted at the center of the vehicle. Therefore, I used the distance between the center of the image and the center of lane lines and converted it to the real-world units by multiple a factor. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code is also in the 14th cell in the jupyter notebook, where I built a pipeline to work on all images in the 'test_images' folder. Here is an example of my result on a test image:

![alt text][Road]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

When trasferring the pipeline to the video, some outliners may occur. To filter those outliners, I first created  a class of line and got two instances: left_line and right_line. The line class stores some information of the lane lines. When outliners occurs, the program can ignore the them and use the information to draw a more reasonable line. 

To filter the outliners, sanity check should be done first. I used four conditions: 1. the radius of two lines cannot differ too much; 2. the detected lane should be reasonably wide as in reality; 3. the width of the detected lane should not differ too much at each point; 4. the direction of the quadratic curves should be consistent. If all conditions can be satisfied, the lines are credible and should be added into the line class. If not, discard the detection. 

To smooth the detection, I averaged the last five credible detections to draw the lane lines, which can also filter some noise and outliners. 

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

In my implementation, the biggest problem I faced was that the performance heavily depended the hand-craft features and thresholds. As it took much time and there didn't exist a explicit approach, this method tended to be instable and was not robust enough to tackle many different scenarios. When there are more disturbances like patches on the road surface, it would fail, as validated in the challenge video. To improve that, more robust features should be extracted. Also, the algorithm should exploit more intuitions about what a lane should be, what a lane should correspond to and etc. One potential approach might be using semantic information. 