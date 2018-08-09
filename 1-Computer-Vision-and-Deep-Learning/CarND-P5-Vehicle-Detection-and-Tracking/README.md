## Writeup
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # "Image References"
[Data-exploration-1]: ./examples/1-data-exploration-vehicles.png
[Data-exploration-2]: ./examples/2-data-exploration-non-vehicles.png
[Samples-colorspace]: ./examples/3-data-samples-colorspace.png
[Test-image-colorspace]: ./examples/4-test-image-colorspace.png
[Vehicle-HOG]: ./examples/5-vehicle-HOG.png
[Non-vehicle-HOG]: ./examples/6-non-vehicle-HOG.png
[Example-1]: ./examples/7-example-1.png
[Example-2]: ./examples/8-example-2.png
[Example-3]: ./examples/9-example-3.png
[Example-4]: ./examples/10-example-4.png
[video1]: ./project_video/project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  



### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 8th code cell of the IPython notebook. 

Before extracting HOG features, I first explored the data set (code cell 2) and different color spaces (code 3, 4 and 5). The total number of vehicle samples is 9214 (after data augmentation) and the number of non-vehicles is 10743 (after data augmentation). Data augmentation will be explained below when describing the SVM classifier. both 5 randomly picked samples of vehicles and non-vehicles are shown as follows

![alt text][Data-exploration-1]

![][Data-exploration-2]

As seen, almost all the vehicle samples have bounding boxes that fit their size, not too large or not too small. This issue will be recalled when we discuss sliding windows below. Also, few shadow senarios can be seen from the non-vehicle samples. Then I explored the color space using the five vehicle samples. 

![][Samples-colorspace]

In HSV mode, features in each color channels were relatively independent, not like in the RGB mode. Independent/orthogonal features can help the learning process. Also, V channel preserved the shape information, which is useful in extracting HOG feature. Another test image was picked. Similar conclusions could be found. 

![][Test-image-colorspace]

Therefore, I used V channel in HSV mode to extracting the HOG feature. I then explored different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. Here is an example using the HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![][Vehicle-HOG]

![][Non-vehicle-HOG]



#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and chose a combination so that the HOG feature of a vehicle generally formed a closed-loop square outside and some horizontal lines inside the square while the HOG of a non-vehicle did not show this property. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Before training a classifier, I first extracted the features of color histograms and bins (code cell 6 and 7), also in HSV mode because orthogonal features help learning. Then I concatenated the three types of features (code cell 9). I performed the feature extraction over the entire data set (code cell 10 and 11), also including the shuffling, train/test splitting, and feature normalization using StandardScaler(). 

I trained a RBF-kernel SVM classifier in code cell 14 and 15 with parameters C=1.0 and gamma='auto'. Then I got the accuracy over training set of 99.66% and the accuracy over testing set (actually the dev set) of 99.15%. It revealed that the classifier was well trained, neither underfitting nor overfitting on the training set. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented a sliding window search in code cell 20, 21 and 22. The windows were created slidingly step by step in horizontal and then vertical directions. I also implemented 'HOG only once per frame' in code cell 24. 

The shape size and step size of sliding windows really mattered. 

As mentioned above, almost all the training samples of vehicles had bounding boxes that fit their size, not too large or not too small. However when using sliding windows to crop the image, it was difficult to get an appropriate bounding box, which may cause that features of the vehicle are not significant enough within the sliding windows. In nature, this phenomenon is distribution mismatching between training and test data. I tested different scales. Generally, I only searched the windows on the lower half of the image because vehicles could only appear in that region. Then I only search small region near the horizon when using small scales because vehicles there are very far from the camera. 

Large step size resulted in small number of windows and accelerated the search process. However, large step size could not cover enough sub-images and could also make the feature in a window not significant enough. So I decreased the step size by experiment at the cost of computation. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Some examples of test images. The pipeline worked well. The vehicles were all detected though some false positives existed. Since the bounding boxes of false positives were far less than the vehicles', we could filter them by setting a threshold on the heat map. 

![][Example-1]

![][Example-2]

![][Example-3]



Actually, above images were obtained after data augmentation and retraining the classifier. The initial test image from the video was like this one. As seen, the bounding boxes of false positives roughly equalled to the black car's, which could not be filtered by heat map thresholding. The shadows and black car could not be seperated easily. It was mainly because the training set did not contain enough negative samples of shadows. Also, the deep black car was not well classified as the edge feature like HOG was not significant enough like the white car. Therefore, I performed data augmentation to add the shadow and black car samples into the training set. Retraining the model, then I got the above test images, showing good performance. 

![][Example-4]

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used heat map filter for false positives and combining overlapping bounding boxes in code cell 27 and 29. I calculated the heat map over the last 5 frames and set the threshold to 15. The parameters were tuned by experiments. 

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline would fail at the shadows and very deep cars, which in nature results from distribution mismatching between training and test sets. It can be tackled by data augmentation. Tuning the decision function in SVM may also help in seperate these two classes. 

The key to object detection is to find appropriate bounding boxes. There may be some more efficient ways instead of search the window step by step. 

The bounding boxes are not very accurate, especially in high speed situation. Heat map thresholding could be improved by weighting the boxes using some information like confidence coefficient. 

SVM cannot predict multi samples in a batch. When applied in window search in a image, SVM could be very slow. Instead, neural networks can predict in batch mode and exploit the parallel efficency of GPU, which makes the process very quick. 

It was very slow to generate the video. One reason is the SVM. Another reason is that moviepy could only work using one process, even on my server with 12 CPU cores. I tried to implement a multi-process version to generate the video but failed, mainly because the conflict between opencv and multiprocessing libraries (https://github.com/opencv/opencv/issues/5150). 



