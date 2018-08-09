# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[tensorboard]: ./examples/tensorboard.png "Model Visualization"
[center_lane]: ./examples/center_lane.jpg "Center Lane"
[before_recover]: ./examples/before_recover.jpg "Recovery Image (before)"
[in_recover]: ./examples/in_recover.jpg "Recovery Image (in)"
[after_recover]: ./examples/after_recover.jpg "Recovery Image (after)"
[spot]: ./examples/spot.png "Spot"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is listed as follows, which mainly contains three convolutional layers and  three dense layers. The model is adapted from the Nvidia's model in [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316). The total number of parameters is 79,371. 

-------------------------------------------------------------------------------------------------------------

Layer (type)							Output Shape					Param #

--------

cropping2d_1 (Cropping2D)			(None, 113, 320, 3)				0

---------------------------------

lambda_1 (Lambda)					(None, 113, 320, 3)				0
_________________________________________________________________
conv2d_1 (Conv2D) 					(None, 28, 79, 16)					1216
_________________________________________________________________
batch_normalization_1				(Batch (None, 28, 79, 16)			64
_________________________________________________________________
conv2d_2 (Conv2D)					(None, 6, 19, 32)					12832
_________________________________________________________________
batch_normalization_2				(Batch (None, 6, 19, 32)			128
_________________________________________________________________
conv2d_3 (Conv2D)					(None, 1, 4, 64)					51264
_________________________________________________________________
batch_normalization_3				(Batch (None, 1, 4, 64)				256
_________________________________________________________________
flatten_1 (Flatten)						(None, 256)						0
_________________________________________________________________
dense_1 (Dense)						(None, 50)						12850
_________________________________________________________________
batch_normalization_4				(Batch (None, 50)					200
_________________________________________________________________
dense_2 (Dense)						(None, 10)						510

------

batch_normalization_5				(Batch (None, 10)					40

-----

dropout_1 (Dropout)					(None, 10)						0

-------

dense_3 (Dense)						(None, 1)						11

---

Total params: 79,371
Trainable params: 79,027
Non-trainable params: 344

-----



#### 2. Attempts to reduce overfitting in the model

To accelerate the training process, I introduced Batch Normalization layers (code line 97, 103, 109, 120, and 124), which also helped reduce overfitting. Besides, I also tried using Batch Normalization and Dropout (code line 125) together, which reduced overfitting further. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 54). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track, as can be seen in track1.mp4 (track 1) and track2.mp4 (track 2).

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 129).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to draw intuitions from some existing successful models and adapt them to fit my requirement. 

My first step was to use a convolution neural network model similar to the Nvidia's model in "End to End Learning for Self-Driving Cars". I thought this model might be appropriate because hierarchical convolutional layers can draw useful information directly and easily from raw images, and the Nvidia's model has been proven to fit the autonomous driving scenarios. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model. Considering the additional need of accelerating the training process, I first deployed Batch Normalization layers. Batch Normalization can also help reduce overfitting. Then I tried several ways to combine Dropout with Batch Normalization. But it performed not as good as expected. This phenomenon may result from variance shift, as indicated in [Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift](https://arxiv.org/abs/1801.05134). 

The final step was to run the simulator to see how well the car was driving around track one and two. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I checked the raw data and decided to gather my own dataset. For details about how I created the training data, see Section "Creation of the Training Set & Training Process".  

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture is given above. 

Here is a visualization of the architecture in Tensorboard. 

![alt text][tensorboard]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center_lane]

Due to errors and disturbance, the car cannot keep exactly at the center of the lane even if the network has been trained pretty well. It can hardly recover  when the car is disturbed from the center. This is because it has never seen such images in training data. Therefore, recovery data should be added into the training data. One way is using left and right cameras with the steering angle corrected. Another way is gathering more data of recovering from the left and right sides. I tried both of the ways. 

These images show what a recovery looks like:

![alt text][before_recover]
![alt text][in_recover]
![alt text][after_recover]

Then I repeated this process on track two in order to get more data points.

When testing the trained model in track two, I also found an interesting spot. At this spot, the network mis-recognized the previous lower road as the expansion of current road and produce a steering angle near zero. Thus, I gathered more recovery data at this spot. 

![spot][spot]



After the collection process, I had 67,174 number of data points.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 60 as evidenced by experiments. I used an adam optimizer so that manually training the learning rate wasn't necessary.
