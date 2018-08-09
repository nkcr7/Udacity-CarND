# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[class_distributions]: ./plots/class_distributions.jpg "Class Distributions"
[class_variation]: ./plots/class_variation.jpg "Class Variation"
[grayscaled_training_sample]: ./plots/grayscaled_training_sample.jpg "Grayscaled Training Samples"
[training_sample_visualization]: ./plots/training_sample_visualization.jpg "Training Sample Visualization"
[learning_rate]: ./plots/learning_rate.jpg "Learning Rate"
[training_loss]: ./plots/training_loss.jpg "Training Loss"
[valid_loss]: ./plots/valid_loss.jpg "Validation Loss"
[accuracy]: ./plots/accuracy.jpg "Accuracy"
[test1]: ./test_image/test1.jpg "New Test Image 1"
[test2]: ./test_image/test2.jpg "New Test Image 2"
[test3]: ./test_image/test3.jpg "New Test Image 3"
[test4]: ./test_image/test4.jpg "New Test Image 4"
[test5]: ./test_image/test5.jpg "New Test Image 5"
[resized_test_image]: ./plots/resized_test_image.jpg "Resized Test Image"
[grayscaled_test_image]: ./plots/grayscaled_test_image.jpg "Grayscaled Test Image"
[true_label]: ./plots/true_label.jpg "True Label"
[predicted_label]: ./plots/predicted_label.jpg "Predicted Label"



## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799. 
* The size of the validation set is 4410. 
* The size of test set is 12630. 
* The shape of a traffic sign image is (32, 32, 3). 
* The number of unique classes/labels in the data set is 43. 

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribute in different classes (Left: training set; Middle: validation set; Right: test set). The horizontal axis represents the classes and the vertical represents the counts of samples. This plot indicates that the data are unbalanced in different classes, which can harm the learning process. In an extreme case, the network can learn nothing about a class that exists but never occurs. Potential approaches to balance the data include undersampling, oversampling and threshold-moving.    

![][class_distributions]

Then, I plot some training samples by slicing the training set. It indicates high correlation or similarity among sequential samples in training set. In mini-batch gradient descent, the correlation significantly biases the mini-batch gradient as an estimation of batch gradient. Therefore, the training set should be shuffled when used in mini-batch gradient descent. 

![][training_sample_visualization]

The correlation can also be visualized in a plot of class variation versus sample sequence as follows. The plateaus show the same class labels. 

![][class_variation]





### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the colors don't help the classification too much and converting to grayscale reduces the channel from 3 to 1 and reduces the numbers of parameters. 

Here is some examples of traffic sign images after grayscaling.

![alt text][grayscaled_training_sample]

As a last step, I normalized the image data because input normalization helps the learning process. 

As mentioned above, we might need to balance the classes. One way is data augmentation, as a type of oversampling. However, I don't take this step at first. By fine tuning, the network can hit the goal of 93% accuracy on validation set. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

|     Layer      |               Description                |
| :------------: | :--------------------------------------: |
|     Input      | 32x32x1 image (grayscaled and normalized) |
|  Convolution   | 5x5 kernel, 32 filters, 1 stride, valid padding, outputs 28x28x32 |
|      RELU      |                                          |
|  Max pooling   | 2x2 pool size, 2 strides, valid padding,  outputs 14x14x32 |
|  Convolution   | 5x5 kernel, 64 filters, 1 stride, valid padding, outputs 10x10x64 |
|      RELU      |                                          |
|  Max pooling   | 2x2 pool size, 2 strides, valid padding,  outputs 5x5x64 |
|    Flatten     |            Vector size = 1600            |
| Full-connected |                200 units                 |
|      RELU      |                                          |
| Full-connected |                 84 units                 |
|      RELU      |                                          |
|    Dropout     |          Keep probability = 0.5          |
| Full-connected |                 43 units                 |
|    Softmax     |                                          |

Actually, I tried several models. First I tried LeNet model. However, the accuracy on both the training and validation sets were not so high. There was high bias. So I tuned the learning rate, increased the epoch number and made the network larger. Then I got pretty good training accuracy. But there still was a big gap between training and validation accuracy, which meant high variance. So I added a dropout layer as a regularization. Then I get my final model. I also tried batch normalization but got little progress after some tuning. 



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer with learning rate decay. Adam can reduce the oscillation in gradient descent. Learning rate is a fundamental hyperparameter in the learning process and I spent much time to find a solution. The initial learning rate is 0.001, and it decays by 0.9 after every 3000 steps. The batch size is 128. The number of epochs is chosen to be 300. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.9997.
* validation set accuracy of 0.9451. 
* test set accuracy of 0.9344.

The plots of learning rate, training loss, validation loss and accuracy are as follows.

![][learning_rate]

![][training_loss]

![][valid_loss]

![][accuracy]

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?
  * The first architecture was LeNet because it was simple and well-organized. 
* What were some problems with the initial architecture?
  *  The accuracy on both the training and validation sets were not so high. There was high bias and variance and it was underfitting.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  * So I increased the number of filters in each convolutional layer and increased the number of units in fully connected layer, which made the network larger. Underfitting means the network didn't have enough capacity to model the classification problem. 
  * Combined with several tuned parameters, I got pretty good training accuracy. But there still was a big gap between training and validation accuracy, which meant high variance. So I added a dropout layer as a regularization. Then I get my final model. 
  * I also tried batch normalization but got little progress after some tuning. 
* Which parameters were tuned? How were they adjusted and why?
  * I tuned the learning rate, increased the epoch number. Learning rate is fundamental because low learning rate results in slow learning process and high rate results in oscillation or even instability. Increasing epoch number also enables the network learn more steps and reduces the bias. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  * A convolution layer works well because it captures the property of translation invariance in an image. Also,  weights are shared in a feature map, which reduces the total amount of parameters and boostes the learning process. 
  * a dropout layer helps because it reduces the high dependence of output on some specific neurons by randomly drop out some of them. It enhances the capability of generalization.  

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][test1] ![alt text][test2] ![alt text][test3] 
![alt text][test4] ![alt text][test5]



The first image might be difficult to classify because of the noisy background. 

The second image might be difficult to classify because when converted to low resolution to feed into the network, the image is similar to a 'General Caution' sign. 

The third image might be difficult to classify because of the watermark on the sign. 

The fourth image might be difficult to classify because of the noisy background. 

The fifth image might be easy to classify. 

However, above are just some guesses. Actually, the results are different. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

|        Image         |      Prediction      |
| :------------------: | :------------------: |
| Speed limit (30km/h) | Speed limit (30km/h) |
|     Pedestrians      |   General caution    |
|       No entry       |       No entry       |
|         Stop         |    Priority road     |
|   Turn right ahead   | Speed limit (80km/h) |

The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This is terrible compared with the accuracy on the test set. 

To trace back to the error, I made some analysis. In this experiment, I first loaded the downloaded images and resized them to the shape of 32x32x3 to fit the network. The resized test images are shown here

![][resized_test_image]

Then the images were grayscaled

![][grayscaled_test_image]

I also plotted the some examples of true labels in the training set, namely, speed limit (30km/h), pedestrians, no entry, stop, and turn right ahead. 

![][true_label]

As seen, there were something different in the wrong-predicted images. The 'stop' and 'turn right ahead' signs in test images were lengthened in the horizontal direction. Convolutional networks can capture the translation invariance but cannot capture rotation or scaling invariance (actually, little rotation invariance can be tolerated by the maxpooling operation). Since there was little rotation or scaling samples in the training set, the network would fail to classify rotated or scaled images properly. One solution is using data augmentation to add rotated or scaled images into the training set. The 'pedestrians' sign was also  wrongly classified. From the above plot of class distributions, we could infer that this was because the class of predestrians (class id is 27 in signnames.csv) occupies such a small portion in the training set that the networks can hardly learn something useful on this class. Therefore, these minority classes should also be augmented. 

As comparison, the examples of the predicted labels in the training set are 

![][predicted_label]

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

Note that all the probabilities in the tables below are rounded.

For the first image, the model is relatively sure that this is a Speed limit (30km/h) sign (probability of 1.00), and the image does contain a Speed limit (30km/h) sign. The top five soft max probabilities were

| Probability |      Prediction       |
| :---------: | :-------------------: |
|    1.00     | Speed limit (30km/h)  |
|      0      | Wild animals crossing |
|      0      | Speed limit (20km/h)  |
|      0      |      Keep right       |
|      0      |     Slippery Road     |

For the second image,  

| Probability |      Prediction       |
| :---------: | :-------------------: |
|    1.00     |    General caution    |
|      0      | Speed limit (30km/h)  |
|      0      | Speed limit (70km/h)  |
|      0      | Wild animals crossing |
|      0      |       No entry        |

For the third image, 

| Probability |                Prediction                |
| :---------: | :--------------------------------------: |
|    1.00     |                 No entry                 |
|      0      |                No passing                |
|      0      |                   Stop                   |
|      0      | No passing for vehicles over 3.5 metric tons |
|      0      |                Ahead only                |

For the fourth image,

| Probability |      Prediction      |
| :---------: | :------------------: |
|    1.00     |    Priority road     |
|      0      | Speed limit (50km/h) |
|      0      | Speed limit (80km/h) |
|      0      | Roundabout mandatory |
|      0      |        Yield         |

For the fifth image, 

| Probability |              Prediction               |
| :---------: | :-----------------------------------: |
|    0.91     |         Speed limit (80km/h)          |
|    0.09     | Right-of-way at the next intersection |
|      0      |              Ahead only               |
|      0      |             Priority road             |
|      0      |           Turn right ahead            |


