# Udacity-CarND

This repository includes several projects of Udacity Self-driving Car Nano Degree.  

## Term 1: Computer Vision and Deep Learning

Term 1 mainly focuses on perception of self-driving cars. Perception helps the cars understand the surrounding environment. 

Typically, perception can be fulfilled using several sensors like radar, LIDAR and camera. Radar has low resolution but does well in extreme weather. It is also cheap and most used automative sensor for object detection and  tracking. LIDAR can get extemely accurate depth information and 360 degrees of visibility. Its resolution is much higher than radar. However, LIDAR is expensive.  Human brains use similar sensor technology like camera. It is cheap and has the highest resolution. Therefore, camera can provide huge data, which is fundemental in deep learning. However, it is bad in depth estimation and not good in extreme weather. 

Computer vision, getting data from cameras, is investigated in this term. Traditional methods, as well as deep learning approach, are employed. 

In this term, five projects are covered:

- Lane lines detection
  - Goal: using traditional computer vision methods to find lane lines on the road in normal case and extreme case (e.g. low contrast resulting from shading or low illumination)
  - Main steps:
    - Image enhancement and color space conversion
    - Masking a region of interest
    - Canny edge detection
    - Hough transform
    - Line filtering and smoothing
- Traffic sign classifier
  - Goal: Using neural networks to recognize traffic signs on the road
  - Main steps:
    - Load the dataset
    - Explore, summarize and visualize the dataset
      - Class balancing
      - Correlation checking
      - Data augmentation
    - Design, train and test a model architecture
      - Starting from a LeNet-like convolutional neural network
      - Balancing underfitting and overfitting
        - Network structure modification
        - Hyperparameter tuning
        - Dropout
        - Batch normalization
    - Use the model to make predictions on new images
    - Analyze the softmax probabilities of the new images
- Behavioral cloning
  - Goal: Using end-to-end deep learning to clone good driving behavior (directly predicting steering angles from images)
  - Main steps:
    - Using the simulator to collect data of good driving behavior
    - Data augmentation (especially covering edge cases)
    - Correcting steering angles for left and right cameras
    - Build a convolution neural network in Keras that predicts steering angles from images
      - Starting from Nvidia's solution
      - Structure modification
      - Batch normalization
      - Dropout
      - Hyperparameter tuning
    - Train and validate the model with a training and validation set
    - Test that the model successfully drives around track one without leaving the road
- Advanced lane lines detection
  - Goal: Using traditional computer vision methods to find the lane lines on the road and calculate numerical estimation of lane curvature and vehicle position
  - Main steps:
    - Camera calibration
    - Image correction
    - Using color transforms, gradients, etc., to create a thresholded binary image
    - Applying a perspective transform to rectify binary image ("birds-eye view")
    - Detecting lane pixels and fitting to find the lane boundary
    - Determining the curvature of the lane and vehicle position with respect to center
    - Warping the detected  lane boundaries back onto the original image
    - Outputing visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position
- Vehicle detection and tracking
  - Goal: using combined traditional and machine learning methods to detect and track vehicles in images and videos
  - Main steps:
    - Creating a labeled training set of images
    - Applying a color space transform
    - Performing a Histogram of Oriented Gradients (HOG) feature extraction
    - Appending binned color features and histograms of color
    - Training a classifier 
      - Linear SVM
      - RBF-kernel SVM
    - Implementing a sliding-window technique and using the trained classifer to search for vehicles in images
    - Running the pipeline on a video stream and creating a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles
    - Estimating a bounding box for vehicles detected