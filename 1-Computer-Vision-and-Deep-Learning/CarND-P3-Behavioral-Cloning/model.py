import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers.core import Lambda, Dense, Flatten, Dropout
from keras.layers import Conv2D, Cropping2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import cv2
import csv
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import h5py


# hyperparameters
CAMERA_CORRECTION = 0.2
UPPER_RATIO = 0.8
LOWER_RATIO = 0.1
EPOCHES = 60
BATCH_SIZE = 128
# DECAY = 1e-4

# load driving_log.csv
work_path = './data/'
path = work_path + 'driving_log.csv' # original data
samples = []
index = 0
with open(path) as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
    	if index is not 0:
    		if abs(float(line[3])) > 0.002:
		    	center_path = work_path + 'IMG/' + line[0].split('/')[-1]
		    	left_path = work_path + 'IMG/' + line[1].split('/')[-1]
		    	right_path = work_path + 'IMG/' + line[2].split('/')[-1]
		    	center_angle = float(line[3])
		    	left_angle = float(line[3]) + CAMERA_CORRECTION
		    	right_angle = float(line[3]) - CAMERA_CORRECTION
		    	samples.append([center_path, center_angle])
		    	samples.append([left_path, left_angle])
		    	samples.append([right_path, right_angle])

    	index += 1


# get image shape
source_path = samples[0][0]
image = cv2.imread(source_path)
image_shape = image.shape
print('Shape of raw image: ',str(image_shape))

# split training and validation sets
train_samples, valid_samples =train_test_split(samples, test_size=0.2)

# construct image-loading generator
# Note: cv2 reads the images in BGR mode but drive.py reads in RGB mode
# In Nvidia's paper, the images fed into the network are in YUV mode
def batch_generator(samples, batch_size=32):
    # true batch_size is batch_size * 3 since there are three cameras
    samples_num = len(samples)
    while 1:
        samples = shuffle(samples) # shuffle for each epoch
        # one epoch
        for offset in range(0, samples_num, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            steering = []
            for sample in batch_samples:
                source_path = sample[0]
                image = cv2.imread(source_path) # cv2: BGR mode
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert to RGB mode
                images.append(image)
                steering.append(sample[1])
            
            images = np.array(images)
            steering = np.array(steering)
            yield shuffle(images, steering) # shuffle because of the correlated left-center-right images

# create training and validation generators
train_generator = batch_generator(train_samples, BATCH_SIZE)
valid_generator = batch_generator(valid_samples, BATCH_SIZE)

# build models
model = Sequential()
vertical_crop = (int((1 - UPPER_RATIO) * image_shape[0]),
                 int(LOWER_RATIO * image_shape[0]))
horizontal_crop = (0, 0)
model.add(Cropping2D(cropping=(vertical_crop, horizontal_crop),
                     input_shape=image_shape))
model.add(Lambda(lambda x: x / 127.5 - 1.))
model.add(Conv2D(filters=16,
                 kernel_size=5,
                 strides=(4,4),
                 padding='valid',
                 activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32,
                kernel_size=5,
                strides=(4,4),
                padding='valid',
                activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64,
                 kernel_size=5,
                 strides=(4,4),
                 padding='valid',
                 activation='relu'))
model.add(BatchNormalization())
# model.add(Conv2D(filters=64,
#                  kernel_size=3,
#                  strides=(2,2),
#                  padding='valid',
#                  activation='relu'))
model.add(Flatten())
# model.add(Dense(100,
#                activation='relu'))
model.add(Dense(50,
                activation='relu'))
model.add(BatchNormalization())
# model.add(Dropout(0.5))
model.add(Dense(10,
                activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1,
                activation=None))

optim = keras.optimizers.Adam()
model.compile(loss='mse', optimizer=optim)
print(model.summary())

tb_callback = keras.callbacks.TensorBoard(log_dir='./logs/run3', histogram_freq=0, batch_size=BATCH_SIZE, write_graph=True, write_grads=False, write_images=True)
# model fit
model.fit_generator(train_generator,
                    steps_per_epoch=len(train_samples) / BATCH_SIZE,
                    validation_data=valid_generator,
                    validation_steps=len(valid_samples) / BATCH_SIZE ,
                    nb_epoch=EPOCHES,
                    callbacks=[tb_callback])

model.save('./model.h5')

