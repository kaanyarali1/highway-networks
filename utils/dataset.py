import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import linalg as LA
import time
from tensorflow.keras import Model

"""
This function is used for downloading the data. The name of the dataset is passed as argument. 
It returns the training set,test set and some meta information about the dataset. Dataset is shuffled before return operation.
"""
def loadData(name):
	(ds_train, ds_test), ds_info = tfds.load(
	    name,
	    split=['train', 'test'],
	    shuffle_files=True,
	    as_supervised=True,
	    with_info=True,
	)
	return (ds_train, ds_test), ds_info

#This function is used for normalizing the input images by dividing 255 at each pixel value.
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

#This function is used for prefetching the data, seperating the data to minibatches and normalizing the input images.
def prepareTrainDataset(ds_train,ds_info):
	ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	ds_train = ds_train.cache()
	ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
	ds_train = ds_train.batch(128) #mini-batch size 128
	ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
	return ds_train

def prepareTestDataset(ds_test):
	ds_test = ds_test.map(
	    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	ds_test = ds_test.batch(128)
	ds_test = ds_test.cache()
	ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
	return ds_test

#Function for converting tfds type to numpy array for both labels and input images.
def convertDatasetNumpy(ds_train):
	true_categories=[]
	data=[]
	for x, y in ds_train:
  		data.append(x)
  		true_categories.append(y)
	true_categories=np.concatenate([*true_categories],axis=0)
	data=np.concatenate([*data],axis=0)
	return data,true_categories