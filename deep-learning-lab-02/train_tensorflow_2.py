# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 01:21:03 2019

@author: Oshikuru
"""

import pickle
import os
import tensorflow as tf

import numpy as np

import nn_tensorflow_2

def shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y

def unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='latin1')
  fo.close()
  return dict

def class_to_onehot(Y):
  Yoh=np.zeros((len(Y),max(Y)+1))
  Yoh[range(len(Y)),Y] = 1
  return Yoh

DATA_DIR = 'F:\GitRepos\deep-learning-lab-02/datasets/CIFAR/'
SAVE_DIR = "F:\GitRepos\deep-learning-lab-02/source/fer/out_tensorflow_2/"

config = {}
config['max_epochs'] = 30
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['lr_policy'] = {1:{'lr':1e-2}, 9:{'lr':8e-3}, 15:{'lr':6e-3}, 21:{'lr':4e-3}, 27:{'lr':2e-3}}

img_height = 32
img_width = 32
num_channels = 3
train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
train_y = []
for i in range(1, 6):
  subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
  train_x = np.vstack((train_x, subset['data']))
  train_y += subset['labels']
train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0,2,3,1)
train_y = np.array(train_y, dtype=np.int32)

subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0,2,3,1).astype(np.float32)
test_y = np.array(subset['labels'], dtype=np.int32)

valid_size = 5000
train_x, train_y = shuffle_data(train_x, train_y)
valid_x = train_x[:valid_size, ...]
valid_y = train_y[:valid_size, ...]
train_x = train_x[valid_size:, ...]
train_y = train_y[valid_size:, ...]
data_mean = train_x.mean((0,1,2))
data_std = train_x.std((0,1,2))

train_x = (train_x - data_mean) / data_std
valid_x = (valid_x - data_mean) / data_std
test_x = (test_x - data_mean) / data_std

train_y = class_to_onehot(train_y)
valid_y = class_to_onehot(valid_y)
test_y = class_to_onehot(test_y)

tfcnn = nn_tensorflow_2.TensorflowNN()

tfcnn.train(train_x, train_y, valid_x, valid_y, config)
tfcnn.evaluate("Test", test_x, test_y)