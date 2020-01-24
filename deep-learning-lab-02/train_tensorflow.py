# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 23:17:49 2019

@author: Oshikuru
"""

import time

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import nn_tensorflow

DATA_DIR = 'F:\GitRepos\deep-learning-lab-02/datasets/MNIST/'
SAVE_DIR = "F:\GitRepos\deep-learning-lab-02/source/fer/out_tensorflow/"

config = {}
config['max_epochs'] = 8
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['lr_policy'] = {1:{'lr':1e-1}, 3:{'lr':1e-2}, 5:{'lr':1e-3}, 7:{'lr':1e-4}}

#np.random.seed(100) 
np.random.seed(int(time.time() * 1e6) % 2**31)
dataset = input_data.read_data_sets(DATA_DIR, one_hot=True)
train_x = dataset.train.images
train_x = train_x.reshape([-1, 1, 28, 28])
train_y = dataset.train.labels
valid_x = dataset.validation.images
valid_x = valid_x.reshape([-1, 1, 28, 28])
valid_y = dataset.validation.labels
test_x = dataset.test.images
test_x = test_x.reshape([-1, 1, 28, 28])
test_y = dataset.test.labels
train_mean = train_x.mean()
train_x -= train_mean
valid_x -= train_mean
test_x -= train_mean

tfcnn = nn_tensorflow.TensorflowNN()

tfcnn.train(train_x, train_y, valid_x, valid_y, config)
tfcnn.evaluate("Test", test_x, test_y)

