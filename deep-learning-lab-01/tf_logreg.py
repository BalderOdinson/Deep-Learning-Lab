# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:39:01 2019

@author: Oshikuru
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import data

class TFLogreg:
  def __init__(self, D, C, param_delta=0.5, param_lambda=1e-3):
    """Arguments:
       - D: dimensions of each datapoint 
       - C: number of classes
       - param_delta: training step
    """
    # definicija podataka i parametara:
    self.X = tf.placeholder(tf.float32, [None, D])
    self.Y_ = tf.placeholder(tf.float32, [None, C])
    self.W = tf.Variable(tf.random_normal([D, C], stddev=0.35), tf.float32)
    self.b = tf.Variable(tf.zeros([C]), tf.float32)
    self.param_lambda = tf.constant(param_lambda, tf.float32)

    # formulacija modela: izračunati self.probs
    #   koristiti: tf.matmul, tf.nn.softmax
    self.probs = tf.nn.softmax(tf.matmul(self.X, self.W) + self.b)
    
    # formulacija gubitka: self.loss
    reg_loss = 0.5*self.param_lambda*tf.reduce_sum(self.W*self.W)
    self.loss = tf.reduce_mean(-tf.reduce_sum(self.Y_ * tf.log(self.probs), reduction_indices=1)) + reg_loss

    # formulacija operacije učenja: self.train_step
    self.train_step = tf.train.GradientDescentOptimizer(param_delta).minimize(self.loss)

    # instanciranje izvedbenog konteksta: self.session
    self.session = tf.Session()

  def train(self, X, Yoh_, param_niter):
    """Arguments:
       - X: actual datapoints [NxD]
       - Yoh_: one-hot encoded labels [NxC]
       - param_niter: number of iterations
    """
    # incijalizacija parametara
    #   koristiti: tf.initializers.global_variables 
    self.session.run(tf.initializers.global_variables())
    
    # optimizacijska petlja
    #   koristiti: tf.Session.run
    for i in range(param_niter):
        loss,_ = self.session.run([self.loss, self.train_step], 
                                                feed_dict={self.X: X, self.Y_: Yoh_})
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

  def eval(self, X):
    """Arguments:
       - X: actual datapoints [NxD]
       Returns: predicted class probabilites [NxC]
    """
    return self.session.run(self.probs, 
                 feed_dict={self.X: X})
    
def calc_class(X):
    y = tflr.eval(X)
    return np.argmax(y, axis=1) * np.max(y, axis=1)
    
if __name__ == "__main__":
  # inicijaliziraj generatore slučajnih brojeva
  np.random.seed(100)
  tf.set_random_seed(100)

  # instanciraj podatke X i labele Yoh_
  X,Y_ = data.sample_gmm_2d(6, 2, 10)
  Yoh_ = data.class_to_onehot(Y_)

  # izgradi graf:
  tflr = TFLogreg(X.shape[1], Yoh_.shape[1], 0.06,1)

  # nauči parametre:
  tflr.train(X, Yoh_, 1000)

  # dohvati vjerojatnosti na skupu za učenje
  probs = tflr.eval(X)
  Y = np.argmax(probs, axis=1)

  # ispiši performansu (preciznost i odziv po razredima)
  accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
  AP = data.eval_AP(Y_)
  print (accuracy, recall, precision, AP)

  # iscrtaj rezultate, decizijsku plohu
  rect=(np.min(X, axis=0), np.max(X, axis=0))
  data.graph_surface(calc_class, rect, offset=0.5)
  data.graph_data(X, Y_, Y, special=[])
  plt.show()