# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 19:56:58 2019

@author: Oshikuru
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def train(points):
    ## 1. definicija računskog grafa
    # podatci i parametri
    X  = tf.placeholder(tf.float32, [None])
    Y_ = tf.placeholder(tf.float32, [None])
    a = tf.Variable(0.0)
    b = tf.Variable(0.0)
    N = tf.constant(points.shape[0], tf.float32)
    
    # afini regresijski model
    Y = a * X + b
    
    # kvadratni gubitak
    loss = tf.reduce_sum((Y-Y_)**2) / N
    
    # optimizacijski postupak: gradijentni spust
    trainer = tf.train.GradientDescentOptimizer(0.1)
    
    ## 2. inicijalizacija parametara
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    
    ## 3. učenje
    # neka igre počnu!
    for i in range(5):
      comp_gradients = trainer.compute_gradients(loss, [a,b])
      apl_gradients = trainer.apply_gradients(comp_gradients)
      val_loss, val_a,val_b, val_grad_a, val_grad_b,_ = sess.run([loss, a,b, comp_gradients[0], comp_gradients[1], apl_gradients], 
                                                                 feed_dict={X: points[:,0], Y_: points[:,1]})
      print(i,val_loss,val_grad_a, val_grad_b)
      
def train_manual(points):
    ## 1. definicija računskog grafa
    # podatci i parametri
    X  = tf.placeholder(tf.float32, [None])
    Y_ = tf.placeholder(tf.float32, [None])
    a = tf.Variable(0.0)
    b = tf.Variable(0.0)
    N = tf.constant(points.shape[0], tf.float32)
    delta = tf.constant(0.1, tf.float32)
    
    # afini regresijski model
    Y = a * X + b
    
    # gradijenti
    grad_a = delta * tf.reduce_sum((Y-Y_)*2 * X) / N
    grad_b = delta * tf.reduce_sum((Y-Y_)*2) / N
    
    calc_a = tf.assign_sub(a, grad_a)
    calc_b = tf.assign_sub(b, grad_b)
    
    # kvadratni gubitak
    loss = tf.reduce_sum((Y-Y_)**2) / N
    
    ## 2. inicijalizacija parametara
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    
    # print_op = tf.print(loss, grad_a, a, grad_b, b, output_stream=sys.stdout)
    
    ## 3. učenje
    # neka igre počnu!
    for i in range(5):
      val_loss, val_a,val_b, val_grad_a, val_grad_b,_,_ = sess.run([loss, a,b, grad_a, grad_b, calc_a, calc_b], 
                                                                 feed_dict={X: points[:,0], Y_: points[:,1]})
      print(i,val_loss, val_grad_a, val_a, val_grad_b, val_b)
      
if __name__=="__main__":
    train(np.array([[1,3],[2,5]]))
    train_manual(np.array([[1,3],[2,5]]))