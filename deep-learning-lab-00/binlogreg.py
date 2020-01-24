# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 18:38:38 2019

@author: Oshikuru
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import data

import pdb
import IPython

param_delta = 0.5
param_niter = 100
param_lambda = 0.01

def binlogreg_classify(X, w, b):
    scores = np.dot(X, w) + b
    return np.exp(scores) / (1 + np.exp(scores))

def binlogreg_train(X,Y_):
  '''
    Argumenti
      X:  podatci, np.array NxD
      Y_: indeksi razreda, np.array Nx1

    Povratne vrijednosti
      w, b: parametri logističke regresije
  '''  
  N = Y_.shape[0]
  D = X.shape[1]
  
  w = np.random.randn(D, 1)
  b = np.random.randn(1,1)
  
  Y__ = np.hsplit(Y_, 2)[1]
  
  # gradijentni spust (param_niter iteracija)
  for i in range(param_niter):
    # klasifikacijske mjere
    scores = np.dot(X, w) + b # N x 1 
    
    # vjerojatnosti razreda c_1
    probs = np.abs((1 / (1 + np.exp(scores))) - Y__) # N x 1

    # gubitak
    loss = - (1 / N) * np.sum(np.log(probs)) + param_lambda * np.linalg.norm(w) # scalar
    
    # dijagnostički ispis
    if i % 10 == 0:
      print("iteration {}: loss {}".format(i, loss))

    # derivacije gubitka po klasifikacijskim mjerama
    dL_dscores = np.exp(scores) / (1 + np.exp(scores)) - Y__  # N x 1
    
    # gradijenti parametara
    grad_w = np.expand_dims((1 / N) * np.sum(dL_dscores * X, axis=0), axis=1) + param_lambda * (1 / (2 * np.linalg.norm(w))) * 2 * w  # D x 1
    grad_b = (1 / N) * np.sum(dL_dscores)  # 1 x 1

    # poboljšani parametri
    w += -param_delta * grad_w
    b += -param_delta * grad_b
   
  return w,b
    

if __name__=="__main__":
    np.random.seed(100)
    # get the training dataset
    X,Y_ = data.sample_gauss(2, 100)
    # train the model
    w,b = binlogreg_train(X, data.class_to_onehot(Y_))
    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w,b)
    Y = probs>0.5
    
    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y[:,-1], Y_)
    AP = data.eval_AP(Y_)
    print (accuracy, recall, precision, AP)
    
    # graph the decision surface
    rect=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda x: binlogreg_classify(x,w,b), rect, offset=0.5)
  
    # graph the data points
    data.graph_data(X, Y_, Y[:,-1], special=[])

    plt.show()
    
    