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

param_delta = 0.1
param_delta_b = 2
param_niter = 100

def logreg_classify(X, W, b):
    scores = np.dot(X, np.transpose(W)) + np.transpose(b)
    expscores = np.exp(scores - np.max(scores))
    return expscores / np.expand_dims(np.sum(expscores, axis=1), axis=1)

def logreg_train(X,Y_):
  '''
    Argumenti
      X:  podatci, np.array NxD
      Y_: indeksi razreda, np.array NxC

    Povratne vrijednosti
      w, b: parametri logističke regresije
  '''  
  N = Y_.shape[0]
  D = X.shape[1]
  C = Y_.shape[1]
  
  W = np.random.randn(C, D)
  b = np.random.randn(C, 1)
  
  # gradijentni spust (param_niter iteracija)
  for i in range(param_niter):
      
    scores = np.dot(X, np.transpose(W)) + np.transpose(b)  # N x C
    expscores = np.exp(scores - np.max(scores)) # N x C
    
    # nazivnik sofmaksa
    sumexp = np.expand_dims(np.sum(expscores, axis=1), axis=1) # N x 1

    # logaritmirane vjerojatnosti razreda 
    probs = expscores / sumexp  # N x C
    logprobs = np.log(probs)  # N x C

    # gubitak
    loss  = - 1 / N * np.sum(logprobs * Y_)  # scalar
    
    # dijagnostički ispis
    if i % 10 == 0:
      print("iteration {}: loss {}".format(i, loss))

    # derivacije komponenata gubitka po mjerama
    dL_ds = (expscores / sumexp) - Y_    # N x C

    # gradijenti parametara
    grad_W = np.dot(np.transpose(dL_ds), X) / N     # C x D (ili D x C)
    grad_b = (1 / N) * np.expand_dims(np.sum(dL_ds, axis=0), axis=1)    # C x 1 (ili 1 x C)
    
    # poboljšani parametri
    W += -param_delta * grad_W
    b += -param_delta_b * grad_b
   
  return W,b
    

if __name__=="__main__":
    np.random.seed(100)
    # get the training dataset
    X,Y_ = data.sample_gauss(3, 100)
    
    # train the model
    W,b = logreg_train(X, data.class_to_onehot(Y_))
    # evaluate the model on the training dataset
    probs = logreg_classify(X, W,b)
    Y = np.argmax(probs, axis=1)
    
    # report performance
    accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    AP = data.eval_AP(Y_)
    print (accuracy, recall, precision, AP)
    
    # graph the decision surface
    rect=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda x: np.argmax(logreg_classify(x,W,b), axis=1), rect, offset=0.5)
  
    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    plt.show()
    
    