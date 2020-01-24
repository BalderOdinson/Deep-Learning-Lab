# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 23:37:32 2019

@author: Oshikuru
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import data

import pdb
import IPython

param_delta = 0.05
param_delta_b = 3
param_niter = 10000
param_lambda = 1e-3

def fcann2_classify(X, W1, b1, W2, b2):
    scores_hiden = np.dot(X, W1.T) + b1.T
    relu_scores = np.maximum(scores_hiden, 0)
    scores = np.dot(relu_scores, W2.T) + b2.T
    expscores = np.exp(scores - np.max(scores))
    return expscores / np.sum(expscores, axis=1, keepdims=True)

def fcann2_train(X,Y_,H):
  '''
    Argumenti
      X:  podatci, np.array NxD
      Y_: indeksi razreda, np.array NxC
      H: dimenzija skrivenog sloja

    Povratne vrijednosti
      w, b: parametri logističke regresije
  '''  
  N = Y_.shape[0]
  D = X.shape[1]
  C = Y_.shape[1]
  
  W1 = np.random.randn(H, D)
  b1 = np.random.randn(H, 1)
  W2 = np.random.randn(C, H)
  b2 = np.random.randn(C, 1)
  
  # gradijentni spust (param_niter iteracija)
  for i in range(param_niter):
      
    scores_hiden = np.dot(X, W1.T) + b1.T  # N x H
    relu_scores = np.maximum(scores_hiden,0) # N x H
    scores = np.dot(relu_scores, W2.T) + b2.T # N x C
    expscores = np.exp(scores - np.max(scores)) # N x C
    
    # nazivnik sofmaksa
    sumexp = np.sum(expscores, axis=1, keepdims=True) # N x 1

    # logaritmirane vjerojatnosti razreda 
    probs = expscores / sumexp  # N x C
    logprobs = np.log(probs)  # N x C

    # gubitak
    reg_loss = 0.5*param_lambda*np.sum(W1*W1) + 0.5*param_lambda*np.sum(W2*W2)
    loss  = - 1 / N * np.sum(logprobs * Y_) + reg_loss # scalar
    
    # dijagnostički ispis
    if i % 10 == 0:
      print("iteration {}: loss {}".format(i, loss))

    # derivacije komponenata gubitka po mjerama izlaznog sloja
    dL_ds2 = (probs - Y_) / N    # N x C

    # gradijenti parametara izlaznog sloja
    grad_W2 = np.dot(dL_ds2.T, relu_scores) + param_lambda * W2   # C x H (ili H x C)
    grad_b2 = np.expand_dims(np.sum(dL_ds2, axis=0), axis=1)    # C x 1 (ili 1 x C)
    
    # gradijenti gubitka obzirom na nelinearni izlaz prvog sloja
    dL_dh = np.dot(dL_ds2, W2) # N x H
    
    # derivacije komponenata gubitka po mjerama skrivenog sloja
    dL_ds1 = dL_dh * (relu_scores > 0)    # N x H    

    # gradijenti parametara skrivenog sloja
    grad_W1 = np.dot(dL_ds1.T, X) + param_lambda * W1  # H x D (ili D x H)
    grad_b1 = np.expand_dims(np.sum(dL_ds1, axis=0), axis=1)    # C x 1 (ili 1 x C)
    
    # poboljšani parametri skrivenog sloja
    W1 += -param_delta * grad_W1
    b1 += -param_delta_b * grad_b1
    
    # poboljšani parametri izlaznog sloja
    W2 += -param_delta * grad_W2
    b2 += -param_delta_b * grad_b2
   
  return W1,b1,W2,b2

def calc_class(X):
    y = fcann2_classify(X,W1,b1,W2,b2)
    return np.argmax(y, axis=1) * np.max(y, axis=1)
    

if __name__=="__main__":
    np.random.seed(100)
    # get the training dataset
    X,Y_ = data.sample_gmm_2d(6, 2, 10)
    
    # train the model
    W1,b1,W2,b2 = fcann2_train(X, data.class_to_onehot(Y_), 6)
    # evaluate the model on the training dataset
    probs = fcann2_classify(X,W1,b1,W2,b2)
    Y = np.argmax(probs, axis=1)
    
    # report performance
    accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    AP = data.eval_AP(Y_)
    print (accuracy, recall, precision, AP)
    
    # graph the decision surface
    rect=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(calc_class, rect, offset=0.5)
  
    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    plt.show()