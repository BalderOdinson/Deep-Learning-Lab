# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 21:28:39 2019

@author: Oshikuru
"""

import os
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
import skimage.io

import tensorflow as tf
import tensorflow.contrib.layers as layers


class TensorflowNN:
    def build_model(self, inputs, labels, num_classes):
      weight_decay = 1e-3
      conv1sz = 16
      conv2sz = 32
      fc3sz = 512
      with tf.contrib.framework.arg_scope([layers.convolution2d],
          kernel_size=5, stride=1, padding='SAME', activation_fn=tf.nn.relu,
          weights_initializer=layers.variance_scaling_initializer(),
          weights_regularizer=layers.l2_regularizer(weight_decay)):
        inputs = tf.reshape(inputs, shape=[-1, 28, 28, 1])
    
        net = layers.convolution2d(inputs, conv1sz, scope='conv1')
        net = tf.layers.max_pooling2d(net, 2, 2)
        net = layers.convolution2d(net, conv2sz, scope='conv2')
        net = tf.layers.max_pooling2d(net, 2, 2)
    
      with tf.contrib.framework.arg_scope([layers.fully_connected],
          activation_fn=tf.nn.relu,
          weights_initializer=layers.variance_scaling_initializer(),
          weights_regularizer=layers.l2_regularizer(weight_decay)):
    
        # sada definiramo potpuno povezane slojeve
        # ali najprije prebacimo 4D tenzor u matricu
        net = layers.flatten(inputs)
        net = layers.fully_connected(net, fc3sz, scope='fcs3')
    
      logits = layers.fully_connected(net, num_classes, activation_fn=None, scope='logits')
      loss = tf.losses.softmax_cross_entropy(labels, logits, scope='cost')
    
      return logits, loss
    
    def draw_conv_filters(self, epoch, step, weights, save_dir):
      w = weights.copy()
      num_filters = w.shape[3]
      num_channels = w.shape[2]
      k = w.shape[0]
      assert w.shape[0] == w.shape[1]
      w = w.reshape(k, k, num_channels, num_filters)
      w -= w.min()
      w /= w.max()
      border = 1
      cols = 8
      rows = math.ceil(num_filters / cols)
      width = cols * k + (cols-1) * border
      height = rows * k + (rows-1) * border
      img = np.zeros([height, width, num_channels])
      for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r+k,c:c+k,:] = w[:,:,:,i]
      filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
      ski.io.imsave(os.path.join(save_dir, filename), img)
      
    def draw_image(self, img, mean, std):
      img *= std
      img += mean
      img = img.astype(np.uint8)
      ski.io.imshow(img)
      ski.io.show()
      
    def plot_training_progress(self, save_dir, data):
      fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))
    
      linewidth = 2
      legend_size = 10
      train_color = 'm'
      val_color = 'c'
    
      num_points = len(data['train_loss'])
      x_data = np.linspace(1, num_points, num_points)
      ax1.set_title('Cross-entropy loss')
      ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
               linewidth=linewidth, linestyle='-', label='train')
      ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
               linewidth=linewidth, linestyle='-', label='validation')
      ax1.legend(loc='upper right', fontsize=legend_size)
      ax2.set_title('Average class accuracy')
      ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
               linewidth=linewidth, linestyle='-', label='train')
      ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
               linewidth=linewidth, linestyle='-', label='validation')
      ax2.legend(loc='upper left', fontsize=legend_size)
      ax3.set_title('Learning rate')
      ax3.plot(x_data, data['lr'], marker='o', color=train_color,
               linewidth=linewidth, linestyle='-', label='learning_rate')
      ax3.legend(loc='upper left', fontsize=legend_size)
    
      save_path = os.path.join(save_dir, 'training_plot.pdf')
      print('Plotting in: ', save_path)
      plt.savefig(save_path)
      
    def evaluate(self, name, x, y):
        run_ops = [self.loss_op, self.accuracy_op]
        feed_dict = {self.node_x: x, self.node_y: y}
        loss, acc = self.session.run(run_ops, feed_dict=feed_dict)
        
        print(name + " accuracy = %.2f" % acc)
        print(name + " avg loss = %.2f\n" % loss)
        return loss, acc
      
    def train(self, train_x, train_y, valid_x, valid_y, config):
        lr_policy = config['lr_policy']
        batch_size = config['batch_size']
        num_epochs = config['max_epochs']
        save_dir = config['save_dir']
        num_examples = train_x.shape[0]
        assert num_examples % batch_size == 0
        num_batches = num_examples // batch_size
        
        tf.reset_default_graph()
        self.node_x = tf.placeholder(tf.float32, [None, train_x.shape[1], train_x.shape[2], train_x.shape[3]])
        self.node_y = tf.placeholder(tf.float32, [None, train_y.shape[1]])
        logits, self.loss_op = self.build_model(self.node_x, self.node_y, train_y.shape[1])
        pred_classes = tf.argmax(logits, axis=1)
        labels = tf.argmax(self.node_y, axis=1)
        equality = tf.equal(pred_classes, labels)
        self.accuracy_op = tf.reduce_mean(tf.cast(equality, tf.float32))
        self.session = tf.Session()
        self.session.run(tf.initializers.global_variables())
        
        plot_data = {}
        plot_data['train_loss'] = []
        plot_data['valid_loss'] = []
        plot_data['train_acc'] = []
        plot_data['valid_acc'] = []
        plot_data['lr'] = []
        for epoch_num in range(1, num_epochs + 1):
            if epoch_num in lr_policy:
                solver_config = lr_policy[epoch_num]
                train_op = tf.train.GradientDescentOptimizer(solver_config['lr']).minimize(self.loss_op)
            permutation_idx = np.random.permutation(num_examples)
            train_x = train_x[permutation_idx]
            train_y = train_y[permutation_idx]
            
            for step in range(num_batches):
                offset = step * batch_size 
                # s ovim kodom pazite da je broj primjera djeljiv s batch_size
                batch_x = train_x[offset:(step+1)*batch_size, :]
                batch_y = train_y[offset:(step+1)*batch_size, :]
                feed_dict = {self.node_x: batch_x, self.node_y: batch_y}
                start_time = time.time()
                run_ops = [train_op, self.loss_op, logits]
                ret_val = self.session.run(run_ops, feed_dict=feed_dict)
                _, loss_val, logits_val = ret_val
                duration = time.time() - start_time
                if (step+1) % 50 == 0:
                  sec_per_batch = float(duration)
                  format_str = 'epoch %d, step %d / %d, loss = %.2f (%.3f sec/batch)'
                  print(format_str % (epoch_num, step+1, num_batches, loss_val, sec_per_batch))
                if step % 100 == 0:
                    conv1_var = tf.contrib.framework.get_variables('conv1/weights:0')[0]
                    conv1_weights = conv1_var.eval(session=self.session)
                    self.draw_conv_filters(epoch_num, step, conv1_weights, save_dir)
    
            train_loss, train_acc = self.evaluate('Train', train_x, train_y)
            valid_loss, valid_acc = self.evaluate('Validate', valid_x, valid_y)
            plot_data['train_loss'] += [train_loss]
            plot_data['valid_loss'] += [valid_loss]
            plot_data['train_acc'] += [train_acc]
            plot_data['valid_acc'] += [valid_acc]
            plot_data['lr'] += [solver_config['lr']]
            self.plot_training_progress(save_dir, plot_data)