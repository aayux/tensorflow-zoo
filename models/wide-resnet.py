# -*- coding: utf-8 -*-

""" 
Creates a Wide-Reset Model as defined in:
Wide Residual Networks (2017).
Zagoruyko S. and Komodakis N.
arXiv preprint arXiv:1605.07146.
"""

__author__ = "Aayush Yadav"
__email__ = "aayushyadav96@gmail.com"

import tflearn

import numpy as np
import tensorflow as tf

class WideRes22(object):
    def __init__(self, img_dim, n_classes, 
                 in_channels, out_channels, 
                 width_mult, n_blocks, 
                 learning_rate, l2_reg_lambda=0.):
        """
        Constructor, Wide-ResNet-22 Model
        
        Args:
            
            img_dim: dimension(s) of input image; square expected
            n_classes: number of classes in label
            in_channels: input channel dimensionality
            out_channels: tuple with output channel dimensionality
            width_mult: multiplier for width/channels, width_mult=1 for Resnets.
            n_blocks: Stack n_blocks bottleneck blocks/modules
            l2_reg_lambda: Regularisation coefficient for weight decay
        """
        
        print ("Initialising placeholders ...")
        self.input_x = tf.placeholder(tf.float32, shape=[None, img_dim, img_dim, in_channels], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, n_classes], name="input_y")
        self.train_flag = tf.placeholder(tf.bool, name="train_flag")
   
        # First "plain" convolution layer
        with tf.name_scope("conv_0"):
            kernel_shape=[3, 3, in_channels, out_channels[0]]
            with tf.variable_scope("conv_0", reuse=None):
                W = tf.get_variable(shape=kernel_shape,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1), name="weight")
            conv = tf.nn.conv2d(self.input_x, W, strides=[1, 1, 1, 1], padding="SAME", name="conv")
            self.out = tf.nn.relu(tf.contrib.layers.batch_norm(conv, decay=0.9, center=True, scale=True, 
                                          is_training=self.train_flag, zero_debias_moving_mean=True), name="relu")
        
        print ("Computing Residual Group 1 ...")
        # Residual Groups       
        self.out = self.res_group(self.out, stride=1, out_channels=out_channels[0] * width_mult, 
                                  n_blocks=n_blocks, group_id=1)
        print ("Computing Residual Group 2 ...")
        self.out = self.res_group(self.out, stride=2, out_channels=out_channels[1] * width_mult, 
                                  n_blocks=n_blocks, group_id=2)
        print ("Computing Residual Group 3 ...")
        self.out = self.res_group(self.out, stride=2, out_channels=out_channels[2] * width_mult, 
                                  n_blocks=n_blocks, group_id=3)
        
        print ("Pooling outputs and calculating scores ...")
        # Pooling layer
        self.out = tflearn.layers.conv.global_avg_pool(self.out, name="global-avg-pooling")
        self.out = tf.contrib.layers.flatten(self.out)
        
        # Final (unnormalized) scores and predictions
        self.scores = tf.layers.dense(self.out, use_bias=False, units=n_classes, name="scores")
        self.predictions = tf.argmax(self.scores, axis=1, name="predictions")

        # Calculate mean cross-entropy loss
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def res_group(self, input_x, stride, out_channels, n_blocks, group_id):
        """
        Residual Group
        
        Args:
            
            input_x: layer input
            stride: kernel stride convolution
            out_channels: output channel dimensionality of the model (shared)
            n_blocks: Stack n_blocks bottleneck blocks/modules
            group_id: ResNet layer identifier
            
        Returns: A group of n_blocks, k-wide residual blocks
        """

        for block_num in range(n_blocks):
            in_dim = int(np.shape(input_x)[-1])
                
            with tf.name_scope("%s-%s_conv_1" % (group_id, block_num + 1)):
                kernel_shape=[3, 3, in_dim, out_channels]
                with tf.variable_scope(("%s-%s_conv_1" % (group_id, block_num + 1)), reuse=None):
                    W = tf.get_variable(shape=kernel_shape,
                                        initializer=tf.truncated_normal_initializer(stddev=0.1), name="weight")
                conv = tf.nn.conv2d(input_x, W, strides=[1, stride, stride, 1], padding="SAME", name="conv")
                out_1 = tf.nn.relu(tf.contrib.layers.batch_norm(conv, decay=0.9, center=True, scale=True, 
                                          is_training=self.train_flag, zero_debias_moving_mean=True), name="relu")
            
            with tf.name_scope("%s-%s_conv_2" % (group_id, block_num + 1)):                
                kernel_shape=[3, 3, out_channels, out_channels]
                with tf.variable_scope(("%s-%s_conv_2" % (group_id, block_num + 1)), reuse=None):
                    W = tf.get_variable(shape=kernel_shape,
                                        initializer=tf.truncated_normal_initializer(stddev=0.1), name="weight")
                out_2 = tf.nn.conv2d(out_1, W, strides=[1, 1, 1, 1], padding="SAME", name="conv")
                            
            if block_num is 0:
                with tf.name_scope("%s-%s_conv_3" % (group_id, block_num + 1)):
                    kernel_shape=[1, 1, in_dim, out_channels]
                    with tf.variable_scope(("%s-%s_conv_3" % (group_id, block_num + 1)), reuse=None):
                        W = tf.get_variable(shape=kernel_shape,
                                            initializer=tf.truncated_normal_initializer(stddev=0.1), name="weight")
                    conv = tf.nn.conv2d(input_x, W, strides=[1, stride, stride, 1], padding="VALID", name="conv")
                    shortcut = tf.add(out_2, conv)
                stride = 1
            else:
                shortcut = tf.add(out_2, input_x)
                
            input_x = tf.nn.relu(tf.contrib.layers.batch_norm(conv, decay=0.9, center=True, scale=True, 
                                          is_training=self.train_flag, zero_debias_moving_mean=True), name="relu")
                
        return input_x
