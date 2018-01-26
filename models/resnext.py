# -*- coding: utf-8 -*-

""" 
Creates a ResNeXt Model as defined in:
Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.
"""

__author__ = "Aayush Yadav"
__email__ = "aayushyadav96@gmail.com"

import tflearn

import numpy as np
import tensorflow as tf

class ResNeXt29(object):
    def __init__(self, img_dim, n_classes, in_channels, out_channels ,cardinality, n_blocks, l2_reg_lambda):
        """
        Constructor, ResNeXt-29 Model
        
        Args:
            
            img_dim: dimension(s) of input image; square expected
            n_classes: number of classes in label
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            cardinality: number of convolution groups
            n_blocks: Stack n_blocks bottleneck blocks/modules
            l2_reg_lambda: Regularisation coefficient for weight decay
        """
        
        self.input_x = tf.placeholder(tf.float32, shape=[None, img_dim, img_dim, in_channels], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, n_classes], name="input_y")
        self.train_flag = tf.placeholder(tf.bool, name="train_flag")

        # First "plain" convolution layer
        with tf.name_scope("conv_1"):
            kernel_shape=[3, 3, in_channels, out_channels]
            with tf.variable_scope("conv_1", reuse=None):
                W = tf.get_variable(shape=kernel_shape,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1), name="weight")
            conv = tf.nn.conv2d(self.input_x, W, strides=[1, 1, 1, 1], padding="SAME", name="conv")
            b_norm = tf.contrib.layers.batch_norm(conv, decay=0.9, center=True, scale=True, 
                                                  is_training=self.train_flag, zero_debias_moving_mean=True)
            self.out = tf.nn.relu(b_norm, name="relu")
        
        # ResNeXt Layers        
        self.out = self.resnext_layer(self.out, out_dim=64, cardinality=cardinality, 
                                      out_channels=out_channels, n_blocks=n_blocks, layer_id=1)
        self.out = self.resnext_layer(self.out, out_dim=128, cardinality=cardinality, 
                                      out_channels=out_channels, n_blocks=n_blocks, layer_id=2)
        self.out = self.resnext_layer(self.out, out_dim=256, cardinality=cardinality, 
                                      out_channels=out_channels, n_blocks=n_blocks, layer_id=3)
        
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

    def resnext_layer(self, input_x, out_dim, cardinality, out_channels, n_blocks, layer_id):
        """
        RexNeXt bottleneck type B
        
        Args:
            
            input_x: layer input
            out_dim: output channel dimensionality
            cardinality: number of convolution groups
            out_channels: output channel dimensionality of the model (shared)
            n_blocks: Stack n_blocks bottleneck blocks/modules
            layer_id: ResNeXt layer identifier
            
        Returns: Result of a single ResNeXt layer as a Tensor
        """

        for block_num in range(n_blocks):
            input_dim = int(np.shape(input_x)[-1])
            
            if input_dim * 2 == out_dim:
                pad_flag = True
                stride = 2
                channel = input_dim // 2
            else:
                pad_flag = False
                stride = 1
            
            layers_split = []
            for idx in range(cardinality):
                with tf.name_scope("%s-%s_conv-reduce_%s" % (layer_id, block_num + 1, idx + 1)):
                    kernel_shape=[1, 1, input_dim, out_channels]
                    with tf.variable_scope(("%s-%s_conv-reduce_%s" % (layer_id, block_num + 1, idx + 1)), reuse=None):
                        W = tf.get_variable(shape=kernel_shape,
                                            initializer=tf.truncated_normal_initializer(stddev=0.1), name="weight")
                    conv = tf.nn.conv2d(input_x, W, strides=[1, 1, 1, 1], padding="SAME", name="conv")
                    b_norm = tf.contrib.layers.batch_norm(conv, decay=0.9, center=True, scale=True, 
                                              is_training=self.train_flag, zero_debias_moving_mean=True)
                    out_1 = tf.nn.relu(b_norm, name="relu")
                
                with tf.name_scope("%s-%s_conv-conv_%s" % (layer_id, block_num + 1, idx + 1)):
                    kernel_shape=[3, 3, out_channels, out_channels]
                    with tf.variable_scope(("%s-%s_conv-conv_%s" % (layer_id, block_num + 1, idx + 1)), reuse=None):
                        W = tf.get_variable(shape=kernel_shape,
                                            initializer=tf.truncated_normal_initializer(stddev=0.1), name="weight")
                    conv = tf.nn.conv2d(out_1, W, strides=[1, 1, 1, 1], padding="SAME", name="conv")
                    b_norm = tf.contrib.layers.batch_norm(conv, decay=0.9, center=True, scale=True, 
                                              is_training=self.train_flag, zero_debias_moving_mean=True)
                    out_2 = tf.nn.relu(b_norm, name="relu")
                
                layers_split.append(out_2)
            
            split_merge = tf.concat(layers_split, axis=3)
            
            with tf.name_scope("%s-%s_conv-expand" % (layer_id, block_num + 1)):
                kernel_shape=[1, 1, out_channels * cardinality, out_dim]
                with tf.variable_scope(("%s-%s_conv-expand" % (layer_id, block_num + 1)), reuse=None):
                    W = tf.get_variable(shape=kernel_shape,
                                        initializer=tf.truncated_normal_initializer(stddev=0.1), name="weight")
                conv = tf.nn.conv2d(split_merge, W, strides=[1, 1, 1, 1], padding="SAME", name="conv")
                out_3 = tf.contrib.layers.batch_norm(conv, decay=0.9, center=True, scale=True, 
                                          is_training=self.train_flag, zero_debias_moving_mean=True)
                
            if pad_flag:
                padded_x = tf.layers.average_pooling2d(input_x, pool_size=[2,2], strides=1, padding="SAME")
                padded_x = tf.pad(padded_x, [[0, 0], [0, 0], [0, 0], [channel, channel]])
            else :
                padded_x = input_x
                
            input_x = tf.nn.relu(tf.add(out_3, padded_x), name="relu")
        
        return input_x
