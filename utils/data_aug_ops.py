# -*- coding: utf-8 -*-

__author__ = "Aayush Yadav"
__email__ = "aayushyadav96@gmail.com"

import tensorflow as tf

import numpy as np
import tensorflow as tf

class DataAugmentation(object):
    """
    Class containing CIFAR10 data augmentation operations
    """
    
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def standardise(self, features):
        features = []
        
        tf.reset_default_graph()
		input_x = tf.placeholder(tf.float32, shape=[32, 32, 3])
        std = tf.image.per_image_standardization(input_x)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for img in self.features:
                std_img = sess.run([std], feed_dict={input_x: img})
                features.extend(std_img)
            features = np.array(features, dtype=np.float32)
        self.features = features
        return self.features
    
    def flip_and_crop(self):
        features = []
            
        tf.reset_default_graph()
        input_x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        input_x = tf.reshape(input_x, shape=[32, 32, 3])
        paddings = tf.constant([[4, 4], [4, 4], [0, 0]])

        rand_crop = tf.random_crop(input_x, size=[28, 28, 3])
        padded = tf.pad(rand_crop, paddings, "REFLECT")
        flip_lr = tf.image.flip_left_right(padded)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for img in self.features:
                if np.random.rand() < 0.5:
                    aug_img = sess.run([flip_lr], feed_dict = {input_x: img})
                else:
                    aug_img = sess.run([padded], feed_dict = {input_x: img})
                features.extend(aug_img)
            features = np.array(features, dtype=np.float32)
        
        return features, self.labels
    
    def label_switch(self):
        labels = []

        for label in self.labels:
            if np.random.rand() < 0.5:
                label = np.random.randint(0, 10)
            labels.append(label)
        labels = np.array(labels, dtype=np.int32)
        
        return self.features, labels
    
    def add_gauss_noise(self, noise_factor=0.5):
        features = []
        noise = []
        
        tf.reset_default_graph()
        
        input_x = tf.placeholder(tf.float32, shape=[32, 32, 3])
        
        mean, var = tf.nn.moments(input_x, axes=[0, 1])
        for idx in range(3):
            noise.append(tf.random_normal(mean=mean[0], stddev=tf.sqrt(var[0]), shape=[32, 32]))
        gauss_out = tf.reshape(tf.stack(noise), shape=[32, 32, 3])
        add_gauss = tf.add(input_x, tf.multiply(gauss_out, noise_factor))
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for img in self.features:
                aug_img = sess.run(add_gauss, feed_dict = {input_x: img})
                features.append(aug_img)
            features = np.array(features, dtype=np.float32)
            
        return features, self.labels
    
    def random_gaussian(self):
        features = []
        noise = []
        
        tf.reset_default_graph()
        
        input_x = tf.placeholder(tf.float32, shape=[32, 32, 3])
        
        mean, var = tf.nn.moments(input_x, axes=[0, 1])
        for idx in range(3):
            noise.append(tf.random_normal(mean=mean[0], stddev=tf.sqrt(var[0]), shape=[32, 32]))
        gauss_out = tf.reshape(tf.stack(noise), shape=[32, 32, 3])
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for img in self.features:
                gauss_img = sess.run(gauss_out, feed_dict = {input_x: img})
                features.extend(gauss_img)
            features = np.array(features, dtype=np.float32)
            
        return features, self.labels
