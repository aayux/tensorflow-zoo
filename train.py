# -*- coding: utf-8 -*-

""" 
Creates a ResNeXt Model as defined in:
Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.
"""

__author__ = "Aayush Yadav"
__email__ = "aayushyadav96@gmail.com"

import os
import time
import datetime

import pickle as pckl
import tensorflow as tf
import utils.data_utils as utils

from models.resnext import ResNeXt29


print ("Loading dataset ...")
train_file = "./data/train.pckl"

# Preprocess train and test data
if not os.path.exists(train_file):
    print ("Training dump not found. Pickling from CIFAR10 ...")
    cifar10_path = './cifar10'
    utils.process_cifar10(cifar10_path)

# Load train data
x, y = utils.load_cifar10(train_file)

# Split is not cross validated
val_split =  0.2
x_train, y_train, x_val, y_val = utils.split_data(x, y, val_split)
print ("Dataset loaded.")

""" 
Initialise hyperparameters
NOTE:   Number of conv-layers = 3 * n_blocks * number of resnext layers + 2 
        ie., 3 * 3 * 3 + 2 = 29. Also, C = 16 and depth (n_channels), d = 64
        The model is therefore ResNeXt-29 <16, 64d>.
"""


cardinality = 16
n_blocks = 3

img_dim = len(x_train[0])
n_classes = len(y_train[0])
in_channels = 3
out_channels = 64


learning_rate = 0.001
mu_coeff = 0.9

l2_reg_lambda = 0.
jac_reg_alpha = 0.
darc_reg_lambda = 0.

batch_norm = True

batch_size = 64
n_epochs = 100

n_checkpoints = 2
checkpoint_every = 5
validate_every = 5

""" 
Start Training
"""

print ("Starting training ...")

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        resnext = ResNeXt29(img_dim=img_dim, n_classes=n_classes,
                            in_channels=in_channels, out_channels=out_channels,
                            cardinality=cardinality, n_blocks=n_blocks,
                            l2_reg_lambda=l2_reg_lambda)
        
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=mu_coeff, use_nesterov=True)
        grads_and_vars = optimizer.compute_gradients(resnext.loss)             
        
        # Add update ops as a dependency to the train_op
        # Required for moving_mean and moving_variance
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Record the weight distribution into a histogram summary
        weight_summaries = []
        for var in tf.trainable_variables():
            weight_hist_summary = tf.summary.histogram("{}/grad/hist".format(var.name), var)
            weight_summaries.append(weight_hist_summary)
        weight_summaries_merged = tf.summary.merge(weight_summaries)
        
        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", resnext.loss)
        acc_summary = tf.summary.scalar("accuracy", resnext.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, weight_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Validation summaries
        val_summary_op = tf.summary.merge([loss_summary, acc_summary])
        val_summary_dir = os.path.join(out_dir, "summaries", "dev")
        val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=n_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        
        def train_step(x_batch, y_batch):
            feed_dict = {
                resnext.input_x: x_batch,
                resnext.input_y: y_batch,
                resnext.train_flag: True
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, resnext.loss, resnext.accuracy],
                feed_dict)
            
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)
        
        def validation_step(x_batch, y_batch, writer=None):
            feed_dict = {
                resnext.input_x: x_batch,
                resnext.input_y: y_batch,
                resnext.train_flag: False
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, val_summary_op, resnext.loss, resnext.accuracy],
                feed_dict)
            
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
        
        batches = utils.batch_iter(list(zip(x_train, y_train)), batch_size, n_epochs)
        
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            
            if current_step % validate_every == 0:
                print("\nValidation: ")
                validation_step(x_val, y_val, writer=val_summary_writer)
                print ("\n")
            
            if current_step % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))