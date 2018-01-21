# -*- coding: utf-8 -*-

""" 
Creates a ResNeXt Model as defined in:
Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.
"""

__author__ = "Aayush Yadav"
__email__ = "aayushyadav96@gmail.com"

import numpy as np
import pickle as pckl
import tensorflow as tf
import utils.data_utils as utils

from models.resnext import ResNeXt29

print ("Intialising test parameters ...")

# Checkpoint directory from training run
load_checkpoint_dir = "./runs/.../checkpoints"

# Load test data
test_file = './data/test.pckl'
x_test, y_test = pckl.load(open(test_file, "rb"))

# Evaluate on all training data
eval_train = False
batch_size = 64

# Misc Parameters
allow_soft_placement = True
log_device_placement = False

# Evaulation
checkpoint_file = tf.train.latest_checkpoint(load_checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=allow_soft_placement,
      log_device_placement=log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = utils.batch_iter(list(x_test), batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

correct_predictions = float(sum(np.argmax(y_test, axis=1) == all_predictions))
print("Total number of test examples: {}".format(len(y_test)))
print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
