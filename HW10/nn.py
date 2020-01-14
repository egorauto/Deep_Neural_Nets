# Jupyter Notebook code for visualisation

import os
import math
from datetime import datetime
import sys
import time
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
import random

%load_ext tensorboard
!rm -rf ./logs/ 

input_shape = (2, 0)
output_shape = 2

model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(2, use_bias=False),
        layers.ReLU(),
        layers.Dense(output_shape, use_bias=False),
        layers.ReLU()
    ])

model.summary()

loss = tf.keras.losses.mean_squared_error
optimizer = tf.keras.optimizers.Adam()

# tensorboard writer
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
graph_logdir = 'logs/func/%s' % current_time
graph_writer = tf.summary.create_file_writer(graph_logdir)

@tf.function
def graph_trace_function(x, y):
    logits = model(x, training=True)
    loss_value = loss(y, logits)
    tf.gradients(loss_value, x)
    return loss_value

tf.summary.trace_on(graph=True, profiler=True)
loss_value = graph_trace_function(tf.zeros(input_shape), tf.zeros(output_shape))

with graph_writer.as_default():
    tf.summary.trace_export(
        name="my_func_trace",
        step=0,
        profiler_outdir=graph_logdir)

%tensorboard --logdir logs