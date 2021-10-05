from __future__ import division
import math
import numpy as np
from six.moves import xrange
import tensorflow as tf
import matplotlib.pyplot as plt


def add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1.0e4):

    static_shape = x.get_shape().as_list()
    num_dims = len(static_shape) - 2
    channels = tf.shape(x)[-1]
    num_timescales = channels // (num_dims * 2)
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    for dim in xrange(num_dims):
        length = tf.shape(x)[dim + 1]
        position = tf.to_float(tf.range(length))
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
                inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        prepad = dim * 2 * num_timescales
        postpad = channels - (dim + 1) * 2 * num_timescales
        signal = tf.pad(signal, [[0, 0], [prepad, postpad]])
        for _ in xrange(1 + dim):
            signal = tf.expand_dims(signal, 0)
        for _ in xrange(num_dims - 1 - dim):
            signal = tf.expand_dims(signal, -2)
        x += signal
    return x


im1 = tf.zeros([1, 100, 50, 512])  # batch x height x width x color
im2 = add_timing_signal_nd(im1)

plt.imshow(tf.Session().run(im2).squeeze(0)[:,:,0])

a=tf.Session().run(im2).squeeze(0)

# plt.imshow(tf.Session().run(x).squeeze(0)[:,:,0])