import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops


def _variable_with_weight_decay(name, shape, wd=1e-3):
    with tf.device("/cpu:0"): # store all weights in CPU to optimize weights sharing among GPUs
        var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def max_pool3d(input_, k, name='max_pool3d'):
  return tf.nn.max_pool3d(input_, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)


def conv2d(input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        w = _variable_with_weight_decay('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
        b = _variable_with_weight_decay('b', [output_dim])
        
        return tf.nn.bias_add(conv, b)


def cross_conv2d(input_, kernel, d_h=1, d_w=1, padding='SAME', name="cross_conv2d"):
    with tf.variable_scope(name):
        output_dim = kernel.get_shape()[4]
        batch_size = input_.get_shape().as_list()[0]
        b = _variable_with_weight_decay('b', [output_dim])
       
        output = []
        input_list  = tf.unstack(input_)
        kernel_list = tf.unstack(kernel)
        for i in range(batch_size):
            conv = tf.nn.conv2d(tf.expand_dims(input_list[i],0), kernel_list[i], strides=[1, d_h, d_w, 1], padding=padding)
            conv = tf.nn.bias_add(conv, b)
            output.append(conv)

    return tf.concat(output, 0)


def conv3d(input_, output_dim, k_t=3, k_h=3, k_w=3, d_t=1, d_h=1, d_w=1, padding='SAME', name="conv3d"):
    with tf.variable_scope(name):
        w = _variable_with_weight_decay('w', [k_t, k_h, k_w, input_.get_shape()[-1], output_dim])
        conv = tf.nn.conv3d(input_, w, strides=[1, d_t, d_h, d_w, 1], padding=padding)
        b = _variable_with_weight_decay('b', [output_dim])
        
        return tf.nn.bias_add(conv, b)


def relu(x):
    return tf.nn.relu(x)


def fc(input_, output_dim, name='fc'):
    with tf.variable_scope(name):
        w = _variable_with_weight_decay('w', [input_.get_shape()[-1], output_dim])
        b = _variable_with_weight_decay('b', [output_dim])
        
        return tf.matmul(input_, w) + b


def deconv2d(input_, output_shape, k_h=3, k_w=3, d_h=1, d_w=1, padding='SAME', name="deconv2d"):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = _variable_with_weight_decay('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]])
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1], padding=padding)
        b = _variable_with_weight_decay('b', [output_shape[-1]])

        return tf.nn.bias_add(deconv, b)


def deconv3d(input_, output_shape, k_t=3, k_h=3, k_w=3, d_t=1, d_h=1, d_w=1, padding='SAME', name="deconv3d"):
    with tf.variable_scope(name):
        # filter : [depth, height, width, output_channels, in_channels]
        w = _variable_with_weight_decay('w', [k_t, k_h, k_h, output_shape[-1], input_.get_shape()[-1]])
        deconv = tf.nn.conv3d_transpose(input_, w, output_shape=output_shape, strides=[1, d_t, d_h, d_w, 1], padding=padding)
        b = _variable_with_weight_decay('b', [output_shape[-1]])

        return tf.nn.bias_add(deconv, b)