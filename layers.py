import tensorflow as tf
import math

class deconv(object):
    def __init__(self, height, width, in_size, out_size, stride, scope):
        self.stride = stride
        self.scope = scope
        self.out_size = out_size
        with tf.variable_scope(self.scope):
            input_size = height*width*in_size
            W_init = tf.random_normal_initializer(stddev=0.02)
            self.W = tf.get_variable('W', (height, width, out_size, in_size), initializer=W_init)
            self.b = tf.get_variable('b', (out_size), initializer=tf.constant_initializer(0.0))

    def __call__(self, X):
        with tf.variable_scope(self.scope):
            inshape = X.get_shape().as_list()
            out_shape = [inshape[0], self.stride[1] * inshape[1], self.stride[2] * inshape[2], self.out_size]
            deconv = tf.nn.conv2d_transpose(X, self.W, out_shape, self.stride, padding='SAME')
            deconv = tf.reshape(tf.nn.bias_add(deconv, self.b), deconv.get_shape())
            return deconv

class batch_norm(object):
    def __init__(self, scope, e=1e-5, m=0.9, activation='linear', leak=0.1):
        self.e = e
        self.m = m
        self.scope = scope
        self.activation = activation
        self.leak = leak
    def __call__(self, X, train=True):
        with tf.variable_scope(self.scope):
            out = tf.contrib.layers.batch_norm(X, decay=self.m, updates_collections=None, epsilon=self.e, scale=True, is_training=train, scope=self.scope)
            if self.activation =='linear':
                return out
            elif self.activation == 'lrelu':
                return tf.maximum(out, self.leak*out)
            else:
                return self.activation(out)

class conv(object):
    def __init__(self, height, width, in_size, out_size, stride, scope):
        self.stride = stride
        self.scope = scope
        with tf.variable_scope(self.scope):
            input_size = height*width*in_size
            W_init = tf.random_normal_initializer(stddev=0.02)
            self.W = tf.get_variable('W', (height, width, in_size, out_size), initializer=W_init)
            self.b = tf.get_variable('b', (out_size), initializer=tf.constant_initializer(0.0))

    def __call__(self, X):
        with tf.variable_scope(self.scope):
            conv = tf.nn.conv2d(X, self.W, self.stride, padding='SAME')
            conv = tf.reshape(tf.nn.bias_add(conv, self.b), conv.get_shape())
            return conv

class dense(object):
    def __init__(self, in_size, out_size, scope, activation=tf.nn.relu):
        self.scope = scope
        self.activation = activation
        with tf.variable_scope(self.scope):
            input_size = in_size
            W_init = tf.random_normal_initializer(stddev=0.02)
            self.W = tf.get_variable('W', (in_size, out_size), initializer=W_init)
            self.b = tf.get_variable('b', (out_size), initializer=tf.constant_initializer(0.))

    def __call__(self, X):
        with tf.variable_scope(self.scope):
            if self.activation == 'linear':
                return tf.matmul(X, self.W) + self.b
            else:
                return self.activation(tf.matmul(X, self.W) + self.b)
