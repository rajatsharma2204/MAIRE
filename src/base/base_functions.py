import tensorflow as tf
import numpy as np
class base_functions(object):
  slope = 15
  high_const = 0.82
  low_const = 0.02
  def step_function(self, input_var):
    return tf.sigmoid(self.slope * input_var) * 0.2 / 0.5 + 0.6 * (tf.sign(input_var) * 0.5 + 0.5)
    #-(tf.nn.relu(0.5 - tf.sigmoid(self.slope * input_var)) - 0.5) + 0.25 * tf.sign(input_var) + 0.25
    #tf.sigmoid(self.slope * input_var) * 0.2 / 0.5 + 0.6 * (tf.sign(input_var) * 0.5 + 0.5)
    #-(tf.nn.relu(0.5 - tf.sigmoid(self.slope * input_var)) - 0.5) + 0.25 * tf.sign(input_var) + 0.25
    #tf.sigmoid(self.slope * input_var) * 0.2 / 0.5 + 0.6 * (tf.sign(input_var) * 0.5 + 0.5)
  def G(self, x, y):
    return self.step_function(x - y)
    #-(tf.nn.relu(0.5 - tf.sigmoid(self.slope * (x - y))) - 0.5) + 0.25 * tf.sign(x - y) + 0.25
  def GE(self, x, y):
    return self.step_function(x - y + self.low_const)
    #-(tf.nn.relu(0.5 - tf.sigmoid(self.slope * (x - y + self.low_const))) - 0.5) + 0.25 * tf.sign(x - y + self.low_const) + 0.25
  def A(self, x):
    # For multiple sampled points.
    return self.step_function(tf.reduce_mean(x, 1) - self.high_const)
    #-(tf.nn.relu(0.5 - tf.sigmoid(self.slope * (tf.reduce_mean(x, 1) - self.high_const))) - 0.5) + 0.25 * tf.sign(tf.reduce_mean(x, 1) - self.high_const) + 0.25
