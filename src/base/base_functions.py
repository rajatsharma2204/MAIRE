import tensorflow as tf
import numpy as np
class base_functions(object):
  slope = 15
  high_const = 0.82
  low_const = 0.02
  def step_function(self, input_var):
    return tf.sigmoid(self.slope * input_var) * 0.2 / 0.5 + 0.6 * (tf.sign(input_var) * 0.5 + 0.5)
  def G(self, x, y):
    return self.step_function(x - y)
  def GE(self, x, y):
    return self.step_function(x - y + self.low_const)
  def A(self, x):
    return self.step_function(tf.reduce_mean(x, 1) - self.high_const)
