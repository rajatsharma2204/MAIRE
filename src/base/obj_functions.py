from base_functions import base_functions
import tensorflow as tf
import numpy as np
class obj_functions(base_functions):
  def __init__(self, explain_point, f_value_explain_point, lambda_val_1, lambda_val, threshold):
    self.explain_point = explain_point
    self.dimension = explain_point.shape[0]
    self.l_vec = tf.Variable(self.explain_point - np.ones((self.dimension, )) * 0.01)
    self.u_vec = tf.Variable(self.explain_point + np.ones((self.dimension, )) * 0.01)
    #self.l_vec = tf.Variable(np.ones((self.dimension, )) * 0.0)
    #self.u_vec = tf.Variable(np.ones((self.dimension, )))
    self.clip_l = tf.assign(self.l_vec, tf.clip_by_value(self.l_vec, 0.0, 1.0))
    self.clip_u = tf.assign(self.u_vec, tf.clip_by_value(self.u_vec, 0.0, 1.0))
    self.f_value_explain_point_int = f_value_explain_point[0][0]
    self.f_value_explain_point = tf.cast(tf.constant(f_value_explain_point), "float64")
    self.lambda_val = tf.cast(tf.constant(lambda_val), "float64")
    self.lambda_val_1 = tf.cast(tf.constant(lambda_val_1), "float64")
    self.threshold = threshold
    self.threshold_tensor = tf.cast(tf.constant(self.threshold), "float64")
  def h(self, sampled_points):
    return self.A(tf.concat([self.G(sampled_points, self.l_vec), self.GE(self.u_vec, sampled_points)], 1))
  def cov(self, sampled_points):
    return tf.reduce_mean(self.h(sampled_points))
  def prec(self, sampled_points, f_values_sampled_points):
    h_values = self.h(sampled_points)
    h_mul_values = tf.multiply(h_values, 1.0 - tf.square(f_values_sampled_points - self.f_value_explain_point))
    return tf.reduce_mean(h_mul_values) / tf.reduce_mean(h_values)
  def constraint_sum(self):
    return tf.reduce_sum(tf.multiply(self.lambda_val, tf.nn.relu(self.l_vec - self.explain_point))) + tf.reduce_sum(tf.multiply(self.lambda_val, tf.nn.relu(self.explain_point - self.u_vec)))
  def loss(self, sampled_points, f_values_sampled_points):
    analytic_prec = tf.reduce_sum(tf.cast(tf.reduce_all(tf.logical_and(tf.greater(sampled_points, self.l_vec), tf.less(sampled_points, self.u_vec)), 1), "float64") * (1.0 - tf.square(f_values_sampled_points - self.f_value_explain_point))) / tf.reduce_sum(tf.cast(tf.reduce_all(tf.logical_and(tf.greater(sampled_points, self.l_vec), tf.less(sampled_points, self.u_vec)), 1), "float64"))
    #return -1.0 * (tf.pow(ty, 2) * self.cov(sampled_points)) + self.constraint_sum()
    #tf.multiply((0.5 + tf.sign(analytic_prec - self.threshold_tensor) * 0.5), self.cov(sampled_points))
    return -1.0 * self.cov(sampled_points) - 1.0 * (1.0 - tf.cast(tf.is_nan(analytic_prec), "float64")) * tf.multiply(self.lambda_val_1 * (0.5 + tf.sign(- analytic_prec + self.threshold_tensor) * 0.5), self.prec(sampled_points, f_values_sampled_points)) + self.constraint_sum()
