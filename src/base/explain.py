from obj_functions import obj_functions
import numpy as np
import copy
import tensorflow as tf
class explain(obj_functions):
  def sampling(self, sampled_points_num):
    print "Sampling Function not defined."
    exit(0)
  def __init__(self, explain_point, sampled_points_num, f_value_explain_point, lambda_val_1, lambda_val, threshold, plot_function = None):
    self.sampled_points_num = sampled_points_num
    self.valid_vars = None
    self.plot_function = plot_function
    super(explain, self).__init__(explain_point, f_value_explain_point, lambda_val_1, lambda_val, threshold)
    self.construct_graph()
  def construct_graph(self):
    self.sampled_points_placeholder = tf.placeholder("float64", shape = (self.sampled_points_num, self.dimension))
    self.sampled_points_f_values_placeholder = tf.placeholder("float64", shape = (self.sampled_points_num, ))
    self.loss_tensor = self.loss(self.sampled_points_placeholder, self.sampled_points_f_values_placeholder)
    self.cov_tensor = self.cov(self.sampled_points_placeholder)
    self.prec_tensor = self.prec(self.sampled_points_placeholder, self.sampled_points_f_values_placeholder)
    gradients = tf.gradients(self.loss_tensor, [self.l_vec, self.u_vec])
    optimizer = tf.train.AdamOptimizer()
    self.train_one_iteration = optimizer.apply_gradients(zip(gradients, [self.l_vec, self.u_vec]))
    self.analytic_cov = tf.reduce_sum(tf.cast(tf.reduce_all(tf.logical_and(tf.greater(self.sampled_points_placeholder, self.l_vec), tf.less(self.sampled_points_placeholder, self.u_vec)), 1), "float64")) / self.sampled_points_num
    self.analytic_prec = tf.reduce_sum(tf.cast(tf.reduce_all(tf.logical_and(tf.greater(self.sampled_points_placeholder, self.l_vec), tf.less(self.sampled_points_placeholder, self.u_vec)), 1), "float64") * (1.0 - tf.square(self.sampled_points_f_values_placeholder - self.f_value_explain_point))) / tf.reduce_sum(tf.cast(tf.reduce_all(tf.logical_and(tf.greater(self.sampled_points_placeholder, self.l_vec), tf.less(self.sampled_points_placeholder, self.u_vec)), 1), "float64"))
    self.session = tf.Session()
    self.session.run(tf.initialize_all_variables())
  def fit_explanation(self, num_of_iterations, verbose = 1):
    session = self.session
    self.max_cov_above_prec_threshold = 0.0
    if verbose != 0:
      print
      tabular_headings = ["Iteration", "Loss", "Cov", "Prec", "Analytic Cov", "Analytic Prec"]
      row_format = "{:<15}" + "{:<22}" * (len(tabular_headings) - 1)
      print row_format.format(*tabular_headings)
    for iteration in range(num_of_iterations):
      self.temp_l_vec_values, self.temp_u_vec_values = session.run([self.l_vec, self.u_vec])
      sampled_points, f_values_sampled_points = self.sampling(self.sampled_points_num)
      loss_value, cov_value, prec_value, _ = session.run([self.loss_tensor, self.cov_tensor, self.prec_tensor, self.train_one_iteration], feed_dict = {self.sampled_points_placeholder : sampled_points, self.sampled_points_f_values_placeholder : f_values_sampled_points})
      session.run([self.clip_l, self.clip_u])
      analytic_cov_value, analytic_prec_value = session.run([self.analytic_cov, self.analytic_prec], feed_dict = {self.sampled_points_placeholder : sampled_points, self.sampled_points_f_values_placeholder : f_values_sampled_points})
      row_values = [iteration + 1, loss_value, cov_value, prec_value, analytic_cov_value, analytic_prec_value]
      if verbose != 0 and iteration % verbose == 0:
        print row_format.format(*row_values)
      if analytic_prec_value > self.threshold or iteration == 0:# and analytic_cov_value > self.max_cov_above_prec_threshold:
        self.l_vec_values, self.u_vec_values = session.run([self.l_vec, self.u_vec])
        self.ana = (analytic_prec_value, iteration, self.l_vec_values, self.u_vec_values)
        self.max_cov_above_prec_threshold = analytic_cov_value
      if plot_function != None:
        self.plot_function(iteration, self.l_vec_values, self.u_vec_values)
  def greedy_select(self, remove_vars, valid_vars = None):
    vars_removed = []
    if valid_vars == None:
      if self.valid_vars == None:
        self.valid_vars = [True] * self.dimension
    else:
      self.valid_vars = copy.deepcopy(valid_vars)
    sampled_points, f_values_sampled_points = self.sampling(self.sampled_points_num)
    temp_l_vec_values = np.copy(self.l_vec_values)
    temp_u_vec_values = np.copy(self.u_vec_values)
    while remove_vars > 0:
      max_cov_var = -1
      max_cov_value = -1.0
      max_prec_reduce_var = -1
      max_prec_reduce_value = -1.0
      for i in range(0, self.dimension):
        if self.valid_vars[i] == True:
          old_l_value_i = temp_l_vec_values[i]
          old_u_value_i = temp_u_vec_values[i]
          temp_l_vec_values[i] = 0.0
          temp_u_vec_values[i] = 1.0
          analytic_cov_value = np.sum(np.all(np.logical_and(np.greater(sampled_points, temp_l_vec_values), np.less(sampled_points, temp_u_vec_values)), 1).astype(float)) / self.sampled_points_num
          analytic_prec_value = np.sum(np.all(np.logical_and(np.greater(sampled_points, temp_l_vec_values), np.less(sampled_points, temp_u_vec_values)), 1).astype(float) * (1.0 - np.square(f_values_sampled_points - self.f_value_explain_point_int))) / np.sum(np.all(np.logical_and(np.greater(sampled_points, temp_l_vec_values), np.less(sampled_points, temp_u_vec_values)), 1).astype(float))
          if analytic_prec_value > self.threshold:
            if analytic_cov_value > max_cov_value:
              max_cov_value = analytic_cov_value
              max_cov_var = i
          else:
            if analytic_prec_value > max_prec_reduce_value:
              max_prec_reduce_value = analytic_prec_value
              max_prec_reduce_var = i
          temp_l_vec_values[i] = old_l_value_i
          temp_u_vec_values[i] = old_u_value_i
      var_to_remove = -1
      if max_cov_var == -1:
        if max_prec_reduce_var == -1:
          return vars_removed
        else:
          var_to_remove = max_prec_reduce_var
      else:
        var_to_remove = max_cov_var
      self.l_vec_values[var_to_remove] = 0.0
      self.u_vec_values[var_to_remove] = 1.0
      temp_l_vec_values[var_to_remove] = 0.0
      temp_u_vec_values[var_to_remove] = 1.0
      self.valid_vars[var_to_remove] = False
      vars_removed.append(var_to_remove)
      remove_vars -= 1
    return vars_removed
