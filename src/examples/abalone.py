from base.explain import explain
import numpy as np
import os.path
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from sklearn.utils import class_weight
import tensorflow as tf
def plot_function(iteration, l_vals, u_vals):
  # This function can be used to get results after every iteration.
  return
num_of_iterations = 5000
file_identifier = open("../datasets/abalone.data", "r")
line = file_identifier.readline()
dataset = []
labels = []
attribute_names = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight"]
class_names = ["<=9 Rings", ">9 Rings"]
discrete_choices = [None] * 8
categories_attributes = [1]
categories_list = [["M", "F"]]
def transform_orig_to_num(line_split_list):
  data_point = [attribute.strip() for attribute in line_split_list]
  if data_point[0] == "M":
    data_point[0] = 0
  else:
    data_point[0] = 1
  for i in range(1, len(data_point)):
    data_point[i] = float(data_point[i])
  return data_point
while line:
  line_split_list = line.strip().split(',')
  data_point = transform_orig_to_num(line_split_list[:-1])
  if data_point != None:
    if int(line_split_list[-1].strip()) > 9:
      dataset.append(data_point)
      labels.append(1)
    else:
      dataset.append(data_point)
      labels.append(0)
  line = file_identifier.readline()
file_identifier.close()
dataset = np.array(dataset)
labels = np.array(labels)
print np.sum(labels), labels.shape

frac = 0.1
def transform_for_base_model(original_points):
  base_classifier_input_points = np.zeros((original_points.shape[0], 9))
  for j in range(0, original_points.shape[0]):
    base_classifier_input_points[j, int(original_points[j, 0])] = 1
  for i in range(1, 8):
    base_classifier_input_points[:, i + 1] = (original_points[:, i] - np.mean(dataset[:, i])) * 1.0 / np.std(dataset[:, i])
  return base_classifier_input_points
def transform_for_explanation(original_points, explain_orig_point_num):
  explanations_input_points = np.zeros(original_points.shape)
  for i in range(0, original_points.shape[0]):
    if explain_orig_point_num[0, 0] == original_points[i, 0]:
      explanations_input_points[i, 0] = (2.0 / 3)
    else:
      explanations_input_points[i, 0] = (1.0 / 3)
  for i in range(1, 8):
    explanations_input_points[:, i] = (original_points[:, i] - np.min(dataset[:, i])) * 1.0 / (np.max(dataset[:, i]) - np.min(dataset[:, i])) * (1.0 - 2 * frac) + frac
  return explanations_input_points
base_classifier_input_data = transform_for_base_model(dataset)
weights_filename = "../saved_models/abalone_weights.hdf5"
base_classifier_input_dim = 8 - len(categories_attributes)
for one_attribute_categories in categories_list:
  base_classifier_input_dim += len(one_attribute_categories)
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(labels),
                                                 labels)
model = Sequential()
model.add(Dense(256, activation='tanh', input_dim = base_classifier_input_dim))
model.add(Dropout(0.25))
model.add(Dense(512, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
if os.path.isfile(weights_filename):
  model.load_weights(weights_filename)
else:
  checkpointer = ModelCheckpoint(filepath = weights_filename, verbose = 1, save_best_only = True)
  model.fit(base_classifier_input_data[0:int(0.8 * base_classifier_input_data.shape[0]), :], labels[0:int(0.8 * base_classifier_input_data.shape[0])], shuffle = True, batch_size = 32, validation_split = 0.2, epochs = 100, callbacks = [checkpointer], class_weight = list(class_weights))
class abalone(explain):
  def sampling(self, num_sampled_points):
    sampled_indices = range(0, int(0.8 * dataset.shape[0]))
    f_values_points = f_values_dataset[sampled_indices]
    return exp_dataset, f_values_points
runs = 5
sampled_points_num = 50
size_GExp = 10
coverage_vals = np.zeros((runs, size_GExp))
precision_vals = np.zeros((runs, size_GExp))
coverage_vals_random = np.zeros((runs, size_GExp))
precision_vals_random = np.zeros((runs, size_GExp))
print "Running for " + str(runs) + " runs for plots with M = " + str(sampled_points_num)
run = 0
while run < runs:
  sampled_points = np.random.choice(int(0.8 * base_classifier_input_data.shape[0]), sampled_points_num)
  f_values_dataset = np.round(model.predict(transform_for_base_model(dataset)))[:, 0]
  l_values_list = np.zeros((sampled_points.shape[0], 8))
  u_values_list = np.zeros((sampled_points.shape[0], 8))
  labels_list = np.zeros((sampled_points.shape[0], ))
  points_selected = []
  points_selected_set = set()
  test_dataset = dataset[int(0.8 * base_classifier_input_data.shape[0]):base_classifier_input_data.shape[0], :]
  predicted_labels = f_values_dataset[int(0.8 * base_classifier_input_data.shape[0]):base_classifier_input_data.shape[0]]
  in_range_all_points = np.zeros((sampled_points.shape[0], int(0.8 * base_classifier_input_data.shape[0])))

  for i in range(sampled_points.shape[0]):
    print "Run : " + str(run + 1) + "/" + str(runs) + ", Point Num : " + str(i + 1) + "/" + str(sampled_points_num)
    explain_point_orig_num = np.array([dataset[sampled_points[i], :]])
    exp_dataset = transform_for_explanation(dataset, explain_point_orig_num)[0:int(0.8 * base_classifier_input_data.shape[0]), :]
    f_value_explain_point = np.array([[f_values_dataset[sampled_points[i]]]])
    abalone_object = abalone(transform_for_explanation(explain_point_orig_num, explain_point_orig_num)[0], int(0.8 * base_classifier_input_data.shape[0]), f_value_explain_point, 20, 100, 0.95, plot_function)
    model.load_weights(weights_filename)
    labels_list[i] = f_value_explain_point[0][0]
    
    # Verbose 0 means nothing would be printed. For any non-zero value, precision and coverage would be printed after those many iterations.
    abalone_object.fit_explanation(num_of_iterations, verbose = 1000)
    if abalone_object.l_vec_values[0] > (1.0/3) and abalone_object.u_vec_values[0] > (2.0/3):
      l_values_list[i, 0] = explain_point_orig_num[0, 0] - 0.01
      u_values_list[i, 0] = explain_point_orig_num[0, 0] + 0.01
    elif abalone_object.l_vec_values[0] < (1.0/3) and abalone_object.u_vec_values[0] > (2.0/3):
      l_values_list[i, 0] = -0.01
      u_values_list[i, 0] = 1 + 0.01
    for j in range(1, 8):
      l_values_list[i, j] = (max(abalone_object.l_vec_values[j], frac - 0.001) - frac) / (1.0 - 2 * frac) * (np.max(dataset[:, j]) - np.min(dataset[:, j])) + np.min(dataset[:, j])
      u_values_list[i, j] = (min(abalone_object.u_vec_values[j], (1.0 - frac) + 0.001) - frac) / (1.0 - 2 * frac) * (np.max(dataset[:, j]) - np.min(dataset[:, j])) + np.min(dataset[:, j])
    in_range_all_points[i, :] = np.all(np.logical_and(np.greater(dataset[0:int(0.8 * base_classifier_input_data.shape[0]), :], l_values_list[i, :]), np.less(dataset[0:int(0.8 * base_classifier_input_data.shape[0]), :], u_values_list[i, :])), 1)

  max_cov_exp = -0.1
  max_cov_ind = None
  for i in range(sampled_points_num):
    cov_local_exp = np.mean(np.all(np.logical_and(np.greater(test_dataset, l_values_list[i, :]), np.less(test_dataset, u_values_list[i, :])), 1))
    if cov_local_exp > max_cov_exp:
      max_cov_exp = cov_local_exp
      max_cov_ind = i
  points_selected.append(max_cov_ind)
  points_selected_set.add(points_selected[-1])
  bogus_selection = False
  for i in range(size_GExp):
    if i != 0:
      max_sym_diff = 0
      selection_point_index = -1
      for j in range(sampled_points.shape[0]):
        if j not in points_selected_set:
          sym_diff = 0
          for k in range(len(points_selected)):
            sym_diff += np.sum(np.logical_xor(in_range_all_points[j, :], in_range_all_points[points_selected[k], :]))
          if sym_diff > max_sym_diff:
            max_sym_diff = sym_diff
            selection_point_index = j
      if selection_point_index == -1:
        break
      else:
        points_selected.append(selection_point_index)
        points_selected_set.add(selection_point_index)
    coverage = 0
    precision = 0
    for j in range(test_dataset.shape[0]):
      cov_add = False
      prec_add = False
      min_cov = 1.1
      for k in range(len(points_selected)):
        if np.all(np.logical_and(np.greater(test_dataset[j, :], l_values_list[points_selected[k], :]), np.less(test_dataset[j, :], u_values_list[points_selected[k], :]))):
          cov_add = True
          if min_cov > np.sum(in_range_all_points[points_selected[k], :]) * 1.0 / int(0.8 * dataset.shape[0]):
            min_cov = np.sum(in_range_all_points[points_selected[k], :]) * 1.0 / int(0.8 * dataset.shape[0])
            if labels_list[points_selected[k]] == predicted_labels[j]:
              prec_add = True
            else:
              prec_add = False
      if cov_add:
        coverage += 1
        if prec_add:
          precision += 1
    if coverage != 0.0:
      precision = precision * 1.0 / coverage
      coverage = coverage * 1.0 / test_dataset.shape[0]
      coverage_vals[run, i], precision_vals[run, i] = coverage, precision
    else:
      bogus_selection = True
      break
  bogus_selection_1 = False
  points_selected = []
  for i in range(size_GExp):
    points_selected.append(i)
    coverage = 0
    precision = 0
    for j in range(test_dataset.shape[0]):
      cov_add = False
      prec_add = False
      decisions = []
      for k in range(len(points_selected)):
        if np.all(np.logical_and(np.greater(test_dataset[j, :], l_values_list[points_selected[k], :]), np.less_equal(test_dataset[j, :], u_values_list[points_selected[k], :]))):
          cov_add = True
          decisions.append(labels_list[points_selected[k]] == predicted_labels[j])
      if cov_add:
        coverage += 1
        if np.mean(decisions) > 0.5:
          precision += 1
    if coverage != 0.0:
      precision = precision * 1.0 / coverage
      coverage = coverage * 1.0 / test_dataset.shape[0]
      coverage_vals_random[run, i], precision_vals_random[run, i] = coverage, precision
    else:
      bogus_selection_1 = True
      break
  if not bogus_selection and not bogus_selection_1:
    run += 1
  tf.reset_default_graph()
  K.clear_session()
  model = Sequential()
  model.add(Dense(256, activation='tanh', input_dim = base_classifier_input_dim))
  model.add(Dropout(0.25))
  model.add(Dense(512, activation='tanh'))
  model.add(Dropout(0.5))
  model.add(Dense(1024, activation='tanh'))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
x_vals = range(1, size_GExp + 1)
cov_means = np.mean(coverage_vals, 0)
cov_stds = np.std(coverage_vals, 0)
prec_means = np.mean(precision_vals, 0)
prec_stds = np.std(precision_vals, 0)
cov_rand_means = np.mean(coverage_vals_random, 0)
cov_rand_stds = np.std(coverage_vals_random, 0)
prec_rand_means = np.mean(precision_vals_random, 0)
prec_rand_stds = np.std(precision_vals_random, 0)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
axes = plt.gca()
axes.set_ylim([-0.1, 1.1])
plt.ylabel("Test Coverage", fontsize = 18)
plt.xlabel("Number of Explanations", fontsize = 18)
for tick in axes.xaxis.get_major_ticks():
  tick.label.set_fontsize(18) 
for tick in axes.yaxis.get_major_ticks():
  tick.label.set_fontsize(18) 
(_, caps, _) = plt.errorbar(x_vals, cov_means, cov_stds, marker = "o", color = "blue", capsize = 5)
for cap in caps:
  cap.set_markeredgewidth(1)
(_, caps, _) = plt.errorbar(x_vals, cov_rand_means, cov_rand_stds, marker = "o", color = "red", capsize = 5)
for cap in caps:
  cap.set_markeredgewidth(1)
ax.legend(["MSD Select", "Random Select"], fontsize = 18)
plt.axis("on")
plt.savefig("../results/Abalone_Coverage.pdf", bbox_inches = "tight")
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
axes = plt.gca()
axes.set_ylim([-0.1, 1.1])
plt.ylabel("Test Precision", fontsize = 18)
plt.xlabel("Number of Explanations", fontsize = 18)
for tick in axes.xaxis.get_major_ticks():
  tick.label.set_fontsize(18) 
for tick in axes.yaxis.get_major_ticks():
  tick.label.set_fontsize(18) 
(_, caps, _) = plt.errorbar(x_vals, prec_means, prec_stds, marker = "o", color = "blue", capsize = 5)
for cap in caps:
  cap.set_markeredgewidth(1)
(_, caps, _) = plt.errorbar(x_vals, prec_rand_means, prec_rand_stds, marker = "o", color = "red", capsize = 5)
for cap in caps:
  cap.set_markeredgewidth(1)
ax.legend(["MSD Select", "Random Select"], fontsize = 18)
plt.axis("on")
plt.savefig("../results/Abalone_Precision.pdf", bbox_inches = "tight")
