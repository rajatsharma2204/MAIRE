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
file_identifier = open("../datasets/adult.data", "r")
line = file_identifier.readline()
dataset = []
labels = []
attribute_names = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital-Status", "Occupation", "Relationship", "Race", "Sex", "Capital-Gain", "Capital-Loss", "Hours-per-week", "Native-Country"]
class_names = ["<=50K", ">50K"]
discrete_choices = [None] * 14
categories_attributes = [1, 3, 5, 6, 7, 8, 9, 13]
education_map = {
    '10th': 'Dropout', '11th': 'Dropout', '12th': 'Dropout', '1st-4th':
    'Dropout', '5th-6th': 'Dropout', '7th-8th': 'Dropout', '9th':
    'Dropout', 'Preschool': 'Dropout', 'HS-grad': 'High School grad',
    'Some-college': 'High School grad', 'Masters': 'Masters',
    'Prof-school': 'Prof-School', 'Assoc-acdm': 'Associates',
    'Assoc-voc': 'Associates',
}
occupation_map = {
    "Adm-clerical": "Admin", "Armed-Forces": "Military",
    "Craft-repair": "Blue-Collar", "Exec-managerial": "White-Collar",
    "Farming-fishing": "Blue-Collar", "Handlers-cleaners":
    "Blue-Collar", "Machine-op-inspct": "Blue-Collar", "Other-service":
    "Service", "Priv-house-serv": "Service", "Prof-specialty":
    "Professional", "Protective-serv": "Other", "Sales":
    "Sales", "Tech-support": "Other", "Transport-moving":
    "Blue-Collar",
}
country_map = {
    'Cambodia': 'SE-Asia', 'Canada': 'British-Commonwealth', 'China':
    'China', 'Columbia': 'South-America', 'Cuba': 'Other',
    'Dominican-Republic': 'Latin-America', 'Ecuador': 'South-America',
    'El-Salvador': 'South-America', 'England': 'British-Commonwealth',
    'France': 'Euro_1', 'Germany': 'Euro_1', 'Greece': 'Euro_2',
    'Guatemala': 'Latin-America', 'Haiti': 'Latin-America',
    'Holand-Netherlands': 'Euro_1', 'Honduras': 'Latin-America',
    'Hong': 'China', 'Hungary': 'Euro_2', 'India':
    'British-Commonwealth', 'Iran': 'Other', 'Ireland':
    'British-Commonwealth', 'Italy': 'Euro_1', 'Jamaica':
    'Latin-America', 'Japan': 'Other', 'Laos': 'SE-Asia', 'Mexico':
    'Latin-America', 'Nicaragua': 'Latin-America',
    'Outlying-US(Guam-USVI-etc)': 'Latin-America', 'Peru':
    'South-America', 'Philippines': 'SE-Asia', 'Poland': 'Euro_2',
    'Portugal': 'Euro_2', 'Puerto-Rico': 'Latin-America', 'Scotland':
    'British-Commonwealth', 'South': 'Euro_2', 'Taiwan': 'China',
    'Thailand': 'SE-Asia', 'Trinadad&Tobago': 'Latin-America',
    'United-States': 'United-States', 'Vietnam': 'SE-Asia'
}
married_map = {
    'Never-married': 'Never-Married', 'Married-AF-spouse': 'Married',
    'Married-civ-spouse': 'Married', 'Married-spouse-absent':
    'Separated', 'Separated': 'Separated', 'Divorced':
    'Separated', 'Widowed': 'Widowed'
}
condense_maps = [None] * 14
condense_maps[3] = education_map
condense_maps[5] = married_map
condense_maps[6] = occupation_map
condense_maps[13] = country_map

condense_map_numbers = [None] * 14

original_categories_list = [["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"], ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"], ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"], ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"], ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"], ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"], ["Female", "Male"], ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]]
for i in range(0, len(categories_attributes)):
  discrete_choices[categories_attributes[i]] = {}
  if condense_maps[categories_attributes[i]] == None:
    for j in range(0, len(original_categories_list[i])):
      discrete_choices[categories_attributes[i]][original_categories_list[i][j]] = j
  else:
    condense_map_numbers[categories_attributes[i]] = {}
    unique_values = list(set(condense_maps[categories_attributes[i]].values()))
    for k in range(0, len(unique_values)):
      condense_map_numbers[categories_attributes[i]][unique_values[k]] = k
    for j in range(0, len(original_categories_list[i])):
      if original_categories_list[i][j] not in condense_maps[categories_attributes[i]]:
        k += 1
        condense_map_numbers[categories_attributes[i]][original_categories_list[i][j]] = k
        condense_maps[categories_attributes[i]][original_categories_list[i][j]] = original_categories_list[i][j]
      discrete_choices[categories_attributes[i]][condense_maps[categories_attributes[i]][original_categories_list[i][j]]] = condense_map_numbers[categories_attributes[i]][condense_maps[categories_attributes[i]][original_categories_list[i][j]]]
categories_list = []
for i in range(0, 14):
  if discrete_choices[i] != None:
    categories_list.append(sorted(discrete_choices[i], key = discrete_choices[i].get))
def transform_orig_to_num(line_split_list):
  data_point = [attribute.strip() for attribute in line_split_list]
  for i in range(0, len(data_point)):
    if condense_maps[i] != None:
      if data_point[i] not in condense_maps[i]:
        return None
      data_point[i] = condense_map_numbers[i][condense_maps[i][data_point[i]]]
    elif discrete_choices[i] == None:
      data_point[i] = int(data_point[i])
    else:
      if data_point[i] not in discrete_choices[i]:
        return None
      data_point[i] = discrete_choices[i][data_point[i]]
  return data_point
while line:
  line_split_list = line.strip().split(',')
  data_point = transform_orig_to_num(line_split_list[:-1])
  if data_point != None:
    if line_split_list[-1].strip() == ">50K":
      dataset.append(data_point)
      labels.append(1)
    else:
      dataset.append(data_point)
      labels.append(0)
  line = file_identifier.readline()
file_identifier.close()
dataset = np.array(dataset)
labels = np.array(labels)

frac = 0.1
def transform_for_base_model(original_points):
  categories_attributes_index = 0
  next_index = 0
  base_classifier_input_dim = 14 - len(categories_attributes)
  for one_attribute_categories in categories_list:
    base_classifier_input_dim += len(one_attribute_categories)
  base_classifier_input_points = np.zeros((original_points.shape[0], base_classifier_input_dim))
  for i in range(0, 14):
    if categories_attributes_index < 14 and i != categories_attributes[categories_attributes_index]:
      base_classifier_input_points[:, next_index] = (original_points[:, i] - np.mean(dataset[:, i])) * 1.0 / np.std(dataset[:, i])
      next_index += 1
    else:
      for j in range(0, original_points.shape[0]):
        base_classifier_input_points[j, next_index + int(original_points[j, i])] = 1
      next_index += len(categories_list[categories_attributes_index])
      categories_attributes_index += 1
  return base_classifier_input_points
def transform_for_explanation(original_points, explain_orig_point_num):
  next_index = 0
  explanations_input_points = np.zeros(original_points.shape)
  for i in range(0, 14):
    if i != categories_attributes[next_index]:
      explanations_input_points[:, i] = (original_points[:, i] - np.min(dataset[:, i])) * 1.0 / (np.max(dataset[:, i]) - np.min(dataset[:, i])) * (1.0 - 2 * frac) + frac
    else:
      for j in range(0, original_points.shape[0]):
        if explain_orig_point_num[0, i] == original_points[j, i]:
          explanations_input_points[j, i] = (2.0 / 3)
        else:
          explanations_input_points[j, i] = (1.0 / 3)
      next_index += 1
  return explanations_input_points
base_classifier_input_data = transform_for_base_model(dataset)
weights_filename = "../saved_models/adult_weights.hdf5"
base_classifier_input_dim = 14 - len(categories_attributes)
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
class adult(explain):
  def sampling(self, num_sampled_points):
    sampled_indices = range(0, int(0.8 * dataset.shape[0]))
    f_values_points = f_values_dataset[sampled_indices]
    return exp_dataset[sampled_indices, :], f_values_points
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
  sampled_points = np.zeros((sampled_points_num, ), dtype = int)
  f_values_dataset = np.round(model.predict(transform_for_base_model(dataset)))[:, 0]
  l_values_list = np.zeros((sampled_points_num, 14))
  u_values_list = np.zeros((sampled_points_num, 14))
  labels_list = np.zeros((sampled_points_num, ))
  points_selected = []
  points_selected_set = set()
  subset_points = set()
  test_dataset = dataset[int(0.8 * base_classifier_input_data.shape[0]):base_classifier_input_data.shape[0], :]
  predicted_labels = f_values_dataset[int(0.8 * base_classifier_input_data.shape[0]):base_classifier_input_data.shape[0]]
  in_range_all_points = np.zeros((sampled_points_num, int(0.8 * base_classifier_input_data.shape[0])))

  i = 0
  while i < sampled_points_num:
    new_sampled_point = np.random.randint(int(0.8 * base_classifier_input_data.shape[0]))
    if new_sampled_point in subset_points:
      continue
    if i != 0:
      if np.any(np.all(np.logical_and(np.greater(dataset[new_sampled_point, :], l_values_list[0:i, :]), np.less(dataset[new_sampled_point, :], u_values_list[0:i, :])), 1)):
        continue
    sampled_points[i] = new_sampled_point
    subset_points.add(new_sampled_point)
    print "Run : " + str(run + 1) + "/" + str(runs) + ", Point Num : " + str(i + 1) + "/" + str(sampled_points_num)
    explain_point_orig_num = np.array([dataset[sampled_points[i], :]])
    exp_dataset = transform_for_explanation(dataset, explain_point_orig_num)[0:int(0.8 * base_classifier_input_data.shape[0]), :]
    f_value_explain_point = np.array([[f_values_dataset[sampled_points[i]]]])
    adult_object = adult(transform_for_explanation(explain_point_orig_num, explain_point_orig_num)[0], int(0.8 * dataset.shape[0]), f_value_explain_point, 20, 100, 0.95, plot_function)
    model.load_weights(weights_filename)
    labels_list[i] = f_value_explain_point[0][0]
    
    # Verbose 0 means nothing would be printed. For any non-zero value, precision and coverage would be printed after those many iterations.
    adult_object.fit_explanation(num_of_iterations, verbose = 1000)
    next_index = 0
    for j in range(0, 14):
      if j != categories_attributes[next_index]:
        l_values_list[i, j] = (max(adult_object.l_vec_values[j], frac - 0.001) - frac) / (1.0 - 2 * frac) * (np.max(dataset[:, j]) - np.min(dataset[:, j])) + np.min(dataset[:, j])
        u_values_list[i, j] = (min(adult_object.u_vec_values[j], (1.0 - frac) + 0.001) - frac) / (1.0 - 2 * frac) * (np.max(dataset[:, j]) - np.min(dataset[:, j])) + np.min(dataset[:, j])
      else:
        if adult_object.l_vec_values[j] > (1.0/3) and adult_object.u_vec_values[j] > (2.0/3):
          l_values_list[i, j] = explain_point_orig_num[0, j] - 0.01
          u_values_list[i, j] = explain_point_orig_num[0, j] + 0.01
        elif adult_object.l_vec_values[j] < (1.0/3) and adult_object.u_vec_values[j] > (2.0/3):
          l_values_list[i, j] = -0.01
          u_values_list[i, j] = len(categories_list[next_index]) + 0.01
        next_index += 1
    in_range_all_points[i, :] = np.all(np.logical_and(np.greater(dataset[0:int(0.8 * base_classifier_input_data.shape[0]), :], l_values_list[i, :]), np.less(dataset[0:int(0.8 * base_classifier_input_data.shape[0]), :], u_values_list[i, :])), 1)
    i += 1
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
      for j in range(sampled_points_num):
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
plt.savefig("../results/Adult_Coverage.pdf", bbox_inches = "tight")
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
plt.savefig("../results/Adult_Precision.pdf", bbox_inches = "tight")
