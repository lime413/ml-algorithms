import numpy as np

from GradientBoosting_class import GradientBoosting
from RandomForest_class import RandomForest
from LinearClassifier_class import LinearClassifier
from DecisionTree_class import DecisionTree
from Default import get_sample, get_mean_for_classes, get_f_metrics

# options for classes
N_classes = 10
N_objects = 1000
first_mean = 0.1
different_between_mean = 0.5
standard_deviation = 0.3

# options for trees
N_options_for_tree_step = 100
maxTreeDepth = 1000  # include root node

# options for random forest
N_algorithms_forest = 3
maxTreeDepth_forest = 3
forest_proportion = 0.7

# options for gradient boosting
N_algorithms_boosting = 5
maxTreeDepth_boosting = 3

# only for output
metricsRoundingOrder = 6

# generate data
classes_mean = get_mean_for_classes(N_classes, first_mean, different_between_mean)

N_objects_one_class = int(N_objects / N_classes)
X_train, y_train = get_sample(N_objects, N_classes, classes_mean, standard_deviation)
X_test, y_test = get_sample(N_objects, N_classes, classes_mean, standard_deviation)

# comparison
print("F - average of harmonic mean values for each class\n")

print("\n10 classes")

print("Decision Tree without depth limitation")
DT = DecisionTree(N_classes, N_options_for_tree_step, maxTreeDepth)
DT.fit(X_train, y_train)
F_metrics = get_f_metrics(DT.predict(X_test), y_test, N_classes)
print("F = ", np.round(F_metrics, metricsRoundingOrder))

print("Random Forest with", N_algorithms_forest, "basic algorithms (max tree depth -", maxTreeDepth_forest, ")")
RF = RandomForest(forest_proportion, N_classes, N_options_for_tree_step, maxTreeDepth)
F_metrics = RF.fit_n_get_fmetrics(X_train, y_train, X_test, y_test, N_algorithms_forest)
print("F = ", np.round(F_metrics, metricsRoundingOrder))

print("Gradient Boosting with", N_algorithms_boosting, "basic algorithms (max tree depth -", maxTreeDepth_boosting, ")")
GB = GradientBoosting(N_classes, N_algorithms_boosting, N_options_for_tree_step, maxTreeDepth)
F_metrics = GB.fit_get_fmetrics(X_train, y_train, X_test, y_test)
print("F = ", np.round(F_metrics, metricsRoundingOrder))

N_objects = 200
mean_1 = 0.5
mean_2 = 0.9
N_classes = 2
x1 = np.random.normal(mean_1, standard_deviation, N_objects).reshape(-1, 2)
x2 = np.random.normal(mean_2, standard_deviation, N_objects).reshape(-1, 2)
X_train = np.vstack([x1,x2])
y_train = np.hstack([np.ones(int(N_objects / 2)), np.zeros(int(N_objects / 2))]).reshape(-1, 1)
x1 = np.random.normal(mean_1, standard_deviation, N_objects).reshape(-1, 2)
x2 = np.random.normal(mean_2, standard_deviation, N_objects).reshape(-1, 2)
X_test = np.vstack([x1,x2])
y_test = np.hstack([np.ones(int(N_objects / 2)), np.zeros(int(N_objects / 2))]).reshape(-1, 1)

print("\n\n2 classes")

print("Linear Classifier")
LC = LinearClassifier()
LC.fit(X_train, y_train)
predictions = np.round(LC.predict(X_train))
F_metrics = get_f_metrics(y_test, predictions, N_classes)
print("F = ", np.round(F_metrics, metricsRoundingOrder))

print("Decision Tree without depth limitation")
DT = DecisionTree(N_classes, N_options_for_tree_step, maxTreeDepth)
DT.fit(X_train, y_train)
F_metrics = get_f_metrics(DT.predict(X_test), y_test, N_classes)
print("F = ", np.round(F_metrics, metricsRoundingOrder))

print("Random Forest with ", N_algorithms_boosting, " basic algorithms")
RF = RandomForest(forest_proportion, N_classes, N_options_for_tree_step, maxTreeDepth)
F_metrics = RF.fit_n_get_fmetrics(X_train, y_train, X_test, y_test, N_algorithms_forest)
print("F = ", np.round(F_metrics, metricsRoundingOrder))

print("Gradient Boosting with", N_algorithms_boosting, "basic algorithms (max tree depth - )", maxTreeDepth)
GB = GradientBoosting(N_classes, N_algorithms_boosting, N_options_for_tree_step, maxTreeDepth)
F_metrics = GB.fit_get_fmetrics(X_train, y_train, X_test, y_test)
print("F = ", np.round(F_metrics, metricsRoundingOrder))