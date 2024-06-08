from RandomForest_class import *
from Default import get_sample, get_mean_for_classes
import matplotlib.pyplot as plt

# options for classes
N_classes = 10
N_objects = 500
first_mean = 0.1
different_between_mean = 0.5
standard_deviation = 0.3

# options for random forest
N_options_for_tree_step = 100
maxTreeDepth = 4  # include root node
forest_proportion = 0.7
N_algorithms = np.array([3, 4, 5, 10, 20, 50, 70, 100])

metricsRoundingOrder = 6  # only for output

# example

classes_mean = get_mean_for_classes(N_classes, first_mean, different_between_mean)

X_train, y_train = get_sample(N_objects, N_classes, classes_mean, standard_deviation)
X_validate, y_validate = get_sample(N_objects, N_classes, classes_mean, standard_deviation)

print("F - average of harmonic mean values for each class\n\nRandom Forest")
random_forest = RandomForest(forest_proportion, N_classes, N_options_for_tree_step, maxTreeDepth)
n_best_forest = 0
F_best_forest = 0
F = np.array([])
for i in range(0, N_algorithms.shape[0]):
    F_cur = random_forest.fit_n_get_fmetrics(X_train, y_train, X_validate, y_validate, N_algorithms[i])
    F = np.append(F, F_cur)
    print("F with", N_algorithms[i], "basic algorithms:", np.round(F_cur, metricsRoundingOrder))
    if F_cur > F_best_forest:
        F_best_forest = F_cur
        n_best_forest = N_algorithms[i]
print("Best F:", np.round(F_best_forest, metricsRoundingOrder), ", best number of basic algorithms:", n_best_forest)
plt.plot(N_algorithms, F)
plt.show()
