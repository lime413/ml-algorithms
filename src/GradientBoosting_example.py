import matplotlib.pyplot as plt
from GradientBoosting_class import *
from Default import get_sample, get_mean_for_classes

# options for classes
N_classes = 10
N_objects = 1000
first_mean = 0.1
different_between_mean = 0.5
standard_deviation = 0.3

# options for trees
N_options_for_tree_step = 100
maxTreeDepth = 3  # include root node

# options for gradient boosting
N_algorithms = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])

# only for output
metricsRoundingOrder = 6

# example

classes_mean = get_mean_for_classes(N_classes, first_mean, different_between_mean)

X_train, y_train = get_sample(N_objects, N_classes, classes_mean, standard_deviation)
X_validate, y_validate = get_sample(N_objects, N_classes, classes_mean, standard_deviation)

print("F - average of harmonic mean values for each class\n\nGradient Boosting")
N_algorithms_best = 0
F_metrics_best = 0
F = np.array([])
for i in range(0, N_algorithms.shape[0]):
    GB = GradientBoosting(N_classes, N_algorithms[i], N_options_for_tree_step, maxTreeDepth)
    F_cur = GB.fit_get_fmetrics(X_train, y_train, X_validate, y_validate)
    F = np.append(F, F_cur)
    print("F with", N_algorithms[i], "basic algorithms:", np.round(F_cur, metricsRoundingOrder))
    if F_cur > F_metrics_best:
        F_metrics_best = F_cur
        N_algorithms_best = N_algorithms[i]
print("Best F:", np.round(F_metrics_best, metricsRoundingOrder), ", best number of basic algorithms:", N_algorithms_best)

plt.plot(N_algorithms, F)
plt.show()
