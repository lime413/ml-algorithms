from KNearestNeighbours_class import *
from Default import get_sample, get_mean_for_classes
import numpy as np
import matplotlib.pyplot as plt

# options for classes
N_classes = 5
N_objects = 250
first_mean = 0.1
different_between_mean = 0.4
standard_deviation = 0.3

# options for algorithm
K_neighbours = np.array([3, 4, 5, 6, 7, 8, 9, 10])

# options for output
metricsRoundingOrder = 6

# example

classes_mean = get_mean_for_classes(N_classes, first_mean, different_between_mean)

N_objects_one_class = int(N_objects / N_classes)
X_train, y_train = get_sample(N_objects, N_classes, classes_mean, standard_deviation)
X_validate, y_validate = get_sample(N_objects, N_classes, classes_mean, standard_deviation)

print("F - average of harmonic mean values for each class\n\nGradient Boosting")
K_neighbours_best = 0
F_metrics_best = 0
F = np.array([])
for i in range(0, K_neighbours.shape[0]):
    KNN = KNearestNeighbours(N_classes)
    F_cur = KNN.fit_k_get_fmetrics(X_train, y_train, X_validate, y_validate, K_neighbours[i])
    F = np.append(F, F_cur)
    print("F with", K_neighbours[i], "neighbours:", np.round(F_cur, metricsRoundingOrder))
    if F_cur > F_metrics_best:
        F_metrics_best = F_cur
        K_neighbours_best = K_neighbours[i]
print("Best F:", np.round(F_metrics_best, metricsRoundingOrder), ", best number of basic algorithms:", K_neighbours_best)
plt.plot(K_neighbours, F)
plt.show()

print("Test sample")
X_test, y_test = get_sample(N_objects, N_classes, classes_mean, standard_deviation)
KNN = KNearestNeighbours(K_neighbours_best)
F_test = KNN.fit_k_get_fmetrics(X_train, y_train, X_validate, y_validate, K_neighbours_best)
print("F with", K_neighbours_best, "neighbours:", F_test)
