from Default import get_f_metrics
import numpy as np


class KNearestNeighbours:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    @staticmethod
    def distance(x1, x2):
        return np.sqrt((x2[0] - x1[0]) ** 2 + (x2[1] - x1[1]) ** 2)

    def fit_k_get_fmetrics(self, attributes, answers, attributes_test, answers_test, k_neighbours):
        predictions = np.zeros(answers_test.shape[0])
        for i in range(0, attributes_test.shape[0]):
            knn_x = np.zeros([k_neighbours, 2])
            knn_y = np.zeros(k_neighbours)
            for j in range(0, k_neighbours):
                knn_x[j] = attributes_test[i] + 30
            for j in range(0, attributes.shape[0]):
                a = self.distance(attributes_test[i], attributes[j])
                if a < self.distance(knn_x[k_neighbours - 1], attributes_test[i]):
                    k = k_neighbours - 2
                    while a < self.distance(knn_x[k], attributes_test[i]):
                        k -= 1
                        if k == -1:
                            break
                    knn_x = np.vstack([knn_x[0:k + 1], attributes[j], knn_x[k + 2:k_neighbours]])
                    knn_y = np.hstack([knn_y[0:k + 1], answers[j], knn_y[k + 2:k_neighbours]])
            knn_y = np.array([int(i) for i in knn_y])
            predictions[i] = np.bincount(knn_y).argmax()

        return get_f_metrics(answers_test, predictions, self.n_classes)
