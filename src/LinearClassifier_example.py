from LinearClassifier_class import *
from Default import get_accuracy_precision_recall
import numpy as np
import matplotlib.pyplot as plt

# options for classes
N_objects = 200
N_classes = 2
mean_first_class = 0.5
mean_second_class = 0.9
standard_deviation = 0.3

# example

attributes_first_class = np.random.normal(mean_first_class, standard_deviation, N_objects).reshape(-1, 2)
attributes_second_class = np.random.normal(mean_second_class, standard_deviation, N_objects).reshape(-1, 2)
X_train = np.vstack([attributes_first_class, attributes_second_class])
y_train = np.hstack([np.ones(int(N_objects / 2)), np.zeros(int(N_objects / 2))]).reshape(-1, 1)

# visualization of data
plt.scatter(X_train[:, 0][0:int(N_objects / 2)], X_train[:, 1][0:int(N_objects / 2)], c='green')
plt.scatter(X_train[:, 0][int(N_objects / 2):], X_train[:, 1][int(N_objects / 2):], c='red')

LC = LinearClassifier()
LC.fit(X_train, y_train)
predictions_float = LC.predict(X_train)  # float value from 0 to 1

predictions = predictions_float.round()  # int value, 0 or 1
accuracy, precision, recall = get_accuracy_precision_recall(y_train, predictions)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)

# calculation confidence of prediction
# and split points (above the line / below the line)
X_above = np.empty(shape=(1, 2))
predictions_above = np.empty(1)
X_below = np.empty(shape=(1, 2))
predictions_below = np.empty(1)
predictions_float = np.round(predictions_float, 4)
for i in range(0, N_objects):
    if predictions_float[i] < 0.5:
        predictions_above = np.vstack([predictions_above, 1 - predictions_float[i]])
        X_above = np.vstack([X_above, X_train[i]])
    else:
        predictions_below = np.vstack([predictions_below, predictions_float[i]])
        X_below = np.vstack([X_below, X_train[i]])
X_below = X_below[1:]
predictions_below = predictions_below[1:]
X_above = X_above[1:]
predictions_above = predictions_above[1:]

attr1_max = np.max(X_train[:, 0])
attr1_min = np.min(X_train[:, 0])
attr2_max = np.max(X_train[:, 1])
attr2_min = np.min(X_train[:, 1])

line_first_point = (attr1_min, (-LC.weights[1] * attr1_min - LC.weights[0]) / LC.weights[2])
line_second_point = (attr1_max, (-LC.weights[1] * attr1_max - LC.weights[0]) / LC.weights[2])

plt.figure()
# drawing line
plt.plot([line_first_point[0], line_second_point[0]], [line_first_point[1], line_second_point[1]])

plt.scatter(X_above[:, 0], X_above[:, 1], c=predictions_above, cmap='Reds')
plt.scatter(X_below[:, 0], X_below[:, 1], c=predictions_below, cmap='Greens')
ax = plt.gca()
ax.set_xlim([attr1_min - 0.1, attr1_max + 0.1])
ax.set_ylim([attr2_min - 0.1, attr2_max + 0.1])
plt.show()
