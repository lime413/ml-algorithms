import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from DecisionTree_class import DecisionTree
from Default import get_sample, get_mean_for_classes, get_accuracy_precision_recall

# options for classes
N_classes = 6
N_objects = 200
first_mean = 0.1
different_between_mean = 0.5
standard_deviation = 0.3

# options for tree
N_options_for_tree_step = 100
maxTreeDepth = 100  # include root node

# options for output
metricsRoundingOrder = 6
classes_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']

# example

classes_mean = get_mean_for_classes(N_classes, first_mean, different_between_mean)

N_objects_one_class = int(N_objects / N_classes)
X_train, y_train = get_sample(N_objects, N_classes, classes_mean, standard_deviation)

for i in range(0, N_classes):
    plt.scatter(X_train[:, 0][i * N_objects_one_class:(i + 1) * N_objects_one_class],
                X_train[:, 1][i * N_objects_one_class:(i + 1) * N_objects_one_class], c=classes_colors[i])

DT = DecisionTree(N_classes, N_options_for_tree_step, maxTreeDepth)
DT.fit(X_train, y_train)
predictions = DT.predict(X_train)

accuracy, precision, recall = get_accuracy_precision_recall(y_train, predictions)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)


def draw_rectangles(x_min, y_min, x_max, y_max, node):
    if nodes[node][0] == -1:
        if round(classes[node]) == 1:
            color = 'red'
        elif round(classes[node]) == 2:
            color = 'orange'
        elif round(classes[node]) == 3:
            color = 'yellow'
        elif round(classes[node]) == 4:
            color = 'green'
        elif round(classes[node]) == 5:
            color = 'blue'
        else:
            color = 'purple'
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, color=color, alpha=0.2)
        ax.add_patch(rect)
    else:
        j = nodes[node][0]
        t = nodes[node][1]
        if j == 0:
            draw_rectangles(x_min, y_min, t, y_max, nodes[node][2])
            draw_rectangles(t, y_min, x_max, y_max, nodes[node][3])
        else:
            draw_rectangles(x_min, y_min, x_max, t, nodes[node][2])
            draw_rectangles(x_min, t, x_max, y_max, nodes[node][3])


classes = DT.classes
nodes = DT.nodes
abscissa_min = np.min(X_train[:, 0]) - 0.5
ordinate_min = np.min(X_train[:, 1]) - 0.5
abscissa_max = np.max(X_train[:, 0]) + 0.5
ordinate_max = np.max(X_train[:, 1]) + 0.5

fig, ax = plt.subplots()
draw_rectangles(abscissa_min, ordinate_min, abscissa_max, ordinate_max, 0)

for i in range(0, N_classes):
    plt.scatter(X_train[:, 0][i * N_objects_one_class:(i + 1) * N_objects_one_class],
                X_train[:, 1][i * N_objects_one_class:(i + 1) * N_objects_one_class], c=classes_colors[i])
plt.show()
