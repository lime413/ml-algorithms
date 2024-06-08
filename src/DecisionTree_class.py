import numpy as np


class DecisionTree:
    def __init__(self, n_classes, n_options_for_tree_step, max_depth):
        self.index = 0
        self.nodes = {}
        self.ys = {}
        self.classes = {}
        self.n = n_options_for_tree_step
        self.max_depth = max_depth
        self.n_classes = n_classes

    def error_criterion(self, j, t, attributes, answers):
        n = attributes.shape[0]
        attributes_left, attributes_right, answers_left, answers_right = self.split(j, t, attributes, answers)
        return (attributes_left.shape[0] / n) * self.entropy(answers_left) + (
                    attributes_right.shape[0] / n) * self.entropy(answers_right)

    @staticmethod
    def split(j, t, attribute_matrix, answers):
        n = attribute_matrix.shape[0]
        attributes_left = np.ones((1, 2))
        attributes_right = np.ones((1, 2))
        answers_left = np.ones(1)
        answers_right = np.ones(1)
        for i in range(0, n):
            if attribute_matrix[i][j] < t:
                attributes_left = np.vstack([attributes_left, attribute_matrix[i]])
                answers_left = np.hstack([answers_left, answers[i]])
            else:
                attributes_right = np.vstack([attributes_right, attribute_matrix[i]])
                answers_right = np.hstack([answers_right, answers[i]])
        return attributes_left[1:], attributes_right[1:], answers_left[1:], answers_right[1:]

    def entropy(self, answers):
        n_objects = answers.shape[0]
        unique, counts = np.unique(answers, return_counts=True)
        counts = counts[counts > 0] / n_objects
        counts = np.sum(-counts * np.emath.logn(self.n_classes, counts))
        return counts

    def step(self, attributes, answers):
        attr_maxes = np.array([np.max(attributes[:, 0]), np.max(attributes[:, 1])])
        attr_minis = np.array([np.min(attributes[:, 0]), np.min(attributes[:, 1])])
        var0 = np.linspace(attr_minis[0], attr_maxes[0], self.n)
        var1 = np.linspace(attr_minis[1], attr_maxes[1], self.n)
        var0 = var0[1:(var0.shape[0] - 1)]
        var1 = var1[1:(var1.shape[0] - 1)]
        var = np.vstack([var0, var1])
        err_crit_best = self.error_criterion(0, var[0][0], attributes, answers)
        j_best = 0
        t_best = var[0][0]
        for j in range(0, 2):
            for i in range(0, self.n - 2):
                err_crit_current = self.error_criterion(j, var[j][i], attributes, answers)
                if err_crit_current < err_crit_best:
                    t_best = var[j][i]
                    j_best = j
                    err_crit_best = err_crit_current
        return j_best, t_best

    def count_nodes(self, attributes, answers, node, depth):
        # parent_node : j, t, left_children_node, right_children_node
        # children_node = -1 => дальше не идем
        j, t = self.step(attributes, answers)
        self.nodes[node] = np.array([j, t, self.index + 1, self.index + 2])
        self.index += 2
        attributes_left, attributes_right, answers_left, answers_right = self.split(j, t, attributes, answers)
        entropy_left = self.entropy(answers_left)
        entropy_right = self.entropy(answers_right)
        if entropy_left > 0.4 and depth < self.max_depth - 1:
            self.count_nodes(attributes_left, answers_left, self.nodes[node][2], depth + 1)
        else:
            self.nodes[self.nodes[node][2]] = -np.ones(4)
            self.ys[self.nodes[node][2]] = answers_left
        if entropy_right > 0.4 and depth < self.max_depth - 1:
            self.count_nodes(attributes_right, answers_right, self.nodes[node][3], depth + 1)
        else:
            self.nodes[self.nodes[node][3]] = -np.ones(4)
            self.ys[self.nodes[node][3]] = answers_right

    def fit(self, attributes, answers):
        self.index = 0
        self.nodes = {}
        self.ys = {}
        self.classes = {}
        self.count_nodes(attributes, answers, 0, 1)
        for node in self.nodes.keys():
            if self.nodes[node][0] == -1:
                # predict on mean value
                # self.classes[node] = np.mean(self.ys[node])
                # predict on median value
                y_part = list(self.ys[node])
                self.classes[node] = max(set(y_part), key=y_part.count)

    def predict(self, attributes):
        predictions = np.zeros(attributes.shape[0])
        for i in range(0, attributes.shape[0]):
            node = 0
            node_array = self.nodes[node]
            while node_array[2] != -1:
                j = int(node_array[0])
                t = node_array[1]
                if attributes[i][j] < t:
                    node = node_array[2]
                else:
                    node = node_array[3]
                node_array = self.nodes[node]
            predictions[i] = int(self.classes[node])
        return predictions
