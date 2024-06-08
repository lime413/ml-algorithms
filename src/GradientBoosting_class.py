from DecisionTree_class import *
from Default import get_f_metrics


class GradientBoosting:
    def __init__(self, n_classes, n_algorithms, n_options_for_tree_step, max_tree_depth):
        self.n_trees_target = n_algorithms
        self.n_trees_current = 0
        self.n_classes = n_classes
        self.n_options_for_tree_step = n_options_for_tree_step
        self.max_tree_depth = max_tree_depth
        self.trees = [DecisionTree(n_classes, n_options_for_tree_step, max_tree_depth)]

    def predict(self, attributes):
        predictions = np.zeros(attributes.shape[0])
        for i in range(0, self.n_trees_current):
            predictions = predictions + (self.trees[i]).predict(attributes)
        return predictions

    def fit(self, attributes, answers):
        self.trees[0].fit(attributes, answers)
        self.n_trees_current = 1
        for i in range(1, self.n_trees_target):
            self.trees.append(DecisionTree(self.n_classes, self.n_options_for_tree_step, self.max_tree_depth))
            grad_function_loss = answers - self.predict(attributes)
            self.trees[i].fit(attributes, grad_function_loss)
            self.n_trees_current = self.n_trees_current + 1

    def fit_get_fmetrics(self, attributes, answers, attributes_test, answers_test):
        self.fit(attributes, answers)
        predictions = self.predict(attributes_test)
        return get_f_metrics(answers_test, predictions, self.n_classes)
