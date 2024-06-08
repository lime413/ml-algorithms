import numpy as np
from DecisionTree_class import DecisionTree
import random
from datetime import datetime
from Default import get_f_metrics
from multiprocessing.dummy import Pool as ThreadPool


class RandomForest:
    def __init__(self, proportion, n_classes, n_options_for_tree_step, max_tree_depth, max_workers=12):
        self.proportion = proportion
        self.n_options = n_options_for_tree_step
        self.max_tree_depth = max_tree_depth
        self.n_classes = n_classes
        self.max_workers = max_workers

    def random_tree(self, data):
        decision_tree = DecisionTree(self.n_classes, self.n_options, self.max_tree_depth)
        attr_ans, attributes_test, n_proportion = data
        random.seed(datetime.now().timestamp())
        random.shuffle(attr_ans)
        attributes, answers = [list(t) for t in zip(*(attr_ans[0:n_proportion]))]
        attributes = np.asarray(attributes)
        answers = np.asarray(answers)
        decision_tree.fit(attributes, answers)
        predictions = decision_tree.predict(attributes_test)
        return predictions

    def fit_n_get_fmetrics(self, attributes, answers, attributes_test, answers_test, n_algorithms):
        attr_ans = list(zip(attributes, answers))
        n_proportion = int(attributes.shape[0] * self.proportion)

        pool = ThreadPool(self.max_workers)
        results = pool.map(self.random_tree, [(attr_ans, attributes_test, n_proportion)] * n_algorithms)
        pool.close()
        pool.join()

        results = np.asarray(results).T
        results = np.array([np.array([int(j) for j in i]) for i in results])

        predictions = np.array([])
        for i in range(0, results.shape[0]):
            values, counts = np.unique(results[i], return_counts=True)
            predictions = np.append(predictions, values[counts.argmax()])

        return get_f_metrics(answers_test, predictions, self.n_classes)
