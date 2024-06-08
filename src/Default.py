import numpy as np


def get_sample(n_objects, n_classes, mean_arr, standard_deviation):
    n1 = int(n_objects / n_classes)
    attributes = np.random.normal(mean_arr[0], standard_deviation, n1 * 2).reshape(-1, 2)
    answers = np.ones(n1)
    for i in range(1, n_classes):
        attributes = np.vstack(
            [attributes, np.random.normal(mean_arr[i], standard_deviation, n1 * 2).reshape(-1, 2)])
        answers = np.hstack([answers, np.ones(n1) * (i + 1)])
    return attributes, answers


def get_accuracy_precision_recall(y, predictions):
    accuracy = {}
    number_obj_one_class = {}
    tp = {}
    fp = {}
    fn = {}
    precision = {}
    recall = {}
    classes = np.unique(y)
    for j in range(0, classes.shape[0]):
        accuracy[classes[j]] = 0
        number_obj_one_class[classes[j]] = 0
        tp[classes[j]] = 0
        fp[classes[j]] = 0
        fn[classes[j]] = 0
        precision[classes[j]] = 0
        recall[classes[j]] = 0
        for i in range(0, y.shape[0]):
            if y[i] == classes[j]:
                number_obj_one_class[classes[j]] += 1
                if predictions[i] == classes[j]:
                    tp[classes[j]] += 1
                    accuracy[classes[j]] += 1
                else:
                    fn[classes[j]] += 1
            elif predictions[i] == classes[j]:
                fp[classes[j]] += 1
    for j in range(0, classes.shape[0]):
        if tp[classes[j]] + fp[classes[j]] == 0:
            precision[classes[j]] = 1
        else:
            precision[classes[j]] = tp[classes[j]] / (tp[classes[j]] + fp[classes[j]])
        if tp[classes[j]] + fn[classes[j]] == 0:
            recall[classes[j]] = 1
        else:
            recall[classes[j]] = tp[classes[j]] / (tp[classes[j]] + fn[classes[j]])
        if accuracy[classes[j]] != 0:
            accuracy[classes[j]] = accuracy[classes[j]] / number_obj_one_class[classes[j]]
    return (np.asarray(list(accuracy.values())),
            np.asarray(list(precision.values())),
            np.asarray(list(recall.values())))


def get_mean_for_classes(n_classes, first_mean, different):
    mean = np.array([first_mean])
    for i in range(0, n_classes - 1):
        mean = np.append(mean, mean[i] + different)
    return mean


def get_f_metrics(y, predictions, n_classes):
    tp = {}
    fp = {}
    fn = {}
    f = {}
    classes = np.unique(y)
    for j in range(0, classes.shape[0]):
        tp[classes[j]] = 0
        fp[classes[j]] = 0
        fn[classes[j]] = 0
        f[classes[j]] = 0
        for i in range(0, y.shape[0]):
            if y[i] == classes[j] and predictions[i] == classes[j]:
                tp[classes[j]] += 1
            elif y[i] == classes[j] and predictions[i] != classes[j]:
                fn[classes[j]] += 1
            elif y[i] != classes[j] and predictions[i] == classes[j]:
                fp[classes[j]] += 1
    for j in range(0, classes.shape[0]):
        if tp[classes[j]] + fp[classes[j]] == 0:
            precision = 1
        else:
            precision = tp[classes[j]] / (tp[classes[j]] + fp[classes[j]])
        if tp[classes[j]] + fn[classes[j]] == 0:
            recall = 1
        else:
            recall = tp[classes[j]] / (tp[classes[j]] + fn[classes[j]])
        if recall + precision == 0:
            f[classes[j]] = 0
        else:
            f[classes[j]] = 2 * precision * recall / (precision + recall)
    return np.sum(np.asarray(list(f.values()))) / n_classes
