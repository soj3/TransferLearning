from statistics import stdev, mean
from typing import List, Dict, Set
from collections import Counter
from example import Example
import numpy as np
import random


def calculate_stats(arr_of_confusion: List[Dict]):
    accuracies = []
    precisions = []
    recalls = []
    for matrix in arr_of_confusion:
        accuracies.append((matrix["tp"] + matrix["tn"]) / (sum(matrix.values())))
        precisions.append(matrix["tp"] / (matrix["tp"] + matrix["fp"]))
        recalls.append(matrix["tp"] / (matrix["tp"] + matrix["fn"]))

    print_formatted_stats("Accuracy", accuracies)
    print_formatted_stats("Precision", precisions)
    print_formatted_stats("Recall", recalls)


def print_formatted_stats(stat_str, to_stat):
    if len(to_stat) > 1:
        print(f"{stat_str}: {round(mean(to_stat), 3)}, {round(stdev(to_stat), 3)}")
    else:
        print(f"{stat_str}: {round(to_stat[0], 3)}")


def calculate_aroc(arr_of_confidence):
    sort_conf = sorted(arr_of_confidence, key=lambda conf: conf[1], reverse=True)
    total_positive = len([x for x in sort_conf if x[0]])
    total_negative = len(sort_conf) - total_positive
    roc_points = []
    fp, tp = 0, 0
    for label, confidence in sort_conf:
        if label:
            tp += 1
        else:
            fp += 1
        fpr = fp / total_negative if total_negative > 0 else 0
        tpr = tp / total_positive if total_positive > 0 else 0
        roc_points.append([fpr, tpr])

    print(f"Area under ROC: {calculate_integral(roc_points):0.3f}")


def calculate_integral(points):
    area = 0
    pt_idx = 1
    while pt_idx < len(points):
        dx = points[pt_idx][0] - points[pt_idx - 1][0]
        area += ((points[pt_idx][1] + points[pt_idx - 1][1]) / 2) * dx
        pt_idx += 1

    return area


def n_fold_cross_validation(examples, num_folds=5):
    """
    returns a list of length num_folds containing
    tuples of the form (train_set, test_set)
    """
    random.shuffle(examples, random.seed(12345))
    batches = [examples[i::num_folds] for i in range(num_folds)]
    n_fold_sets = [
        (
            [example for flat in (batches[:i] + batches[i + 1 :]) for example in flat],
            batch,
        )
        for i, batch in enumerate(batches)
    ]

    return np.array(n_fold_sets)


def most_common_labels(examples: List[Example], top_n: int = 1) -> List:
    """
    return a list of the top n class labels from a list of examples, where
    each example is an array of feature values and a class label
    """
    top_labels = Counter([example.label for example in examples]).most_common(top_n)
    return [label[0] for label in top_labels]


def calculate_label_occurrences(examples):
    """
    Finds the occurrences of positive examples for a given attribute or value
    attr_idx: this can be specified if a specific attribute should be counted
    """
    positive_examples = sum([example.label for example in examples])
    return [positive_examples, len(examples) - positive_examples]


def calculate_nominal_occurrences(examples: List[Example], attr_idx: int):
    """
    Finds the occurrences of each value of a nominal attribute
    """
    value_occs = {}

    for example in examples:
        value = example[attr_idx]

        if value not in value_occs:
            value_occs[value] = [0, 0]

        value_occs[example[attr_idx]][int(example.label != 1)] += example.weight

    return [value_occ for key, value_occ in value_occs.items()]
