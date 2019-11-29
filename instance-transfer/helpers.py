from collections import Counter
from typing import List, Set
from example import Example

import random


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
    return n_fold_sets


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
    positive_examples = sum([1 for example in examples if example.label == 1])
    return [positive_examples, len(examples) - positive_examples]


def calculate_nominal_occurrences(examples: List[Example], attr_idx: int):
    """
    Finds the occurrences of each value of a nominal attribute
    """

    num_features = len(examples)

    value_occs = {feature: [0, 0] for feature in range(num_features)}

    for example in examples:
        value_occs[example.features[attr_idx]][int(example.label != 1)] += 1

    return [value_occ for key, value_occ in value_occs.items()]
