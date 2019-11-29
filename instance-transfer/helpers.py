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
    positive_examples = sum(
        [example.weight for example in examples if example.label == 1]
    )

    negative_examples = sum(
        [example.weight for example in examples if example.label == 0]
    )

    return [positive_examples, negative_examples]


def class_split_continuous(examples: List[Example], ftr: int):
    """
    return values of which to split on for a continuous attribute
    """
    split_values = []
    i = 1
    while i < len(examples):
        if examples[i].label != examples[i - 1].label:
            # Iterate while vals are same
            while (
                i < len(examples) - 1
                and examples[i].features[ftr] == examples[i - 1].features[ftr]
            ):
                i += 1

            # Append split
            split_values.append(
                (examples[i].features[ftr] + examples[i - 1].features[ftr]) / 2.0
            )

        i += 1
    return split_values


def calculate_continuous_occurrences(
    splits: List[float], examples: List[Example], ftr: int
):
    """
    Finds the occurrences of each value of a continuous attribute
    """
    split_occs = {}
    total_occs = [[0, 0], [0, 0]]

    for example in examples[1:]:
        total_occs[1][int(example.label != 1)] += example.weight
    total_occs[0][int(not examples[0].label)] += examples[0].weight

    n_split = 0
    for example in examples[1:]:
        if example.features[ftr] <= splits[n_split]:
            total_occs[0][int(not example.label)] += example.weight
            total_occs[1][int(not example.label)] -= example.weight
        else:
            split_occs[splits[n_split]] = total_occs
            n_split += 1

        if n_split == len(splits):
            break
    return split_occs
