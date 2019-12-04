from collections import Counter
from typing import List, Set
from example import Example
from bin import Bin
import numpy as np
import random
from copy import deepcopy


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
    positive_examples = sum([1 for example in examples if example.label == 1])
    return [positive_examples, len(examples) - positive_examples]


def calculate_continuous_occurrences(
    examples: List[Example], num_of_bins: int, attr_idx: int
):
    """
    Finds the occurrences of each value of a continuous attribute
        returns array of bins
    """
    examples = sorted(examples, key=lambda example: (example.features[attr_idx]))

    bins = calculate_cont_bin_values(examples, num_of_bins, attr_idx)

    pos_ex = [ex for ex in examples if ex.label]
    neg_ex = [ex for ex in examples if not ex.label]

    pos_bins = deepcopy(bins)
    neg_bins = deepcopy(bins)

    calculate_continuous_occurrences_label(pos_ex, pos_bins, attr_idx)
    calculate_continuous_occurrences_label(neg_ex, neg_bins, attr_idx)

    return pos_bins, neg_bins


def calculate_cont_bin_values(examples: List[Example], num_of_bins: int, attr_idx: int):
    num_in_bin = int(len(examples) / num_of_bins)
    bin_separators = []

    for bin_idx in range(1, num_of_bins):
        bin_separators.append(examples[bin_idx * num_in_bin].features[attr_idx])

    bins = []
    i = 0

    bins.append(Bin(float("-inf"), bin_separators[i]))
    while i < len(bin_separators) - 1:
        bins.append(Bin(bin_separators[i], bin_separators[i + 1]))
        i += 1
    bins.append(Bin(bin_separators[-1], float("+inf")))

    return bins


def calculate_continuous_occurrences_label(examples: List[Example], bins, attr_idx):
    bin_idx = 0
    for example in examples:
        while (
            not bins[bin_idx].fits_in_bin(example.features[attr_idx])
            and bin_idx < len(bins) - 1
        ):
            bin_idx += 1
        if bins[bin_idx].fits_in_bin(example.features[attr_idx]):
            bins[bin_idx].add_to_bin(example.weight)

    return bins
