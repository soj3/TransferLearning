from stump import Stump
from gain import info_gain
from helpers import *


def ID3(examples):
    """
    returns a tree built using the ID3 algorithm, with each successive level
    containing the attribute with the most information gain for a set of examples
    """

    num_features = len(examples[0].features)

    best_gain = -1
    best_attribute = None

    for idx in range(num_features):
        temp_gain = best_split_nominal(examples, idx)
        if temp_gain > best_gain:
            best_gain = temp_gain
            best_attribute = idx

    # Gets all possible values of the attribute
    values_set = set(ex.features[best_attribute] for ex in examples)

    # Sets decision attribute
    root = Stump(best_attribute)
    for value in values_set:
        filtered_examples = [
            example for example in examples if example.features[best_attribute] == value
        ]

        label = most_common_labels(filtered_examples)

        root.values[value] = label
    return root


def best_split_nominal(examples, attr_idx):
    """
    finds the information gain or gain ratio of a nominal attribute
    """
    label_occ = calculate_label_occurrences(examples)
    nominal_occ = calculate_nominal_occurrences(examples, attr_idx)

    return info_gain(label_occ, nominal_occ)
