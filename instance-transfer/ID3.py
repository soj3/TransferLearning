from stump import Stump
from gain import info_gain
from mldata import *
from helpers import *


def ID3(examples):
    """
    returns a tree built using the ID3 algorithm, with each successive level
    containing the attribute with the most information gain for a set of examples
    """
    attrs = examples.schema

    best_gain = -1
    best_attribute = None
    best_attribute_idx = None

    for idx, attribute in enumerate(attrs):
        temp_gain = best_split_nominal(attrs, examples, idx)
        if temp_gain > best_gain:
            best_gain = temp_gain
            best_attribute = attribute
            best_attribute_idx = idx

    # Sets decision attribute

    root = Stump(best_attribute)
    for value in root.attribute.values:
        filtered_examples = [
            example for example in examples if example[best_attribute_idx] == value
        ]

        label = most_common_labels(filtered_examples)

        root.values[value] = label
    return root


def best_split_nominal(attributes, examples, attr_idx):
    """
    finds the information gain or gain ratio of a nominal attribute
    """
    label_occ = calculate_label_occurrences(examples)
    nominal_occ = calculate_nominal_occurrences(attributes, examples, attr_idx)

    return info_gain(label_occ, nominal_occ)
