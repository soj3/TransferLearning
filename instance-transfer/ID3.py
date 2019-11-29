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
    best_split = None

    for idx in range(num_features):
        temp_gain, temp_split = best_split_continuous(examples, idx)
        if temp_gain > best_gain:
            best_gain = temp_gain
            best_attribute = idx
            best_split = temp_split

    # Sets decision attribute
    root = Stump(best_attribute)
    root.split_val = best_split

    less_exs = [ex for ex in examples if ex.features[best_attribute] < best_split]
    root.less_than = most_common_labels(less_exs)[0]

    great_exs = [ex for ex in examples if ex.features[best_attribute] >= best_split]
    root.greater_than = most_common_labels(great_exs)[0]

    return root


def best_split_continuous(examples: List[Example], attr_idx):
    """
    finds the information gain or gain ratio of a nominal attribute
    """

    # Get label occurrences
    label_occ = calculate_label_occurrences(examples)

    # Sort by attribute
    examples = sorted(examples, key=lambda ex: (ex.features[attr_idx], ex.label))

    # Find splits and find the split occurrences
    splits = class_split_continuous(examples, attr_idx)
    split_occs = calculate_continuous_occurrences(splits, examples, attr_idx)
    # Loop through splits and find max info gain
    best_split = None
    best_split_gain = -1
    for split, cond_occ in split_occs.items():
        temp_gain = info_gain(label_occ, cond_occ)

        if temp_gain > best_split_gain:
            best_split_gain = temp_gain
            best_split = split

    return (best_split_gain, best_split)
