import sys
from example import Example
from typing import Any, List


class Stump(object):
    """
    instance of a decision stump
    """

    def __init__(self, attribute: Any = None):
        self.attribute = attribute

        self.split_val = None

        self.less_than = None
        self.less_than_vals = [0, 0]

        self.greater_than = None
        self.greater_than_vals = [0, 0]

    def __repr__(self):
        return "<Attribute: {}, Split value: {}>".format(self.attribute, self.split_val)

    def evaluate_example(self, example: Example):
        value = example.features[self.attribute]

        if value < self.split_val:
            label = self.less_than
            occs = self.less_than_vals
        else:
            label = self.greater_than
            occs = self.greater_than_vals

        conf = occs[example.label] / sum(occs)

        return label, conf

    def find_confidences(self, data: List[Example]):
        for ex in data:
            value = ex.features[self.attribute]
            label = ex.label

            if value < self.split_val:
                self.less_than_vals[label] += 1
            else:
                self.greater_than_vals[label] += 1

