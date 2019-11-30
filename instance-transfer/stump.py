import sys
from typing import List
from example import Example


class Stump(object):
    """
    instance of a decision stump
    """

    def __init__(self, attribute: int = None):
        self.attribute = attribute
        self.values = {}

        self.confidences = {}

    def evaluate_example(self, example: Example):
        value = example.features[self.attribute]
        label = example.label

        if value not in self.values:
            value = min(self.values.keys(), key=lambda x: abs(x - value))

        occs = self.confidences[value]
        conf = occs[label] / sum(occs)

        return self.values[value], conf

    def find_confidences(self, examples: List[Example]):
        self.confidences = {key: [0, 0] for key in self.values.keys()}

        for ex in examples:
            value = ex.features[self.attribute]
            label = ex.label

            self.confidences[value][label] += 1

    def __repr__(self):
        return "<Attribute: {}, Values Size: {}>".format(
            self.attribute, len(self.values)
        )
