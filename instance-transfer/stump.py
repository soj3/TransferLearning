import sys
from typing import Any


class Stump(object):
    """
    instance of a decision stump
    """

    def __init__(self, attribute: Any = None):
        self.attribute = attribute
        self.values = {}

    def __repr__(self):
        return "<Attribute: {}, Values Size: {}>".format(
            self.attribute, len(self.values)
        )

    def evaluate_example(self, example):
        return self.values[example.features[self.attribute]]

