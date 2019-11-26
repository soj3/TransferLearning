import sys


class Stump(object):
    """
    instance of a decision stump
    """

    def __init__(self, attribute=None):
        self.attribute = attribute
        self.values = {}

    def __repr__(self):
        return "<Attribute: {}, Values Size: {}>".format(
            self.attribute, len(self.values)
        )

    def evaluate_example(self, example, schema):
        """
        propagates examples through the built tree in order to classify them
        """
        for idx, feature in enumerate(schema):
            if feature.name == self.attribute.name:
                break

        return self.values[example[idx]]

