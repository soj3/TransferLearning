from ID3 import ID3
from stump import Stump
from typing import List
from example import Example


class DStump(object):
    def __init__(self):
        self.stump: Stump = None

    def fit(self, data: List[Example]):
        self.stump = ID3(data)

    def classify(self, example: Example):
        return self.stump.evaluate_example(example)
