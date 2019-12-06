from typing import List, Dict, Tuple

"""
example class to contain data, which has a label and the list of features for each data point
"""


class Example(object):

    def __init__(self, label: int):
        self.label = label

        self.features = []


"""
contains the vocabulary and a method to create the features based on the vocabulary
"""


class SentimentExample(Example):

    def __init__(self, words: Dict[str, int], label: int, weight: int = 1):

        super().__init__(label, weight)

        self.words = words

    def create_features(self, vocab: List[Tuple[str, int]]) -> None:

        self.vocab = [x[0] for x in vocab]

        for v, __ in vocab:

            if v in self.words:

                self.features.append(self.words[v])

            else:

                self.features.append(0)
