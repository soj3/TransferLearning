from typing import List, Dict, Tuple
from collections.abc import MutableSequence


class Example(MutableSequence):
    def __init__(self, label: int, weight: float):
        self.weight = weight
        self.label = label
        self.features = []

    def __getitem__(self, id):
        return self.features.__getitem__(id)

    def __setitem__(self, key, value):
        return self.features.__setitem__(key, value)

    def __delitem__(self, key):
        return self.features.__delitem__(key)

    def __len__(self):
        return self.features.__len__()

    def insert(self, key, value):
        return self.features.insert(key, value)


e = Example(label=1, weight=1)


class SentimentExample(Example):
    def __init__(self, words: Dict[str, int], label: int, weight: float = 1):
        super().__init__(label, weight)
        self.words = words

    def create_features(self, vocab: List[Tuple[str, int]]) -> None:
        self.features = []
        for v, __ in vocab:
            if v in self.words:
                self.features.append(self.words[v])
            else:
                self.features.append(0)

    def __repr__(self):
        return f"{self.features}, {self.label}\n"

    def __str__(self):
        return self.__repr__()
