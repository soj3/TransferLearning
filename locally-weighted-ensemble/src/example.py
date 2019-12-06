from typing import List, Dict, Tuple
from collections.abc import Sequence, Hashable


class Example(Sequence, Hashable):
    def __init__(self, label: int, weight: float = 1):
        self.label = label
        self.features = ()

    def __getitem__(self, id):
        return self.features.__getitem__(id)

    def __len__(self):
        return self.features.__len__()

    def __iter__(self):
        return self.features.__iter__()

    def __eq__(self, value):
        return (
            isinstance(value, Example)
            and self.label == value.label
            and self.features == value.features
        )

    def __ne__(self, value):
        return not self.__eq__(value)

    def __hash__(self):
        return hash((self.label, self.features))


class SentimentExample(Example):
    def __init__(self, words: Dict[str, int], label: int, weight: float = 1):
        super().__init__(label)
        self.words = words

    def create_features(self, vocab: List[Tuple[str, int]]) -> None:
        self.features = tuple(self.words[v] if v in self.words else 0 for v, _ in vocab)

    def __repr__(self):
        return f"{self.features}, {self.label}\n"

    def __str__(self):
        return self.__repr__()
