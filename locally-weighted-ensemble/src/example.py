from typing import List, Dict, Tuple


class Example(object):
    def __init__(self, label: int, weight: int):
        self.weight = weight
        self.label = label
        self.features = []


class SentimentExample(Example):
    def __init__(self, words: Dict[str, int], label: int, weight: int = 1):
        super().__init__(label, weight)
        self.words = words

    def create_features(self, vocab: List[Tuple[str, int]]) -> None:
        for v, __ in vocab:
            if v in self.words:
                self.features.append(self.words[v])
            else:
                self.features.append(0)

    def __repr__(self):
        return f"{self.features}, {self.label}\n"
