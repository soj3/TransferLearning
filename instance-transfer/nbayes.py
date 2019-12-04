from node import Node
from helpers import *
from typing import Any, List, Tuple
from model import AbstractModel
from example import Example


class nbayes(AbstractModel):
    """
    Naive Bayes definition for a binary classifier
    """

    def __init__(self):
        super().__init__()
        self.bins = 25
        self.pNode = None

    # find where we fit on the training set
    def fit(self, examples: List[Example]):
        self.pNode = Node()
        self.pNode.pos_neg_occs = calculate_label_occurrences(examples)

        num_features = len(examples[0].features)

        # Find conditional probabilities
        for attr_idx in range(num_features):
            cNode = Node(attr_idx)

            pos_bins, neg_bins = calculate_continuous_occurrences(
                examples, self.bins, attr_idx
            )

            cNode.pos_bins = pos_bins
            cNode.neg_bins = neg_bins
            cNode.pos_neg_occs = self.pNode.pos_neg_occs
            self.pNode.add_child(cNode)

    # input is an example, output is if the classifier said it was true or false and its confidence
    def classify(self, example: List[Any]) -> Tuple[bool, float]:
        pos_prob, neg_prob = self.pNode.evaluate_example(example)
        if pos_prob > 0.5:
            return False, pos_prob
        else:
            return True, neg_prob
