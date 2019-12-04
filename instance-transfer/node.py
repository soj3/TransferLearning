from math import log


class Node(object):
    """
    instance of an n-ary tree node
    """

    def __init__(self, attribute=None):
        self.attribute = attribute

        self.children = []
        self.pos_neg_occs = []

        self.pos_bins = []
        self.neg_bins = []

    def add_child(self, child):
        self.children.append(child)

    def evaluate_example(self, example):
        if len(self.children) == 0:
            print("Error")
            return
        else:
            pos_prob = log(self.pos_neg_occs[0] / sum(self.pos_neg_occs))
            neg_prob = log(self.pos_neg_occs[1] / sum(self.pos_neg_occs))

            for child in self.children:
                tPos, tNeg = child.get_cond_prob(example)
                pos_prob += log(tPos + 1e-5)
                neg_prob += log(tNeg + 1e-5)

            pos_norm = pos_prob / (pos_prob + neg_prob)
            neg_norm = neg_prob / (pos_prob + neg_prob)
            return pos_norm, neg_norm

    def get_cond_prob(self, example):
        val = example.features[self.attribute]

        for bin_idx in range(len(self.pos_bins)):
            if self.pos_bins[bin_idx].fits_in_bin(val):
                pos_prob = self.calculate_cond_prob(self.pos_bins[bin_idx], 0)
                neg_prob = self.calculate_cond_prob(self.neg_bins[bin_idx], 1)

        return pos_prob, neg_prob

    def calculate_cond_prob(self, bin, pos_neg_idx):
        prior = self.pos_neg_occs[pos_neg_idx] / sum(self.pos_neg_occs)

        return bin.count / self.pos_neg_occs[pos_neg_idx]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<Pos_neg_occ: {}\n<Attribute: {}\n<Pos_bin: {}\n<Neg_bin: {}".format(
            self.pos_neg_occs, self.attribute, self.pos_bins, self.neg_bins
        )
