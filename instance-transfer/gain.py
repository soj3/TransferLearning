import math

# threshold value for double comparison to zero
EPSILON = 1e-5


def info_gain(label_occ, cond_occ_list):
    """
    Calculates the information gains for one attribute
    label_occ: occurrences of the label [pos_occ, neg_occ]
    cond_occ_list: a list of a lists of
        occurrences for conditionally splitting on that specific attribute
    """
    label_entropy = shannon_entropy(label_occ)
    total_examples = sum(label_occ)

    cond_entropy_list = [
        (sum(cond_occ) / total_examples) * shannon_entropy(cond_occ)
        for cond_occ in cond_occ_list
    ]
    return label_entropy - sum(cond_entropy_list)


def shannon_entropy(occ_list):
    """
    Calculates the shannon entropy of a list of occurrences
    containing exhaustive outcomes of the probability mass function
    """

    prob_list = [occ / sum(occ_list) for occ in occ_list if sum(occ_list) > EPSILON]

    return sum([-prob * math.log2(prob) for prob in prob_list if prob > EPSILON])

