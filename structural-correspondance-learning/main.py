import data
import argparse as argp
from information_gain import calc_mutual_info


LAMBDA = 1e-3
MU = 1e-1


def source_loss():
    pass


def correct_misalignments():
    pass


def alpha_dist():
    pass


def select_pivots(labeled_source, unlabeled_source, unlabeled_target, source_vocab, target_vocab, num_pivots=1000):
    # want to choose the num_pivots features with the highest mutual information gain to the source label
    # sort the features according to how many times they occur in both the source and target domains
    # then, choose the num_pivots features with the highest mutual info to the source label

    # criteria for pivots: occurs more than 50 times, occurs in more than 5 examples, occurs in both domains
    pivots = []
    potential_pivots = []
    # from the unlabeled source and unlabeled target data, find features that fulfill these criteria
    new_dict1 = {k: v for (k, v) in source_vocab if v > 50}
    new_dict2 = {k: v for (k, v) in target_vocab if v > 50}
    for key in new_dict1.keys():
        if key in new_dict2.keys():
            num_occ1, num_occ2 = 0, 0
            for example in unlabeled_source:
                if key in example.words.keys():
                    num_occ1 += 1
            for example in unlabeled_target:
                if key in example.words.keys():
                    num_occ2 += 1
            if num_occ1 > 5 and num_occ2 > 5:
                potential_pivots.append(key)

    # create a dictionary containing the potential pivot features and their corresponding info gain to source
    info = {}
    for feature in potential_pivots:
        info[feature] = calc_mutual_info(labeled_source, feature)
    # sort according to mutual information
    sorted_info = sorted(info.items(), key=lambda item: item[1], reverse=True)

    # add top num_pivots to pivot list
    for i in range(num_pivots):
        pivots.append(sorted_info[i][0])

    return pivots




def scl():
    pass


def main():
    ap = argp.ArgumentParser()
    ap.add_argument("-s", "--source", required=True, help="Source domain")
    ap.add_argument("-t", "--target", required=True, help="Target domain")
    args = vars(ap.parse_args())
    labeled_source, unlabeled_source, source_vocab = data.collect_review_data(args["source"])
    _, unlabeled_target, target_vocab = data.collect_review_data(args["target"])

    # split source and target datasets into training and testing data
    #baseline classifier


    #classifier trained in domain

    #classifiers trained on new domains




source_labeled, source_unlabeled, source_vocab = data.collect_review_data("books")
_, target_unlabeled, target_vocab = data.collect_review_data("dvd")

pivots = select_pivots(source_labeled, source_unlabeled, target_unlabeled, source_vocab, target_vocab, num_pivots = 30)
print(pivots)