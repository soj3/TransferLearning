import data
import argparse as argp
import sklearn.linear_model as model
import sklearn.decomposition as skd
from information_gain import calc_mutual_info
from utils import *
import numpy as np


LAMBDA = 1e-3
MU = 1e-1
NUM_PIVOTS = 30
NUM_FEATURES = 100


def source_loss():
    pass


def correct_misalignments():
    pass


def alpha_dist():
    pass


def select_pivots(labeled_source, unlabeled_source, unlabeled_target, source_vocab, target_vocab, num_pivots=NUM_PIVOTS):
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


def get_pivot_predictor_weights(data, vocab, pivots):
    weights = []
    j = 1
    # for each pivot, we create a classifier that predicts the likelihood of that pivot appearing in the example,
    # given all of the other features (i.e. words) of the example
    for pivot in pivots:
        x = []
        y = []
        # remove the pivot from the vocabulary
        temp_vocab = [(k, v) for (k, v) in vocab if k != pivot]
        # Here the class label is 1 or 0 depending on the appearance of the pivot in the example
        # maybe i should change this to -1 because we want the classifier to output a negative number if the
        # pivot is not there?
        for i in range(len(data)):
            if pivot in data[i].words:
                y.append(1)
            else:
                y.append(0)
            data[i].create_features(temp_vocab)
            x.append(data[i].features)
        print("Training pivot predictor", j)
        # train a Stochastic gradient descent classifier using the modified Huber loss function
        classifier = model.SGDClassifier(loss="modified_huber")
        classifier.fit(x, y)
        weight = []
        for i in classifier.coef_[0]:
            weight.append(i)
        weights.append(weight)
        j += 1
    return weights


def create_classifiers(pivot_matrix, train_source, train_source_labels, train_target, train_target_labels):

    classifiers = []

    # train the baseline classifier, which is a linear model with no adaptation
    print("Training baseline...")
    baseline = model.SGDClassifier(loss="modified_huber")
    baseline.fit(train_source, train_source_labels)
    classifiers.append(baseline)

    # train the source classifier with adaptation
    print("Training source classifier...")
    for ex in train_source:
        adapted_features = np.dot(ex, pivot_matrix)
        ex.extend(adapted_features)
    source = model.SGDClassifier(loss="modified_huber")
    source.fit(train_source, train_source_labels)
    classifiers.append(source)

    # train the target classifier with adaptation
    print("Training target classifier...")
    for ex in train_target:
        adapted_features = np.dot(ex, pivot_matrix)
        ex.extend(adapted_features)
    target = model.SGDClassifier(loss="modified_huber")
    target.fit(train_target, train_target_labels)
    classifiers.append(target)

    return classifiers


def scl(source, target):
    print("Reading data...")
    source_labeled, source_unlabeled, source_vocab = data.collect_review_data(source)
    target_labeled, target_unlabeled, target_vocab = data.collect_review_data(target)

    print("Selecting pivots...")
    pivots = select_pivots(source_labeled, source_unlabeled, target_unlabeled, source_vocab, target_vocab)

    # create a binary classifier for each pivot feature on the combined unlabeled data of the source and target

    unlabeled_data = source_unlabeled + target_unlabeled
    merged_vocab = merge_list(source_vocab, target_vocab)
    final_vocab = merge_pivots_and_vocab(merged_vocab[:NUM_FEATURES], pivots)

    print("Collecting pivot predictor weights...")
    weights = get_pivot_predictor_weights(unlabeled_data, final_vocab[:NUM_FEATURES + 1], pivots)

    # compute the Singular value decomposition of the weights matrix
    print("Calculating SVD...")
    weights = np.asmatrix(weights, dtype=float).transpose()
    svd = skd.TruncatedSVD(n_components=25)
    pivot_matrix = svd.fit_transform(weights)

    print("Training classifiers...")

    for ex in source_labeled:
        ex.create_features(source_vocab[:NUM_FEATURES])
    for ex in target_labeled:
        ex.create_features(target_vocab[:NUM_FEATURES])

    train_source, train_source_labels, test_source, test_source_labels = split_data(source_labeled)
    train_target, train_target_labels, test_target, test_target_labels = split_data(target_labeled)

    classifiers = create_classifiers(pivot_matrix, train_source, train_source_labels, train_target, train_target_labels)


def main():
    # ap = argp.ArgumentParser()
    # ap.add_argument("-s", "--source", required=True, help="Source domain")
    # ap.add_argument("-t", "--target", required=True, help="Target domain")
    # args = vars(ap.parse_args())

    scl("books", "dvd")
    # split source and target datasets into training and testing data
    # baseline classifier

    # classifier trained in domain

    # classifiers trained on new domains


if __name__ == "__main__":
    main()



