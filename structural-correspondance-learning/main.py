import data
import argparse as argp
import sklearn.decomposition as skd
from pivot import *
from classifier import *
from utils import *
import numpy as np


LAMBDA = 1e-3
MU = 1e-1
NUM_PIVOTS = 500
NUM_FEATURES = 5000


def source_loss():
    pass


def correct_misalignments():
    pass


def alpha_dist():
    pass


def scl(source, target):
    print("Reading data...")
    source_labeled, source_unlabeled, source_vocab = data.collect_review_data(source)
    target_labeled, target_unlabeled, target_vocab = data.collect_review_data(target)

    print("Selecting pivots...")
    pivots = select_pivots(source_labeled, source_unlabeled, target_unlabeled, source_vocab, target_vocab, NUM_PIVOTS)

    # create a binary classifier for each pivot feature on the combined unlabeled data of the source and target

    unlabeled_data = source_unlabeled + target_unlabeled
    merged_vocab = merge_list(source_vocab, target_vocab)
    final_vocab = merge_pivots_and_vocab(merged_vocab[:10000], pivots)
    final_vocab = merge_pivots_and_vocab(final_vocab[:10000], pivots)
    print("Collecting pivot predictor weights...")
    weights = get_pivot_predictor_weights(unlabeled_data, final_vocab[:NUM_FEATURES], pivots, NUM_FEATURES-1)

    # compute the Singular value decomposition of the weights matrix
    print("Calculating SVD...")
    row_len = len(weights[0])
    for weight in weights:
        assert len(weight) == row_len
    weights = np.asmatrix(weights, dtype=float).transpose()
    svd = skd.TruncatedSVD(n_components=25)
    pivot_matrix = svd.fit_transform(weights)

    print("Training classifiers...")

    for ex in source_labeled:
        ex.create_features(source_vocab[:NUM_FEATURES-1])
    for ex in target_labeled:
        ex.create_features(target_vocab[:NUM_FEATURES-1])

    train_source, train_source_labels, test_source, test_source_labels = split_data(source_labeled)
    train_target, train_target_labels, test_target, test_target_labels = split_data(target_labeled)

    classifiers = create_classifiers(pivot_matrix, train_source, train_source_labels, train_target, train_target_labels)

    evaluate_classifiers(classifiers, test_source, test_source_labels, test_target, test_target_labels, pivot_matrix)


def main():
    # ap = argp.ArgumentParser()
    # ap.add_argument("-s", "--source", required=True, help="Source domain")
    # ap.add_argument("-t", "--target", required=True, help="Target domain")
    # args = vars(ap.parse_args())

    scl("books", "dvd")
    # scl("books", "kitchen")
    # scl("books", "electronics")
    # scl("dvd", "electronics")
    # scl("dvd", "kitchen")
    # scl("electronics", "kitchen")
    # split source and target datasets into training and testing data
    # baseline classifier

    # classifier trained in domain

    # classifiers trained on new domains


if __name__ == "__main__":
    main()



