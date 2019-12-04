from copy import deepcopy
import data
import argparse as argp
import scipy.optimize as spo
import sklearn.decomposition as skd
from pivot import *
from classifier import *
from utils import *
import numpy as np


LAMBDA = 1e-1
MU = 1e-1
NUM_PIVOTS = 26
NUM_FEATURES = 100
SVD_DIMENSION = 25


def source_loss(weights, original_adapted_weights, train_examples, train_labels):
    loss = 0
    predictions = []
    for ex in train_examples:
        if np.dot(weights, ex) > 0:
            predictions.append(1)
        else:
            predictions.append(-1)

    assert len(predictions) == len(train_labels)

    base_weights = weights[:-SVD_DIMENSION-1]
    adapted_weights = weights[-SVD_DIMENSION-1:-1]
    for i in range(len(predictions)):
        if predictions[i] * train_labels[i] >= 1:
            pass
        elif predictions[i] * train_labels[i] < -1:
            loss += (-4 * predictions[i] * train_labels[i])
        else:
            loss += ((1 - predictions[i] * train_labels[i]) ** 2)

    return loss + (LAMBDA * np.linalg.norm(base_weights) ** 2) + \
        (MU * np.linalg.norm(np.subtract(original_adapted_weights, adapted_weights) ** 2))


def gradient(weights, adapted_weights, train_examples, train_labels):
    grad = 0
    predictions = []
    for ex in train_examples:
        if np.dot(weights,ex) >0:
            predictions.append(1)
        else:
            predictions.append(-1)

    for i in range(len(predictions)):
        if predictions[i] * train_labels[i] + 1 <= 1:
            grad += -4
        elif -1 < predictions[i] * train_labels[i] < 1:
            grad += -2 * (1 - predictions[i] * train_labels[i])

    return [grad]


def correct_misalignments(base_classifier, train_examples, train_labels, pivot_matrix):
    print("Correcting misalignments...")
    adapted_examples = deepcopy(train_examples)
    for ex in adapted_examples:
        #print(len(ex))
        adapted_features = np.dot(ex, pivot_matrix)
        ex.extend(adapted_features)
    weights = base_classifier.coef_[0]
    adapted_weights = weights[-SVD_DIMENSION-1:-1]

    new_weights = spo.fmin(source_loss, weights,  args=(adapted_weights, adapted_examples, train_labels), maxiter=10000)
    new_classifier = LogRegClassifier(new_weights)
    return new_classifier


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
    final_vocab = merge_pivots_and_vocab(merged_vocab, pivots, NUM_FEATURES)
    final_vocab = merge_pivots_and_vocab(final_vocab[:NUM_FEATURES + NUM_PIVOTS], pivots, NUM_FEATURES)
    print("Collecting pivot predictor weights...")
    weights = get_pivot_predictor_weights(unlabeled_data, final_vocab[:NUM_FEATURES], pivots, NUM_FEATURES-1)

    # compute the Singular value decomposition of the weights matrix
    print("Calculating SVD...")
    row_len = len(weights[0])
    for weight in weights:
        assert len(weight) == row_len
    weights = np.asmatrix(weights, dtype=float).transpose()
    svd = skd.TruncatedSVD(n_components=SVD_DIMENSION)
    pivot_matrix = svd.fit_transform(weights)

    print("Training classifiers...")

    for ex in source_labeled:
        ex.create_features(source_vocab[:NUM_FEATURES-1])
    for ex in target_labeled:
        ex.create_features(target_vocab[:NUM_FEATURES-1])

    train_source, train_source_labels, test_source, test_source_labels = split_data(source_labeled)
    train_target, train_target_labels, test_target, test_target_labels = split_data(target_labeled)
    classifiers = create_classifiers(pivot_matrix, train_source, train_source_labels, train_target, train_target_labels)

    for (k, v) in classifiers:
        if k == "Source":
            new_classifier = correct_misalignments(v, train_target[:50], train_target_labels[:50], pivot_matrix)
            classifiers.append(("Corrected Source", new_classifier))
        elif k == "Target":
            new_classifier = correct_misalignments(v, train_source[:50], train_source_labels[:50], pivot_matrix)
            classifiers.append(("Corrected Target", new_classifier))

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



