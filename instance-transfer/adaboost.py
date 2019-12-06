import math
import random
import numpy as np
import sys
from data import *
import argparse
from example import SentimentExample
from helpers import n_fold_cross_validation
from sklearn.naive_bayes import MultinomialNB
from stats import calculate_aroc, calculate_stats
import matplotlib.pyplot as plt


def boost(iterations, features):
    """
    initialize Boost
    """

    print("Collecting Data")

    # Review Data
    # b_data, d_data, e_data, k_data = collect_review_data(features)
    # b_data_folds = n_fold_cross_validation(b_data)
    # d_data_folds = n_fold_cross_validation(d_data)
    # e_data_folds = n_fold_cross_validation(e_data)
    # k_data_folds = n_fold_cross_validation(k_data)

    # Spam Task A Data
    # sp1, sp2, sp3 = collect_spam_a_data(features)
    # sp1_data_folds = n_fold_cross_validation(sp1)
    # sp2_data_folds = n_fold_cross_validation(sp2)
    # sp3_data_folds = n_fold_cross_validation(sp3)

    # Spam Task B Data
    # sps15 = collect_spam_b_data(features)

    # Spam News Group
    nws1, nws2 = collect_newsgroup_data(features)
    nws1_data_folds = n_fold_cross_validation(nws1)
    nws2_data_folds = n_fold_cross_validation(nws2)

    print("Finished Collecting Data")

    confused_matrix_bois = []
    confused_output_bois = []

    d_domain = nws1_data_folds
    s_domain = nws2_data_folds

    d_domain[:, 1] = s_domain[:, 1]

    for idx, (train, test) in enumerate(d_domain):
        print("Running Fold {}".format(idx + 1))

        output, matrix = run_boost(train, test, iterations)
        confused_matrix_bois.append(matrix)
        confused_output_bois += output

    calculate_stats(arr_of_confusion=confused_matrix_bois)
    calculate_aroc(arr_of_confidence=confused_output_bois)


def run_boost(train, test, iterations):
    # Create equal weights for every example
    for ex in train:
        ex.weight = 1 / len(train)

    classifiers = []
    classifier_weights = []

    current_itr = 0

    # run for the input number of iterations
    while current_itr < iterations:
        print("Boost Iteration: {}".format(current_itr + 1))
        classifiers.append(MultinomialNB())

        # randomly sample the training set with replacement
        data = [ex.features for ex in train]
        labels = [ex.label for ex in train]

        classifiers[-1].fit(data, labels)

        # Extract weights and outputs
        weights = [ex.weight for ex in train]
        outputs = np.array(classifiers[-1].predict(data)) != labels

        # Calculate classifier error
        error = weight_error(weights, outputs)
        classifier_weights.append(classifier_weight(error))

        if error >= 0.5:
            break

        # Update the weights
        new_weights = update_weights(weights, outputs, classifier_weights[-1])
        for idx, ex in enumerate(train):
            ex.weight = new_weights[idx]

        current_itr += 1

    outputs = []
    matrix = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    # Run Testing
    for ex in test:

        vote = 0
        conf = 0

        total_class_weight = sum(classifier_weights)

        for classifier_idx, classifier in enumerate(classifiers):
            probs = classifier.predict_proba([ex.features])[0]
            output = int(probs[0] < probs[1])

            coeff = classifier_weights[classifier_idx] / total_class_weight
            vote += coeff * output
            conf += coeff * probs[1]

        # Make the vote discrete
        vote = True if vote >= 0.5 else False

        # Calculate outputs and matrix
        outputs.append((ex.label, conf))
        is_correct = "t" if ex.label == vote else "f"
        is_positive = "p" if vote else "n"
        matrix[is_correct + is_positive] += 1

    return outputs, matrix


def weight_error(weights, output):
    """
    find the error of the weights
    """
    return np.sum(np.multiply(weights, output))


def classifier_weight(error):
    """
    find the weight of the classifier itself
    """
    EPSILON = 1e-10

    return 0.5 * math.log((1 - error + EPSILON) / (error + EPSILON))


def update_weights(weights, output, alpha):
    """
    updated the weights of the data
    """
    weights = np.array(weights)
    converted_output = np.array(list(map(lambda o: -1 if o else 1, output)))
    updated_weights = np.multiply(weights, np.exp(-alpha * converted_output))

    return updated_weights / sum(updated_weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--iterations", help="Number of iterations to run boosting", type=int
    )
    parser.add_argument(
        "-f", "--features", help="Number of features the vocab should use", type=int
    )

    args = parser.parse_args()

    if args.iterations:
        iterations = args.iterations
    else:
        iterations = 20

    if args.features:
        features = args.features
    else:
        features = 500

    boost(iterations, features)
