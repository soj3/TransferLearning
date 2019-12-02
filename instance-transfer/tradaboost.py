import math
import random
import numpy as np
import sys
from data import collect_review_data
from example import SentimentExample
from helpers import n_fold_cross_validation
from dstump import DStump
from stats import calculate_aroc, calculate_stats
import matplotlib.pyplot as plt


def boost():
    """
    initialize Boost
    """

    print("Collecting Data")
    b_data, d_data, e_data, k_data = collect_review_data(100)
    print("Finished Collecting Data")

    iterations = 20
    labeled_data = 100

    confused_matrix_bois = []
    confused_output_bois = []

    b_data_folds = n_fold_cross_validation(b_data)
    d_data_folds = n_fold_cross_validation(d_data)
    e_data_folds = n_fold_cross_validation(e_data)
    k_data_folds = n_fold_cross_validation(k_data)

    d_domain = k_data_folds
    s_domain = d_data_folds

    for f_idx, __ in enumerate(d_domain):
        d_domain[f_idx][0] = d_domain[f_idx][0][:-labeled_data]
        d_domain[f_idx][0].append(s_domain[f_idx][0][labeled_data:])
        d_domain[f_idx][1] = s_domain[f_idx][1]

    for idx, (train, test) in enumerate(d_domain):
        print("Running Fold {}".format(idx + 1))

        output, matrix = run_boost(train, test, iterations, labeled_data)
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
        print()
        print("Boost Iteration: {}".format(current_itr + 1))
        classifiers.append(DStump())

        # randomly sample the training set with replacement
        classifiers[-1].fit(train)

        # Extract weights and outputs
        weights = [ex.weight for ex in train]
        outputs = [classifiers[-1].classify(ex)[0] != ex.label for ex in train]

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
        sum_conf = 0

        total_class_weight = sum(classifier_weights)

        for classifier_idx, classifier in enumerate(classifiers):
            output, conf = classifier.classify(ex)
            vote += (classifier_weights[classifier_idx] / total_class_weight) * int(
                output
            )

            sum_conf += conf

        # Make the vote discrete
        vote = True if vote >= 0.5 else False

        # Calculate confidence for given outcome
        total_conf = sum_conf / len(classifiers)

        # Calculate outputs and matrix
        outputs.append((vote, total_conf))
        is_correct = "t" if ex.label == output else "f"
        is_positive = "p" if output else "n"
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
    boost()
