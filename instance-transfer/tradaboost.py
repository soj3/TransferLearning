import math
import random
import numpy as np
import sys
from data import collect_review_data
from example import SentimentExample
from helpers import n_fold_cross_validation
from dstump import DStump
from stats import calculate_aroc, calculate_stats


def boost():
    """
    initialize Boost
    """

    print("Collecting Data")
    data = collect_review_data("kitchen", 100)
    print("Finished Collecting Data")

    iterations = 100

    confused_matrix_bois = []
    confused_output_bois = []

    data_folds = n_fold_cross_validation(data)

    for train, test in data_folds:
        print("Running Fold")

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
        print("Boost Iteration: {}".format(current_itr))
        classifiers.append(DStump())

        # randomly sample the training set with replacement
        classifiers[-1].fit(train)

        # Extract weights and outputs
        weights = [ex.weight for ex in train]
        outputs = [int(classifiers[-1].classify(ex)[0] != ex.label) for ex in train]

        # Calculate classifier error
        error = weight_error(weights, outputs)
        classifier_weights.append(classifier_weight(error))

        if error < 10e-20 or error >= 0.5:
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
    return 0.5 * math.log((1 - error) / error)


def update_weights(weights, output, alpha):
    """
    updated the weights of the data
    """
    updated_weights = []
    norm = 1 / sum(weights)
    for current_itr in range(len(weights)):
        converted_output = 1 if output[current_itr] else -1

        updated_weights.append(
            norm * weights[current_itr] * math.exp(-alpha * converted_output)
        )
    return updated_weights


if __name__ == "__main__":
    boost()
