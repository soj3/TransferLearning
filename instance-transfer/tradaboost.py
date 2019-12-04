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
    percent_same_data = 0.20

    confused_matrix_bois = []
    confused_output_bois = []

    b_data_folds = n_fold_cross_validation(b_data)
    d_data_folds = n_fold_cross_validation(d_data)
    e_data_folds = n_fold_cross_validation(e_data)
    k_data_folds = n_fold_cross_validation(k_data)

    d_domain = k_data_folds
    s_domain = d_data_folds

    for idx in range(len(d_domain)):
        print("Running Fold {}".format(idx + 1))

        d_train = d_domain[idx][0]
        num_same_data = int(len(s_domain[idx][0]) * percent_same_data)
        s_train = s_domain[idx][0][:num_same_data]
        test = s_domain[idx][1]
        print(
            "Training on {} diff domain examples and {} same domain examples".format(
                len(d_train), len(s_train)
            )
        )
        output, matrix = run_boost(d_train, s_train, test, iterations)
        confused_matrix_bois.append(matrix)
        confused_output_bois += output

    calculate_stats(arr_of_confusion=confused_matrix_bois)
    calculate_aroc(arr_of_confidence=confused_output_bois)


def run_boost(d_train, s_train, test, iterations):
    normalize_weights(s_train, d_train, reset=True)

    classifiers = []

    betas = []
    beta = 1 / (1 + math.sqrt(2 * math.log(len(d_train) / iterations)))

    current_itr = 0

    # run for the input number of iterations
    while current_itr < iterations:
        normalize_weights(s_train, d_train)

        print("Boost Iteration: {}".format(current_itr + 1))
        classifiers.append(DStump())

        # randomly sample the training set with replacement
        classifiers[-1].fit(d_train + s_train)

        # Extract weights and outputs
        d_weights = [ex.weight for ex in d_train]
        d_outputs = [[classifiers[-1].classify(ex)[0], ex.label] for ex in d_train]
        s_weights = [ex.weight for ex in s_train]
        s_outputs = [[classifiers[-1].classify(ex)[0], ex.label] for ex in s_train]

        # Calculate classifier error
        error = weight_error(s_weights, s_outputs)

        if error >= 0.5 or error == 0.0:
            del classifiers[-1]
            break

        betas.append(error / (1 - error))

        # Update the weights
        new_diff_weights = update_diff_weights(d_weights, d_outputs, beta)
        new_same_weights = update_same_weights(s_weights, s_outputs, betas)
        for idx, ex in enumerate(d_train):
            ex.weight = new_diff_weights[idx]
        for idx, ex in enumerate(s_train):
            ex.weight = new_same_weights[idx]

        current_itr += 1

    outputs = []
    matrix = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    # Run Testing
    for ex in test:

        vote = 0
        sum_conf = 0

        use_itr = math.ceil(len(classifiers) / 2)

        for idx in range(use_itr, len(classifiers)):
            output, conf = classifiers[idx].classify(ex)
            vote *= betas[idx] ** -output

            sum_conf += conf

        # Make the vote discrete
        evaluated_betas = np.array(betas[use_itr:])
        boundary = np.prod(evaluated_betas ** -0.5)

        vote = True if vote >= boundary else False

        # Calculate confidence for given outcome
        total_conf = sum_conf / len(classifiers)

        # Calculate outputs and matrix
        outputs.append((vote, total_conf))
        is_correct = "t" if ex.label == output else "f"
        is_positive = "p" if output else "n"
        matrix[is_correct + is_positive] += 1

    return outputs, matrix


def normalize_weights(s_train, d_train, reset=False):
    if reset:
        for ex in d_train:
            ex.weight = 1

        for ex in s_train:
            ex.weight = 1

    weight_sum = sum([ex.weight for ex in d_train])
    weight_sum += sum([ex.weight for ex in s_train])

    for ex in d_train:
        ex.weight /= weight_sum

    for ex in s_train:
        ex.weight /= weight_sum


def weight_error(weights, output):
    """
    find the error of the weights
    """
    clean_outputs = np.abs(np.diff(output)).flatten()
    w_sum = np.sum(np.multiply(weights, clean_outputs))
    return w_sum / np.sum(weights)


def update_diff_weights(weights, output, beta):
    """
    updated the weights of the different domain
    """
    output_sub = np.abs(np.diff(output)).flatten()
    updated_weights = np.multiply(weights, beta ** output_sub)

    return updated_weights


def update_same_weights(weights, output, betas):
    """
    updated the weights of the data
    """
    output_sub = np.abs(np.diff(output)).flatten().astype("float") * -1
    updated_weights = np.multiply(weights, betas[-1] ** -output_sub)

    return updated_weights


if __name__ == "__main__":
    boost()
