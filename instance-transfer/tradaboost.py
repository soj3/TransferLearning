import math
import random
import numpy as np
import sys
from data import collect_review_data
from example import Example
from helpers import n_fold_cross_validation
from sklearn.naive_bayes import MultinomialNB
from stats import calculate_aroc, calculate_stats
import matplotlib.pyplot as plt
from typing import List
import argparse


def boost(iterations, percent, features):
    """
    initialize Boost
    """

    print("Collecting Data")
    b_data, d_data, e_data, k_data = collect_review_data(features)
    print("Finished Collecting Data")

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
        num_same_data = int(len(s_domain[idx][0]) * percent)
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
        classifiers.append(MultinomialNB())

        # randomly sample the training set with replacement
        d_ftr, d_lbls, d_wghts = extract_ex_info(d_train)
        s_ftr, s_lbls, s_wghts = extract_ex_info(s_train)
        classifiers[-1].fit(d_ftr + s_ftr, d_lbls + s_lbls)

        # Extract weights and outputs
        d_outputs = np.array(classifiers[-1].predict(d_ftr)) != d_lbls
        s_outputs = np.array(classifiers[-1].predict(s_ftr)) != s_lbls

        # Calculate classifier error
        error = weight_error(s_wghts, s_outputs)

        if error >= 0.5 or error == 0.0:
            del classifiers[-1]
            break

        betas.append(error / (1 - error))

        # Update the weights
        new_diff_weights = update_diff_weights(d_wghts, d_outputs, beta)
        new_same_weights = update_same_weights(s_wghts, s_outputs, betas)
        for idx, ex in enumerate(d_train):
            ex.weight = new_diff_weights[idx]
        for idx, ex in enumerate(s_train):
            ex.weight = new_same_weights[idx]

        current_itr += 1

    outputs = []
    matrix = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    # Run Testing
    for ex in test:

        vote = 1.0
        sum_conf = 0

        use_itr = math.floor(len(classifiers) / 2)

        for idx in range(use_itr, len(classifiers)):
            probs = classifiers[idx].predict_proba([ex.features])[0]
            output = int(probs[0] < probs[1])

            vote *= betas[idx] ** -output

            sum_conf += max(probs)

        # Make the vote discrete
        evaluated_betas = np.array(betas[use_itr:])
        boundary = np.prod(evaluated_betas ** -0.5)
        vote = True if vote >= boundary else False

        # Calculate confidence for given outcome
        total_conf = sum_conf / len(classifiers[use_itr:])

        # Calculate outputs and matrix
        outputs.append((vote, total_conf))
        is_correct = "t" if ex.label == vote else "f"
        is_positive = "p" if vote else "n"
        matrix[is_correct + is_positive] += 1

    return outputs, matrix


def extract_ex_info(examples: List[Example]):
    labels = [ex.label for ex in examples]
    features = [ex.features for ex in examples]
    weights = [ex.weight for ex in examples]

    return features, labels, weights


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
    w_sum = np.sum(np.multiply(weights, output))
    return w_sum / np.sum(weights)


def update_diff_weights(weights, output, beta):
    """
    updated the weights of the different domain
    """
    updated_weights = np.multiply(weights, beta ** output)
    return updated_weights


def update_same_weights(weights, output, betas):
    """
    updated the weights of the data
    """
    updated_weights = np.multiply(weights, betas[-1] ** np.invert(output))
    return updated_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--iterations", help="Number of iterations to run boosting", type=int
    )
    parser.add_argument(
        "-p", "--percent", help="Percent of target data to use in training", type=float
    )
    parser.add_argument(
        "-f", "--features", help="Number of features the vocab should use", type=int
    )

    args = parser.parse_args()

    if args.iterations:
        iterations = args.iterations
    else:
        iterations = 20

    if args.percent:
        percent = args.percent
    else:
        percent = 0.2

    if args.features:
        features = args.features
    else:
        features = 500

    boost(iterations, percent, features)
