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

    d_domains = [k_data_folds, e_data_folds, b_data_folds]
    s_domain = d_data_folds

    for idx in range(len(s_domain)):
        print("Running Fold {}".format(idx + 1))

        d_train_domains = []
        for d_domain in d_domains:
            d_train_domains.append(d_domain[idx][0])

        num_same_data = int(len(s_domain[idx][0]) * percent_same_data)
        s_train = s_domain[idx][0][:num_same_data]

        test = s_domain[idx][1]

        print(
            "Training on {} diff domains and {} same domain examples".format(
                len(d_train_domains), len(s_train)
            )
        )
        output, matrix = run_boost(d_train_domains, s_train, test, iterations)
        confused_matrix_bois.append(matrix)
        confused_output_bois += output

    calculate_stats(arr_of_confusion=confused_matrix_bois)
    calculate_aroc(arr_of_confidence=confused_output_bois)


def run_boost(d_train_domains, s_train, test, iterations):
    normalize_weights(d_train_domains, s_train, reset=True)

    classifiers = []

    alphas = []
    diff_exs = sum([len(d_train) for d_train in d_train_domains])
    alpha = 0.5 * math.log(1 + math.sqrt(2 * math.log(diff_exs / iterations)))

    current_itr = 0

    # run for the input number of iterations
    while current_itr < iterations:
        print("Boost Iteration: {}".format(current_itr + 1))

        normalize_weights(d_train_domains, s_train)

        # Find best domain
        best_domain_classifier = None
        best_error = 1
        for d_train in d_train_domains:
            clf = DStump()
            clf.fit(d_train + s_train)

            # Extract weights and outputs
            s_weights = [ex.weight for ex in s_train]
            s_outputs = [[clf.classify(ex)[0], ex.label] for ex in s_train]

            # Calculate classifier error
            error = weight_error(s_weights, s_outputs)

            if error < best_error:
                best_error = error
                best_domain_classifier = clf

        classifiers.append(clf)

        if best_error >= 0.5 or best_error == 0.0:
            del classifiers[-1]
            break

        alphas.append(0.5 * math.log((1 - error) / error))

        # Update the weights
        for d_train in d_train_domains:
            d_weights = [ex.weight for ex in d_train]
            d_outputs = [[classifiers[-1].classify(ex)[0], ex.label] for ex in d_train]

            new_diff_weights = update_diff_weights(d_weights, d_outputs, alpha)
            for idx, ex in enumerate(d_train):
                ex.weight = new_diff_weights[idx]

        new_same_weights = update_same_weights(s_weights, s_outputs, alphas)
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
            vote += alphas[idx] * output

            sum_conf += conf

        # Make the vote discrete
        vote = True if vote >= 0 else False

        # Calculate confidence for given outcome
        total_conf = sum_conf / len(classifiers)

        # Calculate outputs and matrix
        outputs.append((vote, total_conf))
        is_correct = "t" if ex.label == output else "f"
        is_positive = "p" if output else "n"
        matrix[is_correct + is_positive] += 1

    return outputs, matrix


def normalize_weights(d_train_domains, s_train, reset=False):
    """
    Normalizes all the training weights
    """
    if reset:
        for d_train in d_train_domains:
            for ex in d_train:
                ex.weight = 1

        for ex in s_train:
            ex.weight = 1

    weight_sum = sum([ex.weight for ex in s_train])

    for d_train in d_train_domains:
        for ex in d_train:
            weight_sum += ex.weight

    for d_train in d_train_domains:
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


def update_diff_weights(weights, output, alpha):
    """
    updated the weights of the different domain
    """
    output_sub = np.abs(np.diff(output)).flatten()
    updated_weights = np.multiply(weights, np.exp(output_sub * -alpha))

    return updated_weights


def update_same_weights(weights, output, alphas):
    """
    updated the weights of the data
    """
    output_sub = np.abs(np.diff(output)).flatten().astype("float") * -1
    updated_weights = np.multiply(weights, np.exp(output_sub * alphas[-1]))

    return updated_weights


if __name__ == "__main__":
    boost()
