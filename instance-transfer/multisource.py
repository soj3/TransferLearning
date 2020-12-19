import math
import numpy as np
from data import *
from example import Example
from sklearn.naive_bayes import MultinomialNB
from helpers import n_fold_cross_validation
from stats import calculate_aroc, calculate_stats
from typing import List
import argparse


def boost(iterations, percent, features):
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
    sp1, sp2, sp3 = collect_spam_a_data(features)
    sp1_data_folds = n_fold_cross_validation(sp1)
    sp2_data_folds = n_fold_cross_validation(sp2)
    sp3_data_folds = n_fold_cross_validation(sp3)

    # Spam Task B Data
    # sps15 = collect_spam_b_data(features)
    # sps15_folds = [n_fold_cross_validation(f) for f in sps15]

    # Spam News Group
    # nws1, nws2 = collect_newsgroup_data(features)
    # nws1_data_folds = n_fold_cross_validation(nws1)
    # nws2_data_folds = n_fold_cross_validation(nws2)

    print("Finished Collecting Data")

    confused_matrix_bois = []
    confused_output_bois = []

    # Domains
    d_domains = [sp1_data_folds, sp2_data_folds]
    s_domain = sp3_data_folds

    for idx in range(len(s_domain)):
        print("Running Fold {}".format(idx + 1))

        d_train_domains = []
        for d_domain in d_domains:
            d_train_domains.append(d_domain[idx][0])

        num_same_data = int(len(s_domain[idx][0]) * percent)
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

    linear = (len(s_train) * 100) / diff_exs
    squared = ((len(s_train) ** 2) * 100) / diff_exs

    multi_param = 1
    exp_param = 1

    current_itr = 0

    # run for the input number of iterations
    while current_itr < iterations:
        print("Boost Iteration: {}".format(current_itr + 1))

        normalize_weights(d_train_domains, s_train)

        # Extract Info from same distribution
        s_ftr, s_lbls, s_wghts = extract_ex_info(s_train)

        # Find best domain
        best_domain_classifier = None
        best_error = 1
        for d_train in d_train_domains:
            d_ftr, d_lbls, d_wghts = extract_ex_info(d_train)

            clf = MultinomialNB()
            clf.fit(d_ftr + s_ftr, d_lbls + s_lbls)

            # Extract weights and outputs
            s_weights = [ex.weight for ex in s_train]
            s_outputs = np.array(clf.predict(s_ftr)) != s_lbls

            # Calculate classifier error
            error = weight_error(s_wghts, s_outputs)

            if error < best_error:
                best_error = error
                best_domain_classifier = clf

        classifiers.append(best_domain_classifier)

        if best_error >= 0.5 or best_error == 0.0:
            del classifiers[-1]
            break

        alphas.append(0.5 * math.log((1 - best_error) / best_error))

        # Update the weights for diff domains
        for d_train in d_train_domains:
            d_ftr, d_lbls, d_wghts = extract_ex_info(d_train)
            d_outputs = np.array(classifiers[-1].predict(d_ftr)) != d_lbls

            new_diff_weights = update_diff_weights(
                d_wghts, d_outputs, alpha, multi_param, exp_param
            )
            for idx, ex in enumerate(d_train):
                ex.weight = new_diff_weights[idx]

        # Update the weights for the same domain
        s_outputs = np.array(classifiers[-1].predict(s_ftr)) != s_lbls

        new_same_weights = update_same_weights(
            s_wghts, s_outputs, alphas, multi_param, exp_param
        )
        for idx, ex in enumerate(s_train):
            ex.weight = new_same_weights[idx]

        current_itr += 1

    outputs = []
    matrix = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    # Run Testing
    for ex in test:

        vote = 0
        conf = 0

        for idx in range(len(classifiers)):
            probs = classifiers[idx].predict_proba([ex.features])[0]
            output = int(probs[0] < probs[1])

            vote += alphas[idx] * output

            conf += alphas[idx] * probs[1]

        # Make the vote discrete
        vote = True if vote >= 0.5 else False

        # Calculate outputs and matrix
        outputs.append((ex.label, conf))
        is_correct = "t" if ex.label == vote else "f"
        is_positive = "p" if vote else "n"
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


def extract_ex_info(examples: List[Example]):
    labels = [ex.label for ex in examples]
    features = [ex.features for ex in examples]
    weights = [ex.weight for ex in examples]

    return features, labels, weights


def weight_error(weights, output):
    """
    find the error of the weights
    """
    w_sum = np.sum(np.multiply(weights, output))
    return w_sum / np.sum(weights)


def update_diff_weights(weights, output, alpha, multi_param, exp_param):
    """
    updated the weights of the different domain
    """
    updated_weights = multi_param * np.multiply(
        weights, np.exp((-2 * exp_param * alpha) * output)
    )

    return updated_weights


def update_same_weights(weights, output, alphas, multi_param, exp_param):
    """
    updated the weights of the data
    """
    updated_weights = multi_param * np.multiply(
        weights, np.exp(output * (2 * exp_param * alphas[-1]))
    )

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

