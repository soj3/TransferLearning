from copy import deepcopy
import data
import sklearn.decomposition as skd
from pivot import *
from classifier import *
from utils import *
from misalignment import *
import numpy as np


LAMBDA = 1e-1
MU = 1e-1
NUM_PIVOTS = 500
SVD_DIMENSION = 50
MIN_PIVOT_APPEARANCE = 50


def scl(source, target):
    print("Reading data...")
    source_pos, source_neg, source_un = data.get_reviews(source)
    target_pos, target_neg, target_un = data.get_reviews(target)

    train_source, train_source_labels, test_source, test_source_labels = split_data(source_pos, source_neg)
    _, _, test_target, test_target_labels = split_data(target_pos, target_neg)
    test_data_target = target_pos + target_neg
    unlabeled = source_un + target_un + train_source
    train_and_unlabeled = source_un + train_source

    dicts, train_sets = get_dicts_and_train_sets(train_source, train_and_unlabeled, unlabeled, target_un)

    print("Selecting pivots...")
    pivots, pivot_appearances = select_pivots(dicts, train_sets, train_source_labels)

    # create a binary classifier for each pivot feature on the combined unlabeled data of the source and target
    print("Shaping data for pivot predictors...")
    num_source_examples = int(len(train_and_unlabeled)/5)
    num_target_examples = int(len(target_un)/5)

    pivot_appearance_matrix = train_sets[2][:, pivot_appearances]
    non_pivot_feature_matrix = np.delete(train_sets[2], pivot_appearances, 1)

    # we don't want to use all the examples because that would take a long time, so we just take a subset

    non_pivot_feature_matrix = non_pivot_feature_matrix[num_source_examples:-num_target_examples][:]

    pivot_appearance_matrix = pivot_appearance_matrix[num_source_examples:-num_target_examples]

    print("Collecting pivot predictor weights...")
    weights = get_pivot_predictor_weights(non_pivot_feature_matrix, pivot_appearance_matrix)

    # compute the Singular value decomposition of the weights matrix
    print("Calculating SVD...")
    row_len = len(weights[0])
    for weight in weights:
        assert len(weight) == row_len
    weights = np.asmatrix(weights, dtype=float).transpose()
    svd = skd.TruncatedSVD(n_components=SVD_DIMENSION)
    pivot_matrix = svd.fit_transform(weights)

    print("Training classifiers...")

    classifiers = create_classifiers(pivot_matrix, pivot_appearances, dicts, train_sets, train_source,
                                     train_source_labels)

    # for (k, v) in classifiers:
    #     if k == "Source":
    #         new_classifier = correct_misalignments(v, train_target[:50], train_target_labels[:50], pivot_matrix)
    #         classifiers.append(("Corrected Source", new_classifier))

    evaluate_classifiers(pivot_matrix, pivot_appearances, dicts, train_sets, test_source, test_source_labels,
                         test_target, test_target_labels, classifiers)


def main():
    # ap = argp.ArgumentParser()
    # ap.add_argument("-s", "--source", required=True, help="Source domain")
    # ap.add_argument("-t", "--target", required=True, help="Target domain")
    # args = vars(ap.parse_args())

    #scl("books", "dvd")
    scl("books", "kitchen")
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



