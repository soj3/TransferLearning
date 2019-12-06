from copy import deepcopy
import data
import sklearn.decomposition as skd
from sklearn.feature_extraction.text import CountVectorizer
from pivot import *
from classifier import *
from utils import *
from misalignment import *
import numpy as np


LAMBDA = 1e-1
MU = 1e-1
NUM_PIVOTS = 500
NUM_FEATURES = 5000
SVD_DIMENSION = 100
MIN_PIVOT_APPEARANCE = 50


def scl(source, target):
    print("Reading data...")
    source_pos, source_neg, source_un = data.get_reviews(source)
    target_pos, target_neg, target_un = data.get_reviews(target)

    train_source, train_source_labels, test_source, test_source_labels = split_data(source_pos, source_neg)
    train_target, train_target_labels, test_target, test_target_labels = split_data(target_pos, target_neg)

    unlabeled = source_un + target_un + train_source
    train_and_unlabeled = source_un + train_source

    dicts = []
    train_sets = []

    dict1 = CountVectorizer(binary=True, min_df=5)
    x_train = dict1.fit_transform(train_source).toarray()

    dicts.append(dict1)
    train_sets.append(x_train)

    source_dict = CountVectorizer(binary=True, min_df=20)
    x_train_source = source_dict.fit_transform(train_and_unlabeled).toarray()

    dicts.append(source_dict)
    train_sets.append(x_train_source)

    unlabeled_dict = CountVectorizer(binary=True, min_df=40)
    x_train_unlabeled = unlabeled_dict.fit_transform(unlabeled).toarray()

    dicts.append(unlabeled_dict)
    train_sets.append(x_train_unlabeled)

    target_dict = CountVectorizer(binary=True, min_df=20)
    x_train_target = target_dict.fit_transform(target_un).toarray()

    dicts.append(target_dict)
    train_sets.append(x_train_target)

    print("Selecting pivots...")
    pivots, pivot_appearances = select_pivots(dicts, train_sets, train_source_labels)

    # create a binary classifier for each pivot feature on the combined unlabeled data of the source and target

    unlabeled_data = source_unlabeled + target_unlabeled
    binary_data = deepcopy(unlabeled_data[:2500])
    for ex in binary_data:
        for i in range(len(ex.features)):
            if ex.features[i] > 1:
                ex.features[i] = 1
            else:
                ex.features[i] = 0
    merged_vocab = merge_list(source_vocab, target_vocab)
    print("Collecting pivot predictor weights...")
    weights, final_vocab = get_pivot_predictor_weights(binary_data, merged_vocab, pivots, NUM_FEATURES)

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
        ex.create_features(final_vocab)
    for ex in target_labeled:
        ex.create_features(final_vocab)

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



