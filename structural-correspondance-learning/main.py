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
NUM_PIVOTS = 50
SVD_DIMENSION = 25
MIN_PIVOT_APPEARANCE = 10
NUM_FOLDS = 5


def scl(source, target):
    print("Reading data...")
    source_pos, source_neg, source_un = data.get_reviews(source)
    target_pos, target_neg, target_un = data.get_reviews(target)

    baseline_source_accs = []
    baseline_target_accs = []
    baseline_source_rocs = []
    baseline_target_rocs = []
    adapted_source_accs = []
    adapted_target_accs = []
    adapted_source_rocs = []
    adapted_target_rocs = []
    corrected_source_accs = []
    corrected_target_accs = []
    corrected_source_rocs = []
    corrected_target_rocs = []
    data_folds, data_fold_labels = split_data(source_pos, source_neg, NUM_FOLDS, 200)
    for i in range(len(data_folds)):
        print('\033[1m' + "Training fold " + str(i + 1) + "..." + '\033[0m')
        test_source = data_folds[i]
        test_source_labels = data_fold_labels[i]
        train_source = []
        train_source_labels = []
        for j in range(len(data_folds)):
            if i != j:
                train_source.extend(data_folds[j])
                train_source_labels.extend(data_fold_labels[j])
        test_target_set, test_target_labels_set = split_data(target_pos, target_neg, 1, 400)
        test_target = test_target_set[0]
        test_target_labels = test_target_labels_set[0]
        unlabeled = source_un + target_un + train_source + test_source
        train_and_unlabeled = source_un + train_source + test_source

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
        # convert the 0's to -1's so that huber loss actually works
        pivot_appearance_matrix_for_huber = np.where(pivot_appearance_matrix == 0, -1, pivot_appearance_matrix)

        print("Collecting pivot predictor weights...")
        weights = get_pivot_predictor_weights(non_pivot_feature_matrix, pivot_appearance_matrix_for_huber)

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

        for (k, v) in classifiers:
            if k == "Source":
                new_classifier = correct_misalignments(v, pivot_matrix, target_pos, target_neg, pivot_appearances,
                                                       dicts)
                classifiers.append(("Corrected Source", new_classifier))

        # fold_source_acc[0] is the baseline accuracy on source, fold_source_acc[1] is the adapted acc on source, and 2
        # is corrected source acc. same goes for target
        fold_source_acc, fold_target_acc, source_roc, target_roc = evaluate_classifiers(pivot_matrix, pivot_appearances,
                                                                                       dicts, train_sets, test_source,
                                                                                       test_source_labels, test_target,
                                                                                       test_target_labels, classifiers)
        baseline_source_accs.append(fold_source_acc[0])
        adapted_source_accs.append(fold_source_acc[1])
        corrected_source_accs.append(fold_source_acc[2])
        baseline_target_accs.append(fold_target_acc[0])
        adapted_target_accs.append(fold_target_acc[1])
        corrected_target_accs.append(fold_target_acc[2])
        baseline_source_rocs.append(source_roc[0])
        adapted_source_rocs.append(source_roc[1])
        corrected_source_rocs.append(source_roc[2])
        baseline_target_rocs.append(target_roc[0])
        adapted_target_rocs.append(target_roc[1])
        corrected_target_rocs.append(target_roc[2])



    avg_baseline_source_acc = np.average(baseline_source_accs)
    avg_adapted_source_acc = np.average(adapted_source_accs)
    avg_corrected_source_acc = np.average(corrected_source_accs)
    avg_baseline_target_acc = np.average(baseline_target_accs)
    avg_adapted_target_acc = np.average(adapted_target_accs)
    avg_corrected_target_acc = np.average(corrected_target_accs)
    std_baseline_source_acc = np.std(baseline_source_accs)
    std_adapted_source_acc = np.std(adapted_source_accs)
    std_corrected_source_acc = np.std(corrected_source_accs)
    std_baseline_target_acc = np.std(baseline_target_accs)
    std_adapted_target_acc = np.std(adapted_target_accs)
    std_corrected_target_acc = np.std(corrected_target_accs)
    avg_baseline_source_roc = np.average(baseline_source_rocs)
    avg_baseline_target_roc = np.average(baseline_target_rocs)
    avg_adapted_source_roc = np.average(adapted_source_rocs)
    avg_adapted_target_roc = np.average(adapted_target_rocs)
    avg_corrected_source_roc = np.average(corrected_source_rocs)
    avg_corrected_target_roc = np.average(corrected_target_rocs)

    print("Average baseline source acc is " + str(round(avg_baseline_source_acc,3)) + " " + u'\xb1' + " " +
          str(round(std_baseline_source_acc, 2)) + ", ROC is " + avg_baseline_source_roc)
    print("Average baseline target acc is " + str(round(avg_baseline_target_acc,3)) + " " + u'\xb1' + " " +
          str(round(std_baseline_target_acc, 2)) + ", ROC is " + avg_baseline_target_roc)
    print("Average adapted source acc is " + str(round(avg_adapted_source_acc,3)) + " " + u'\xb1' + " " +
          str(round(std_adapted_source_acc, 2)) + ", ROC is " + avg_adapted_source_roc)
    print("Average adapted target acc is " + str(round(avg_adapted_target_acc,3)) + " " + u'\xb1' + " " +
          str(round(std_adapted_target_acc, 2)) + ", ROC is " + avg_adapted_target_roc)
    print("Average corrected source acc is " + str(round(avg_corrected_source_acc, 3)) + " " + u'\xb1' + " " +
          str(round(std_corrected_source_acc, 2)) + ", ROC is " + avg_corrected_source_roc)
    print("Average corrected target acc is " + str(round(avg_corrected_target_acc, 3)) + " " + u'\xb1' + " " +
          str(round(std_corrected_target_acc, 2)) + ", ROC is " + avg_corrected_target_roc)

    print("Calculating A-distance...")

    a_dist = get_a_dist(pivot_matrix, pivot_appearances, dicts, source_un, target_un)

    print("A-distance between " + source + " and " + target + " is " + str(a_dist))


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



