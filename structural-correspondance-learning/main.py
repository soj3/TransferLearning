import data
import argparse as argp
import sklearn.linear_model as model
import sklearn.decomposition as skd
from information_gain import calc_mutual_info
import numpy as np


LAMBDA = 1e-3
MU = 1e-1


def source_loss():
    pass


def correct_misalignments():
    pass


def alpha_dist():
    pass


def select_pivots(labeled_source, unlabeled_source, unlabeled_target, source_vocab, target_vocab, num_pivots=500):
    # want to choose the num_pivots features with the highest mutual information gain to the source label
    # sort the features according to how many times they occur in both the source and target domains
    # then, choose the num_pivots features with the highest mutual info to the source label

    # criteria for pivots: occurs more than 50 times, occurs in more than 5 examples, occurs in both domains
    pivots = []
    potential_pivots = []
    # from the unlabeled source and unlabeled target data, find features that fulfill these criteria
    new_dict1 = {k: v for (k, v) in source_vocab if v > 50}
    new_dict2 = {k: v for (k, v) in target_vocab if v > 50}
    for key in new_dict1.keys():
        if key in new_dict2.keys():
            num_occ1, num_occ2 = 0, 0
            for example in unlabeled_source:
                if key in example.words.keys():
                    num_occ1 += 1
            for example in unlabeled_target:
                if key in example.words.keys():
                    num_occ2 += 1
            if num_occ1 > 5 and num_occ2 > 5:
                potential_pivots.append(key)

    # create a dictionary containing the potential pivot features and their corresponding info gain to source
    info = {}
    for feature in potential_pivots:
        info[feature] = calc_mutual_info(labeled_source, feature)
    # sort according to mutual information
    sorted_info = sorted(info.items(), key=lambda item: item[1], reverse=True)

    # add top num_pivots to pivot list
    for i in range(num_pivots):
        pivots.append(sorted_info[i][0])

    return pivots


def get_pivot_predictor_weights(data, vocab, pivots):
    weights = []
    i = 1
    for pivot in pivots:
        print(pivot)
        x = []
        y = []
        temp_vocab = [(k,v) for (k,v) in vocab if k != pivot]
        for i in range(len(data)):
            if pivot in data[i].words:
                y.append(1)
            else:
                y.append(0)
            data[i].create_features(temp_vocab)
            x.append(data[i].features)
        print("Training pivot predictor ", i)
        classifier = model.SGDClassifier(loss="modified_huber")
        classifier.fit(x, y)
        #print(classifier.coef_)
        weights.append(classifier.coef_)
        i +=1
    return weights


def scl(source, target):
    print("Reading data...")
    source_labeled, source_unlabeled, source_vocab = data.collect_review_data(source)
    _, target_unlabeled, target_vocab = data.collect_review_data(target)
    print("Selecting pivots...")
    pivots = select_pivots(source_labeled, source_unlabeled, target_unlabeled, source_vocab, target_vocab)

    # create a binary classifier for each pivot feature on the combined unlabeled data of the source and target
    unlabeled_data = source_unlabeled + target_unlabeled
    merged_vocab = merge_list(source_vocab, target_vocab)
    print("Collecting pivot predictor weights...")
    weights = get_pivot_predictor_weights(unlabeled_data, merged_vocab[:500], pivots)
    weights = np.asmatrix(weights).transpose()
    # compute the Singular value decomposition of the weights matrix
    svd = skd.TruncatedSVD(n_components=25)
    pivot_matrix = svd.fit_transform(weights)
    print(pivot_matrix)

def main():
    #ap = argp.ArgumentParser()
    #ap.add_argument("-s", "--source", required=True, help="Source domain")
    #ap.add_argument("-t", "--target", required=True, help="Target domain")
    #args = vars(ap.parse_args())

    scl("books", "dvd")
    # split source and target datasets into training and testing data
    #baseline classifier


    #classifier trained in domain

    #classifiers trained on new domains



def merge_list(list1, list2):
    dict1 = dict(list1)
    dict2 = dict(list2)
    dict3 = {**dict1, **dict2}
    for key in dict3.keys():
        if key in dict1.keys() and key in dict2.keys():
            dict3[key] = dict1[key] + dict2[key]
    return list(dict3.items())



if __name__ == "__main__":
    main()



