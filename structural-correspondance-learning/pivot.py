import sklearn.linear_model as model
from information_gain import calc_mutual_info


def select_pivots(labeled_source, unlabeled_source, unlabeled_target, source_vocab, target_vocab, num_pivots):
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


def get_pivot_predictor_weights(data, vocab, pivots, NUM_FEATURES):
    weights = []
    j = 1
    # for each pivot, we create a classifier that predicts the likelihood of that pivot appearing in the example,
    # given all of the other features (i.e. words) of the example
    for pivot in pivots:
        x = []
        y = []
        # remove the pivot from the vocabulary
        keys = [k for k, _ in vocab]
        assert pivot in keys
        temp_vocab = [(k, v) for (k, v) in vocab if k != pivot]
        # Here the class label is 1 or 0 depending on the appearance of the pivot in the example
        # maybe i should change this to -1 because we want the classifier to output a negative number if the
        # pivot is not there?
        for i in range(len(data)):
            if pivot in data[i].words:
                y.append(1)
            else:
                y.append(0)
            data[i].create_features(temp_vocab)
            x.append(data[i].features)
        print("Training pivot predictor", j)
        # train a Stochastic gradient descent classifier using the modified Huber loss function
        classifier = model.SGDClassifier(loss="modified_huber")
        classifier.fit(x, y)
        weight = []
        for i in classifier.coef_[0]:
            weight.append(i)
        assert len(weight) == NUM_FEATURES
        weights.append(weight)
        j += 1
    return weights
