import sklearn.linear_model as model
from sklearn.metrics import mutual_info_score
from main import NUM_PIVOTS, MIN_PIVOT_APPEARANCE


def select_pivots(dicts, train_sets, labels):
    # want to choose the num_pivots features with the highest mutual information gain to the source label

    # order of dicts is train_source, source, unlabeled, target
    # criteria for pivots: occurs more than 50 times, occurs in more than 5 examples, occurs in both domains

    # sort the dicts words by their mutual info gain to the label
    sorted_words = sort_pivots(dicts[0], train_sets[0], labels)

    # get the pivots with the highest info gain that appear at minimum MIN_PIVOT_APPEARANCE times
    pivots = []
    pivot_appearances = []
    for i in range(NUM_PIVOTS):

        source_count = sum(train_sets[1][:, dicts[1].get_feature_names().index(sorted_words[i])
                                         if sorted_words[i] in dicts[1].get_feature_names() else 0])
        target_count = sum(train_sets[3][:, dicts[3].get_feature_names().index(sorted_words[i])
                                         if sorted_words[i] in dicts[3].get_feature_names() else 0])

        if source_count > MIN_PIVOT_APPEARANCE and target_count > MIN_PIVOT_APPEARANCE:
            pivots.append(sorted_words[i])
            pivot_appearances.append(dicts[2].get_feature_names().index(sorted_words[i]))

    return pivots, pivot_appearances


def get_pivot_predictor_weights(data, vocab, pivots, NUM_FEATURES):
    weights = []
    j = 1
    # for each pivot, we create a classifier that predicts the likelihood of that pivot appearing in the example,
    # given all of the other features (i.e. words) of the example
    pivot_labels = []
    # remove pivots from vocab, but retain whether or not they occurred in each example
    for pivot in pivots:
        y = []
        vocab = [(k, v) for (k,v) in vocab if k != pivot]
        for i in range(len(data)):
            if pivot in data[i].words:
                y.append(1)
            else:
                y.append(-1)
        pivot_labels.append(y)

    for i in range(len(pivots)):
        keys = [k for k, _ in vocab]
        assert pivots[i] not in keys
        x = []
        labels = pivot_labels[i]
        # Here the class label is 1 or 0 depending on the appearance of the pivot in the example
        # maybe i should change this to -1 because we want the classifier to output a negative number if the
        # pivot is not there?
        for i in range(len(data)):
            data[i].create_features(vocab[:NUM_FEATURES])
            x.append(data[i].features)
        print("Training pivot predictor", j)
        # train a Stochastic gradient descent classifier using the modified Huber loss function
        classifier = model.SGDClassifier(loss="modified_huber")
        classifier.fit(x, labels)
        weight = []
        for l in classifier.coef_[0]:
            weight.append(l)
        assert len(weight) == NUM_FEATURES
        weights.append(weight)
        j += 1
    return weights, vocab[:NUM_FEATURES]


def sort_pivots(dict_words, data, labels):
    print("Sorting potential pivots...")
    sorted_pivots = []
    num_features = data.shape[1]
    info_scores = []
    for i in range(num_features):
        info = mutual_info_score(data[:,i], labels)
        info_scores.append(info)

    info_scores_sorted = sorted(range(len(info_scores)), key=lambda i:info_scores[i], reverse=True)
    for i in range(num_features):
        sorted_pivots.append(dict_words.get_feature_names()[info_scores_sorted[i]])

    return sorted_pivots
