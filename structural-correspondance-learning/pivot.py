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
    for i in range(NUM_PIVOTS * 5):

        source_count = sum(train_sets[1][:, dicts[1].get_feature_names().index(sorted_words[i])
                                         if sorted_words[i] in dicts[1].get_feature_names() else 0])
        target_count = sum(train_sets[3][:, dicts[3].get_feature_names().index(sorted_words[i])
                                         if sorted_words[i] in dicts[3].get_feature_names() else 0])

        if source_count > MIN_PIVOT_APPEARANCE and target_count > MIN_PIVOT_APPEARANCE:
            if len(pivots) < NUM_PIVOTS:
                pivots.append(sorted_words[i])
                pivot_appearances.append(dicts[2].get_feature_names().index(sorted_words[i]))
    assert len(pivots) == NUM_PIVOTS
    return pivots, pivot_appearances


def get_pivot_predictor_weights(non_pivot_feature_matrix, pivot_appearance_matrix):
    weights = []
    j = 1
    # for each pivot, we create a classifier that predicts the likelihood of that pivot appearing in the example,
    # given all of the other features (i.e. words) of the example
    for i in range(NUM_PIVOTS):
        print("Training pivot predictor " + str(j) + "...")
        # train a Stochastic gradient descent classifier using the modified Huber loss function
        classifier = model.SGDClassifier(loss="modified_huber")
        classifier.fit(non_pivot_feature_matrix, pivot_appearance_matrix[:, i])
        weight = []
        for l in classifier.coef_[0]:
            weight.append(l)
        weights.append(weight)
        j += 1
    return weights


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
