from sklearn.linear_model import SGDClassifier
import numpy as np
from copy import deepcopy


def correct_misalignments(base_classifier, pivot_matrix, target_pos, target_neg, pivot_appearances, dicts):
    print("Correcting misalignments...")

    weights = base_classifier.coef_[0]
    temp_weights = deepcopy(weights)
    train_examples = []
    train_example_labels = []

    examples_pos = deepcopy(target_pos)
    examples_neg = deepcopy(target_neg)

    pos_proportion = len(examples_pos)/(len(examples_pos) + len(examples_neg))
    neg_proportion = 1-pos_proportion
    for i in range(int(pos_proportion * 50)):
        train_examples.append(examples_pos.pop())
        train_example_labels.append(1)
    for i in range(int(neg_proportion * 50)):
        train_examples.append(examples_neg.pop())
        train_example_labels.append(-1)

    feature_dict = dicts[2]
    train_dict = dicts[0]
    train_examples_reshaped = train_dict.transform(train_examples).toarray()
    train_examples_unlabeled = feature_dict.transform(train_examples).toarray()
    train_examples_for_adaptation = np.delete(train_examples_unlabeled, pivot_appearances, 1)
    adapted_train_examples = np.dot(train_examples_for_adaptation, pivot_matrix)

    combined_train_examples = np.concatenate((train_examples_reshaped, adapted_train_examples), 1)

    temp_classifier = SGDClassifier(loss="modified_huber")
    temp_classifier.fit(combined_train_examples, train_example_labels, coef_init=temp_weights)

    return temp_classifier
