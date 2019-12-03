import sklearn.linear_model as model
import numpy as np


def evaluate_classifiers(classifiers, test_source, test_source_labels, test_target, test_target_labels):
    pass


def create_classifiers(pivot_matrix, train_source, train_source_labels, train_target, train_target_labels):

    classifiers = []

    # train the baseline classifier, which is a linear model with no adaptation
    print("Training baseline...")
    baseline = model.SGDClassifier(loss="modified_huber")
    baseline.fit(train_source, train_source_labels)
    classifiers.append(("Baseline", baseline))

    # train the source classifier with adaptation
    print("Training source classifier...")
    for ex in train_source:
        adapted_features = np.dot(ex, pivot_matrix)
        ex.extend(adapted_features)
    source = model.SGDClassifier(loss="modified_huber")
    source.fit(train_source, train_source_labels)
    classifiers.append(("Source", source))

    # train the target classifier with adaptation
    print("Training target classifier...")
    for ex in train_target:
        adapted_features = np.dot(ex, pivot_matrix)
        ex.extend(adapted_features)
    target = model.SGDClassifier(loss="modified_huber")
    target.fit(train_target, train_target_labels)
    classifiers.append(("Target", target))

    return classifiers
