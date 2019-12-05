import sklearn.linear_model as model
import numpy as np
from copy import deepcopy


def evaluate_classifiers(classifiers, test_source, test_source_labels, test_target, test_target_labels, pivot_matrix):
    adapted = False
    for name, classifier in classifiers:
        if name == "Baseline":
            print("Testing Baseline...")
            source_predictions = classifier.predict(test_source)
            source_correct = 0
            for i in range(len(source_predictions)):
                if source_predictions[i] == test_source_labels[i]:
                    source_correct += 1
            source_acc = source_correct/len(test_source)
            print("Baseline accuracy on source: ", source_acc)

            target_predictions = classifier.predict(test_target)
            target_correct = 0
            for i in range(len(target_predictions)):
                if target_predictions[i] == test_target_labels[i]:
                    target_correct += 1
            target_acc = target_correct / len(test_target)
            print("Baseline accuracy on target: ", target_acc)
        if not adapted:
            for ex in test_source:
                adapted_features = np.dot(ex, pivot_matrix)
                ex.extend(adapted_features)

            for ex in test_target:
                adapted_features = np.dot(ex, pivot_matrix)
                ex.extend(adapted_features)
            adapted = True

        if name == "Source":
            print("Testing Source...")
            source_predictions = classifier.predict(test_source)
            source_correct = 0
            for i in range(len(source_predictions)):
                if source_predictions[i] == test_source_labels[i]:
                    source_correct += 1
            source_acc = source_correct / len(test_source)
            print("Source accuracy on source: ", source_acc)

            target_predictions = classifier.predict(test_target)
            target_correct = 0
            for i in range(len(target_predictions)):
                if target_predictions[i] == test_target_labels[i]:
                    target_correct += 1
            target_acc = target_correct / len(test_target)
            print("Source accuracy on target: ", target_acc)

        if name == "Target":
            print("Testing Target...")
            source_predictions = classifier.predict(test_source)
            source_correct = 0
            for i in range(len(source_predictions)):
                if source_predictions[i] == test_source_labels[i]:
                    source_correct += 1
            source_acc = source_correct / len(test_source)
            print("Target accuracy on source: ", source_acc)

            target_predictions = classifier.predict(test_target)
            target_correct = 0
            for i in range(len(target_predictions)):
                if target_predictions[i] == test_target_labels[i]:
                    target_correct += 1
            target_acc = target_correct / len(test_target)
            print("Target accuracy on target: ", target_acc)

        if name == "Corrected Source":
            print("Testing Corrected source...")
            source_predictions = classifier.predict(test_source)
            source_correct = 0
            for i in range(len(source_predictions)):
                if source_predictions[i] == test_source_labels[i]:
                    source_correct += 1
            source_acc = source_correct / len(test_source)
            print("Corrected source accuracy on source: ", source_acc)

            target_predictions = classifier.predict(test_target)
            target_correct = 0
            for i in range(len(target_predictions)):
                if target_predictions[i] == test_target_labels[i]:
                    target_correct += 1
            target_acc = target_correct / len(test_target)
            print("Corrected source accuracy on target: ", target_acc)

        if name == "Corrected Target":
            print("Testing corrected target...")
            source_predictions = classifier.predict(test_source)
            source_correct = 0
            for i in range(len(source_predictions)):
                if source_predictions[i] == test_source_labels[i]:
                    source_correct += 1
            source_acc = source_correct / len(test_source)
            print("Corrected target accuracy on source: ", source_acc)

            target_predictions = classifier.predict(test_target)
            target_correct = 0
            for i in range(len(target_predictions)):
                if target_predictions[i] == test_target_labels[i]:
                    target_correct += 1
            target_acc = target_correct / len(test_target)
            print("Corrected target accuracy on target: ", target_acc)


def create_classifiers(pivot_matrix, train_source, train_source_labels, train_target, train_target_labels):
    adapt_source = deepcopy(train_source)
    adapt_target = deepcopy(train_target)
    classifiers = []

    # train the baseline classifier, which is a linear model with no adaptation
    print("Training baseline...")
    baseline = model.SGDClassifier(loss="modified_huber")
    baseline.fit(train_source, train_source_labels)
    classifiers.append(("Baseline", baseline))

    # train the source classifier with adaptation
    print("Training source classifier...")
    for ex in adapt_source:
        adapted_features = np.dot(ex, pivot_matrix)
        ex.extend(adapted_features)
    source = model.SGDClassifier(loss="modified_huber")
    source.fit(adapt_source, train_source_labels)
    classifiers.append(("Source", source))

    # train the target classifier with adaptation
    print("Training target classifier...")
    for ex in adapt_target:
        adapted_features = np.dot(ex, pivot_matrix)
        ex.extend(adapted_features)
    target = model.SGDClassifier(loss="modified_huber")
    target.fit(adapt_target, train_target_labels)
    classifiers.append(("Target", target))

    return classifiers


class LogRegClassifier:
    def __init__(self, weights):
        self.weights = weights

    def predict(self, features):
        predict_array = []
        for feature in features:
            if np.dot(feature, self.weights) > 0:
                predict_array.append(-1)
            else:
                predict_array.append(1)
        return predict_array
