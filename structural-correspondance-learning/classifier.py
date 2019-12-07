def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import sklearn.linear_model as model
from sklearn.metrics import roc_auc_score
import numpy as np


def evaluate_classifiers(pivot_matrix, pivot_appearances, dicts, train_sets, test_source, test_source_labels,
                         test_target, test_target_labels, classifiers):

    # order of dicts is train_source, source, unlabeled, target
    test_dict = dicts[0]
    test_unlabeled_dict = dicts[2]
    target_examples = test_dict.transform(test_target).toarray()
    target_examples_unlabeled = test_unlabeled_dict.transform(test_target).toarray()
    target_examples_for_adaptation = np.delete(target_examples_unlabeled, pivot_appearances, 1)
    adapted_target_examples = target_examples_for_adaptation.dot(pivot_matrix)

    adapted_test_target = np.concatenate((target_examples, adapted_target_examples), 1)

    source_examples = test_dict.transform(test_source).toarray()
    source_examples_unlabeled = test_unlabeled_dict.transform(test_source).toarray()
    source_examples_for_adaptation = np.delete(source_examples_unlabeled, pivot_appearances, 1)
    adapted_source_examples = source_examples_for_adaptation.dot(pivot_matrix)

    adapted_test_source = np.concatenate((source_examples, adapted_source_examples), 1)
    source_acc = []
    target_acc = []
    source_rocs = []
    target_rocs = []
    for name, classifier in classifiers:

        if name == "Baseline":
            print("Testing Baseline...")
            source_predictions = classifier.predict(source_examples)
            baseline_source_score = classifier.decision_function(source_examples)
            source_correct = 0
            for i in range(len(source_predictions)):
                if source_predictions[i] == test_source_labels[i]:
                    source_correct += 1
            baseline_source_acc = source_correct/len(source_examples)
            print("Baseline accuracy on source: ", baseline_source_acc)
            source_acc.append(baseline_source_acc)
            baseline_source_roc = roc_auc_score(test_source_labels, baseline_source_score)
            source_rocs.append(baseline_source_roc)

            target_predictions = classifier.predict(target_examples)
            baseline_target_score = classifier.decision_function(target_examples)
            target_correct = 0
            for i in range(len(target_predictions)):
                if target_predictions[i] == test_target_labels[i]:
                    target_correct += 1
            baseline_target_acc = target_correct / len(target_examples)
            print("Baseline accuracy on target: ", baseline_target_acc)
            target_acc.append(baseline_target_acc)
            baseline_target_roc = roc_auc_score(test_target_labels, baseline_target_score)
            target_rocs.append(baseline_target_roc)

        if name == "Source":
            print("Testing Source...")
            source_predictions = classifier.predict(adapted_test_source)
            adapted_source_score = classifier.decision_function(adapted_test_source)
            source_correct = 0
            for i in range(len(source_predictions)):
                if source_predictions[i] == test_source_labels[i]:
                    source_correct += 1
            adapted_source_acc = source_correct / len(adapted_test_source)
            print("Source accuracy on source: ", adapted_source_acc)
            source_acc.append(adapted_source_acc)
            adapted_source_roc = roc_auc_score(test_source_labels, adapted_source_score)
            source_rocs.append(adapted_source_roc)

            target_predictions = classifier.predict(adapted_test_target)
            adapted_target_score = classifier.decision_function(adapted_target_examples)
            target_correct = 0
            for i in range(len(target_predictions)):
                if target_predictions[i] == test_target_labels[i]:
                    target_correct += 1
            adapted_target_acc = target_correct / len(adapted_test_target)
            print("Source accuracy on target: ", adapted_target_acc)
            target_acc.append(adapted_target_acc)
            adapted_target_roc = roc_auc_score(test_target_labels, adapted_target_score)
            target_rocs.append(adapted_target_roc)

        if name == "Corrected Source":
            print("Testing Corrected source...")
            source_predictions = classifier.predict(adapted_test_source)
            corrected_source_score = classifier.decision_function(adapted_source_examples)
            source_correct = 0
            for i in range(len(source_predictions)):
                if source_predictions[i] == test_source_labels[i]:
                    source_correct += 1
            corrected_source_acc = source_correct / len(adapted_test_source)
            print("Corrected source accuracy on source: ", corrected_source_acc)
            source_acc.append(corrected_source_acc)
            corrected_source_roc = roc_auc_score(test_source_labels, corrected_source_score)
            source_rocs.append(corrected_source_roc)

            target_predictions = classifier.predict(adapted_test_target)
            corrected_target_score = classifier.decision_function(adapted_target_examples)
            target_correct = 0
            for i in range(len(target_predictions)):
                if target_predictions[i] == test_target_labels[i]:
                    target_correct += 1
            corrected_target_acc = target_correct / len(adapted_test_target)
            print("Corrected source accuracy on target: ", corrected_target_acc)
            target_acc.append(corrected_target_acc)
            corrected_target_roc = roc_auc_score(test_target_labels, corrected_target_score)
            target_rocs.append(corrected_target_roc)

    return source_acc, target_acc, source_rocs, target_rocs


def create_classifiers(pivot_matrix, pivot_appearances, dicts, train_sets, train_source, train_source_labels):

    # order of dicts is train_source, source, unlabeled, target

    train_unlabeled_dict = dicts[2]
    source_examples = train_sets[0]
    source_examples_unlabeled = train_unlabeled_dict.transform(train_source).toarray()
    source_examples_for_adaptation = np.delete(source_examples_unlabeled, pivot_appearances, 1)

    adapted_source_examples = source_examples_for_adaptation.dot(pivot_matrix)

    adapted_source = np.concatenate((source_examples, adapted_source_examples), 1)

    classifiers = []

    # train the baseline classifier, which is a linear model with no adaptation
    print("Training baseline...")
    baseline = model.LogisticRegression(C=0.1, solver="lbfgs")
    baseline.fit(train_sets[0], train_source_labels)
    classifiers.append(("Baseline", baseline))

    # train the source classifier with adaptation
    print("Training source classifier...")
    source = model.LogisticRegression(C=0.1, solver="lbfgs")
    source.fit(adapted_source, train_source_labels)
    classifiers.append(("Source", source))

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
