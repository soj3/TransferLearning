from copy import deepcopy
import numpy as np


from classifier import LogRegClassifier
import scipy.optimize as spo

LAMBDA = 1e-1
MU = 1e-1
NUM_ITERATIONS = 10000


def source_loss(weights, original_adapted_weights, train_examples, train_labels):
    loss = 0
    predictions = []
    for ex in train_examples:
        if np.dot(weights, ex) > 0:
            predictions.append(1)
        else:
            predictions.append(-1)

    assert len(predictions) == len(train_labels)

    base_weights = weights[:-SVD_DIMENSION-1]
    adapted_weights = weights[-SVD_DIMENSION-1:-1]
    for i in range(len(predictions)):
        if predictions[i] * train_labels[i] >= 1:
            pass
        elif predictions[i] * train_labels[i] < -1:
            loss += (-4 * predictions[i] * train_labels[i])
        else:
            loss += ((1 - predictions[i] * train_labels[i]) ** 2)

    return loss + (LAMBDA * np.linalg.norm(base_weights) ** 2) + \
        (MU * np.linalg.norm(np.subtract(original_adapted_weights, adapted_weights) ** 2))


def gradient(weights, adapted_weights, train_examples, train_labels):
    grad = 0
    predictions = []
    for ex in train_examples:
        if np.dot(weights,ex) >0:
            predictions.append(1)
        else:
            predictions.append(-1)

    for i in range(len(predictions)):
        if predictions[i] * train_labels[i] + 1 <= 1:
            grad += -4
        elif -1 < predictions[i] * train_labels[i] < 1:
            grad += -2 * (1 - predictions[i] * train_labels[i])

    return [grad]


def correct_misalignments(base_classifier, train_examples, train_labels, pivot_matrix):
    print("Correcting misalignments...")
    adapted_examples = deepcopy(train_examples)
    for ex in adapted_examples:
        #print(len(ex))
        adapted_features = np.dot(ex, pivot_matrix)
        ex.extend(adapted_features)
    weights = base_classifier.coef_[0]
    adapted_weights = weights[-SVD_DIMENSION-1:-1]

    new_weights = spo.fmin(source_loss, weights,  args=(adapted_weights, adapted_examples, train_labels),
                           maxiter=NUM_ITERATIONS)
    new_classifier = LogRegClassifier(new_weights)
    return new_classifier
