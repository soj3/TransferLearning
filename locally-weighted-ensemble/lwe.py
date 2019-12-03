from sklearn.cluster import AgglomerativeClustering
from typing import *
from sklearn.svm import LinearSVC
import numpy as np
from sklearn import metrics


def lwe(
    train_sets: List[List[Any]],
    models: List[object],
    test_data: List[any],
    threshold: float,
    cluster_value: int = 2,
) -> List[any]:
    """
    Inputs:
        - train_data:
            list of k training sets D1, D2,..., Dk
        - k classification models
            list of models M1, M2,..., Mk for k > 1
        - Test set T from a different domain where the classification task is the same
        - Threshold and cluster number c
    Outputs:
        - The set of predicted labels Y for examples in T
    """
    cluster = AgglomerativeClustering()
    predicted_clusters = []
    for train_data in train_sets:
        predicted_clusters += cluster.fit_predict(train_data)

    # TODO extract the actual cluster/class values from
    if purity([ex[-1] for ex in train_data], predicted_clusters) > 0.5:
        pass

    for ex in test_data:
        pass


def generate_neighborhood():
    pass


def compute_prediction(y: List[Any], E, x):
    # wM_i, x = P(M_i | x) is the true model weight that is locally adjusted for x representing the model's effectiveness on the test domain
    return np.sum(np.prod(w[i]))


def purity(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)
