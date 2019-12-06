from typing import Any, Dict, List, Tuple, Union
from collections.abc import Collection

import numpy as np
from networkx import Graph
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering as Cluster
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from example import Example, SentimentExample

n_clusters = 2


def score(y, y_hat):
    if len(y) > 0:
        return sum(
            int(actual == predicted) for actual, predicted in zip(y, y_hat)
        ) / len(y)
    else:
        raise ValueError


def lwe(
    train: List[List[Example]],
    models: List[Union[LogisticRegression, SVC]],
    test: List[Example],
    threshold: float,
    clusters: int = n_clusters,
) -> List[int]:
    """
    Locally Weighted Ensembling Implementation (Gao et. al)
    Parameters:
    -------
        train: list of k training sets D1, D2,..., Dk
        models: list of models M1, M2,..., Mk for k > 1
        test: test set from a different domain with the same task
        threshold: the value at which to choose an example label from weighted ensemble output vs placing the example directly into T'
        clusters: number for localizing the test domain
    Returns:
    -------
        a collection of tuples containing examples in the test set and the probability of a positive classification
    """
    # use a hierarchial clustering model to separate into 'positive' and 'negative' clusters
    cluster = Cluster(n_clusters=clusters)
    cluster_purities = [
        purity([ex.label for ex in data], cluster.fit_predict(data)) for data in train
    ]
    # if cluster purity is poor, then return average of all the predictions
    if avg(cluster_purities) < 0.5:
        weights = {model: 1 / len(models) for model in models}
        # return the weighted probability of label = 1 for every x in T
        return [
            1
            if sum(
                weights[model] * model.predict_proba([x])[-1][-1] for model in models
            )
            > 0.5
            else 0
            for x in test
        ]
    else:
        cluster = Cluster(n_clusters=clusters)
        preds = cluster.fit_predict(test)
        neighborhoods = [
            generate_neighborhood(data=test, model=model, cluster_predictions=preds)
            for model in models
        ]

        t_prime = set()
        outputs = {}
        for x in test:
            local_weights = {model: s(gm, gt, x) for model, gm, gt in neighborhoods}
            # average (sum/len) s(x) >= delta
            if avg(local_weights.values()) >= threshold:
                outputs[x] = sum(
                    [
                        local_weights[model] * model.predict_proba([x])[-1][-1]
                        for model in models
                    ]
                )
            else:
                t_prime.add(x)

        test_predictions = {x: pred for x, pred in zip(test, preds)}
        for x in t_prime:
            # choose the average probability of y=1 of neighbors of x
            outputs[x] = avg(
                [
                    outputs[ex]
                    for ex, label in test_predictions.items()
                    if label == test_predictions[ex] and ex not in t_prime
                ]
            )

        return [1 if outputs[x] > 0.5 else 0 for x in test]


def avg(values: Collection) -> float:
    return sum(values) / len(values)


def s(gm: Graph, gt: Graph, x: Example) -> float:
    """
    Return the similarity of model and cluster graphs in the neighborhood of an example
    Parameters:
    -------
        gm: the graph produced by a base model from a training set
        gt: the graph produced by clustering on the testing set
        x: the example central to the neighborhoods being compared
    Returns:
    -------
        a real valued 0 <= s <= 1 denoting the ratio of common neighbors of x between gm and gt
    """
    gm_neighbors = set(gm.neighbors(x))
    gt_neighbors = set(gt.neighbors(x))
    intersect = len(gm_neighbors & gt_neighbors)
    union = len(gm_neighbors | gt_neighbors)
    assert union != 0
    return intersect / union


def generate_neighborhood(
    data: List[Example],
    model: Union[LogisticRegression, SVC],
    cluster_predictions: List,
) -> Tuple[Graph, Graph]:
    """
    Implementation necessary for the proposed weight caculation in eq. 5 of Gao et. al
    Parameters:
    -------
        train: a single training set of examples
        test: a single testing set of examples from a separate domain from train
        model: a base model which has been trained on train to be evaluated on test and compared with clustering results
        clusters: number for localizing the test domain
    Returns:
    -------
        a tuple of the graphs (gm, gt) as used in eq. 5 for the model weight calculation
    """
    gt, gm = Graph(), Graph()
    gm.add_nodes_from(data)
    gt.add_nodes_from(data)

    assert len(data) == len(cluster_predictions)
    for i, u in enumerate(data):
        for j, v in enumerate(data):
            # if u == v:
            #     continue
            m1 = model.predict([u])[-1]
            m2 = model.predict([v])[-1]
            c1 = cluster_predictions[i]
            c2 = cluster_predictions[j]
            # if the examples have the same predicted output from the model on the test set, add a connecting edge in gm
            if m1 == m2:
                gm.add_edge(u, v)

            # if the examples are members of the same cluster on the test set, add a connecting edge in gt
            if c1 == c2:
                gt.add_edge(u, v)
    return model, gm, gt


def purity(y: List[float], y_hat: List[float]) -> float:
    """
    Parameters:
    -------
        y: the supervised output labels
        y_hat: the clustered output labels
    Returns:
    -------
        the purity of clustering output predictions compared to class annotations
    """
    # compute contingency matrix (also called confusion matrix)
    matrix = metrics.cluster.contingency_matrix(labels_true=y, labels_pred=y_hat)
    # return purity
    return np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)
