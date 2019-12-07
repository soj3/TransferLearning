from typing import Any, Dict, List, Tuple, Union
from collections.abc import Sized
from multiprocessing import Pool
import numpy as np
from networkx import Graph
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering as Cluster
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from example import Example, SentimentExample

n_clusters = 2
verbose = True
n_cores = 8


def score(y, y_hat):
    """
    Return percentage accuracy given two lists of output predictions
    """
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
    pool = Pool(processes=n_cores)
    # use a hierarchial clustering model to separate into 'positive' and 'negative' clusters
    cluster = Cluster(n_clusters=clusters)
    cluster_purities = pool.map(
        lambda data: purity(labels(data), cluster.fit_predict(data)), train
    )

    print(f"Data Clustered with average purity {avg(cluster_purities)}")
    # if cluster purity is poor, then return average of all the predictions
    if avg(cluster_purities) < 0.5:
        print("Poor clustering quality, returning weighted average")
        # return the weighted probability of label = 1 for every x in T
        weight = 1 / len(models)
        # return the equally weighted output positive probability
        outputs = [sum(weight * prediction(model, x) for model in models) for x in test]
        return [round(x) for x in outputs]
    else:
        cluster_preds = Cluster(n_clusters=clusters).fit_predict(test)
        neighborhoods = pool.map(
            lambda model: generate_neighborhood(
                data=test,
                model_predictions=model.predict(test),
                cluster_predictions=cluster_preds,
            ),
            models,
        )

        t_prime = set()
        outputs = {}
        for x in test:
            # average similarity over all neighborhoods
            weights = [s(gm, gt, x) for gm, gt in neighborhoods]
            norm = sum(weights)
            # normalize by the sum of weights to ensure a weighted sum of probabilities does not exceed 1
            weights = [w / norm for w in weights]
            # average s(x) >= delta
            if avg(weights) >= threshold:
                outputs[x] = sum(
                    weight * prediction(model, x)
                    for weight, model in zip(weights, models)
                )
            else:
                t_prime.add(x)

        test_predictions = {x: pred for x, pred in zip(test, cluster_preds)}
        for x in t_prime:
            # find other members of x's cluster if they're already classified with sufficient s_avg
            c_prime = [
                ex
                for ex in test_predictions
                if test_predictions[ex] == test_predictions[x]
                and (ex not in t_prime or ex in outputs)
            ]
            # choose the average probability of y=1 of members in x's cluster
            assert len(c_prime) > 0
            outputs[x] = sum(outputs[ex] for ex in c_prime) / len(c_prime)
        outputs = [outputs[x] for x in test]
        return [round(x) for x in outputs]


def prediction(model: Union[LogisticRegression, SVC], x: Example, pos_label=1):
    pos_idx = list(model.classes_).index(pos_label)
    return model.predict_proba([x])[0][pos_idx]


def avg(values: Sized) -> float:
    assert len(values) > 0
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
    assert union > 0
    return intersect / union


def generate_neighborhood(
    data: List[Example], model_predictions: List[int], cluster_predictions: List[int]
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
    for i in range(len(data)):
        if i % 50 == 0:
            print(f"i: {i}")
        for j in range(i, len(data)):
            u, v = data[i], data[j]
            # print(f"i:{i}, j:{j}")
            # if the examples have the same predicted output from the model on the test set, add a connecting edge in gm
            if model_predictions[i] == model_predictions[j]:
                gm.add_edge(u, v)

            # if the examples are members of the same cluster on the test set, add a connecting edge in gt
            if cluster_predictions[i] == cluster_predictions[j]:
                gt.add_edge(u, v)
    return gm, gt


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


def labels(data: List[Example]):
    return [x.label for x in data]
