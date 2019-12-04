from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering as Cluster
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.base import LinearModel
from sklearn.svm import LinearSVR
from example import Example, SentimentExample
from networkx import Graph
import numpy as np
from typing import List, Any, Dict, Tuple

n_clusters = 2


def lwe(
    train: List[List[Example]],
    models: List[LinearModel],
    test: List[Example],
    threshold: float,
    clusters: int = n_clusters,
) -> List[Tuple[Example, float]]:
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
    labels = [ex.label for ex in test]

    cluster_predictions = []
    model_predictions = []
    cluster_purities = []
    for data, model in zip(train, models):
        cluster_predictions.append(cluster.fit_predict(data))
        cluster_purities.append(purity(labels, cluster_predictions[-1]))
        model.fit(data)
        model_predictions.append(model.predict(test))

    # if cluster purity is irrelevant, then return average of all the predictions
    if sum(cluster_purities) / len(cluster_purities) < 0.5:
        weight = 1 / len(models)
        output = np.zeros(len(test[0]))
        for model in models:
            output = np.add(output, model.predict(test))
        return np.multiply(weight, output)

    neighborhoods = {
        model: generate_neighborhood(data=test, model=model, clusters=n_clusters)
        for model in models
    }
    t_prime = []
    for x in test:
        weights = [s(gm, gt, x) for gm, gt in neighborhoods]
        # average (sum/len) s(x) >= delta
        if sum(weights) / len(weights) >= threshold:
            output = np.zeros(len(test[0]))
            for model in models:
                output = np.add(output, model.predict(test))
        else:
            t_prime.append(x)

    for x in t_prime:
        pass


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

    # return the ratio of intersection and union cardinalities
    return len(gm_neighbors & gt_neighbors) / len(gm_neighbors | gt_neighbors)


def generate_neighborhood(
    data: List[Example], model: LinearModel, clusters: int = n_clusters
) -> Tuple[Graph, Graph]:
    """
    Implementation necessary for the proposed weight caculation in eq. 5 of Gao et. al
    Parameters:
    ------
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

    model_predictions = model.predict(data)

    cluster = Cluster(n_clusters=clusters)
    cluster_predictions = cluster.fit_predict(data)

    preds = zip(data, model_predictions, cluster_predictions)
    for u, m1, c1 in preds:
        for v, m2, c2 in preds:
            if u is not v:
                # if the examples have the same predicted output from the model on the test set, add a connecting edge in gm
                if m1 == m2:
                    gm.add_edge(u, v)

                # if the examples are members of the same cluster on the test set, add a connecting edge in gt
                if c1 == c2:
                    gt.add_edge(u, v)
    return gm, gt


def purity(y: List[float], y_hat: List[float]) -> float:
    """
    Parameters:
    -------
        y: the supervised output labels
        y_hat: the predicted output labels
    Returns:
    -------
        the purity of clustering output predictions compared to class annotations
    """
    # compute contingency matrix (also called confusion matrix)
    matrix = metrics.cluster.contingency_matrix(y, y_hat)
    # return purity
    return np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)
