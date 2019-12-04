from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering as Cluster
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.base import LinearModel
from sklearn.svm import LinearSVR
from example import Example, SentimentExample
from networkx import Graph
import numpy as np
from typing import List, Any, Dict, Tuple


def lwe(
    train_sets: List[List[Example]],
    models: List[LinearModel],
    test_data: List[Example],
    threshold: float,
    num_clusters: int = 2,
) -> List[any]:

    """
    Inputs:

        - train_data:
            list of k training sets D1, D2,..., Dk
        - modesl:
            list of models M1, M2,..., Mk for k > 1
        - test_data:
            test set from a different domain with the same task
        - Threshold
        - cluster number for localizing the test domain
    Outputs:

        - The set of predicted labels Y for examples in T
    """

    # use a hierarchial clustering model to separate into 'positive' and 'negative' clusters
    cluster = Cluster(n_clusters=num_clusters)
    predicted_clusters = [
        prediction
        for prediction in cluster.fit_predict(train_data)
        for train_data in train_sets
    ]


def s(gm: Graph, gt: Graph, x: Example):

    pass


def generate_neighborhood(
    train: List[Example], test: List[Example], model: LinearModel, num_clusters=2
):
    """
    Implementation necessary for the proposed weight caculation in eq. 5 of Gao et. al

    Inputs:

        - train:
            a single training set of examples
        - test:
            a single testing set of examples from a separate domain from train
        - model:
            a base model which has been trained on train to be evaluated on test and compared
            with clustering results

    Outputs:

        - a tuple of the graphs (gm, gt) as used in eq. 5 for the model weight calculation
    """
    gt, gm = Graph(), Graph()
    gm.add_nodes_from(test)
    gt.add_nodes_from(test)

    model.fit(train)
    model_predictions = model.predict(test)

    cluster = Cluster(n_clusters=num_clusters)
    cluster_predictions = cluster.fit_predict(test)

    preds = zip(test, model_predictions, cluster_predictions)
    for u, m1, c1 in preds:
        for v, m2, c2 in preds:
            if u is not v:
                # if the examples have the same predicted output from the model on the test set, add a connecting edge
                if m1 == m2:
                    gm.add_edge(u, v)
                if c1 == c2:
                    gt.add_edge(u, v)
    return gm, gt


def conditional_probability(w, x):
    pass


def compute_prediction(m: LinearModel, examples: List[Example]):
    # wM_i, x = P(M_i | x) is the true model weight that is locally adjusted for x representing the model's effectiveness on the test domain
    m.predict(examples)
    # return np.sum(np.prod(w[i]))


def purity(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)
