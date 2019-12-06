from probability import (features_in_var_clus, joint_prob_dist, co_cluster_indiv_cluster, find_sum_per_feat_cluster,
                         feat_sum_without_zero)
import math
import numpy as np
import evaluate
import copy
from random import sample

"""
splits list into a subset of lists based on the specified number of elements for each list
"""


def subset(lis, num_elements):
    for i in range(0, len(lis), num_elements):
        yield lis[i:i + num_elements]


"""
set the initial clusters for both features and variables
"""


def init_feat_and_var(reviews, num_features, init_num_targ_clus, init_num_feat_clus):
    feat_clus = []
    init_feat = []
    init_reviews_clus = []

    for review in reviews:
        feat_clus.append(review.features)

    list_ex = sample(range(0, num_features), num_features)

    for feat in feat_clus:
        temp = []
        for x in range(0, len(feat)):
            temp.append(feat[list_ex[x]])
        init_feat.append(temp)

    for x in range(0, len(reviews)):
        reviews[x].features = init_feat[x]
        init_reviews_clus.append(reviews[x])

    num_elements_targ = int(len(init_reviews_clus) / init_num_targ_clus)

    init_targ_clus = list(subset(init_reviews_clus, num_elements_targ))

    num_elements_feat = int(len(init_feat) / init_num_feat_clus)

    init_feat_clus = list(subset(init_feat, num_elements_feat))

    return init_targ_clus, init_feat_clus


"""
input the current structure of data clusters and re-arrange it to maximize the information gain
"""


def calc_new_cluster_val(var_clusters, feat_clusters):
    end_clusters = [[] for _ in range(len(var_clusters))]

    # delete a data point and add it to each cluster to check where the data point should be
    for clus in range(len(var_clusters)):
        for data_index in range(len(var_clusters[clus])):
            temp_var_clusters = copy.deepcopy(var_clusters)
            del temp_var_clusters[clus][data_index]
            new_cluster_idx = np.argmin([cluster_func_value(temp_var_clusters, feat_clusters, c_j,
                                                            var_clusters[clus][data_index])
                                         for c_j in range(len(temp_var_clusters))])
            end_clusters[new_cluster_idx].append(var_clusters[clus][data_index])
    return end_clusters


"""
find the KL divergence in order to minimize the mutual information of a cluster for a certain data point
"""


def cluster_func_value(var_clusters, feat_clusters, clus_index, data):
    var_clusters[clus_index].append(data)

    feat_sum_per_cluster, total_feat_sums = find_sum_per_feat_cluster(feat_clusters)
    total_clus_sum_without_zero = feat_sum_without_zero(feat_sum_per_cluster, total_feat_sums)

    clus_var, features_in_clus_data = features_in_var_clus(var_clusters[clus_index])
    joint_prob = 0
    co_cluster_val = 0
    for feat_cluster in range(len(feat_clusters)):
        joint_prob += sum(joint_prob_dist(var_clusters[clus_index], feat_clusters[feat_cluster])[-1])
        co_cluster_val += sum(co_cluster_indiv_cluster(var_clusters[clus_index], feat_clusters[feat_cluster],
                                                       total_clus_sum_without_zero[feat_cluster])[-1])
    clus_value = clus_var[-1]
    if joint_prob == 0 or co_cluster_val == 0:
        temp_min_value = 0
    else:
        temp_min_value = (joint_prob / clus_value) * math.log(joint_prob / co_cluster_val)
    return temp_min_value


"""
method that updates both the target and auxiliary data, and will also output the total entropy after each iteration
"""


def update_clusters(targ_clusters, aux_clus, feat_clusters, iterations):
    acc = evaluate.total_entropy(targ_clusters)
    print("Entropy: %f" % acc)
    if iterations == 0:
        return targ_clusters
    else:
        upd_targ_clusters = calc_new_cluster_val(targ_clusters, feat_clusters)
        upd_aux_clusters = calc_new_cluster_val(aux_clus, feat_clusters)
        return update_clusters(upd_targ_clusters, upd_aux_clusters, feat_clusters, iterations - 1)
