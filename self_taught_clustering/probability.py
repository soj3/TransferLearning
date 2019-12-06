import numpy as np

"""
find the joint probability distribution between a data cluster and a feature cluster. The output is a matrix where 
the rows are represented by the data and the columns are represented by the features
"""


def joint_prob_dist(data_cluster, feat_cluster):
    prob_dist = []
    total = 0
    for data in data_cluster:
        row = []
        row_tot = 0
        for feat_list in feat_cluster:
            row = np.multiply(data.features, feat_list)
            row_tot += sum(row)
        prob_dist.append(list(row))
        total += row_tot
    for row in range(0, len(prob_dist)):
        for elem in range(0, len(prob_dist[row])):
            prob_dist[row][elem] = prob_dist[row][elem] / total
    return prob_dist


"""
finds the sum of each individual cluster as well as the total sum for each feature
"""


def find_sum_per_feat_cluster(feat_clusters):
    feat_sum_per_cluster = []
    for feat_cluster in feat_clusters:
        feat_sum_per_cluster.append(list(map(sum, zip(*feat_cluster))))
    total_feat_sums = list(map(sum, map(lambda l: map(int, l), zip(*feat_sum_per_cluster))))
    return feat_sum_per_cluster, total_feat_sums


"""
finds the sum at each index of a given feature cluster
"""


def find_sum_cluster(feat_cluster):
    feat_sum_per_cluster = list(map(sum, zip(*feat_cluster)))
    return feat_sum_per_cluster


"""
find all the occurrences in the data set of a feature that occurs in a given cluster
"""


def feat_sum_without_zero(feat_sum_per_clus, total_feat_sums):
    clusters_sum = []
    for feat_list in feat_sum_per_clus:
        clus_sum_without_zero = 0
        for feat in range(len(feat_list)):
            if feat_list[feat] != 0:
                clus_sum_without_zero += total_feat_sums[feat]
        clusters_sum.append(clus_sum_without_zero)
    return clusters_sum


"""
finds the sum of the features for each data point in a cluster as well as the total number of features in a cluster
"""


def features_in_var_clus(var_cluster):
    feat_in_data = []
    for data in var_cluster:
        feat_in_data.append(sum(data.features))
    feat_in_clus_data = sum(feat_in_data)
    return feat_in_data, feat_in_clus_data


def find_total_feat_sums(feat_sum_per_cluster):
    return sum(feat_sum_per_cluster)


"""
find the co-cluster joint probability distribution
"""


def co_cluster_indiv_cluster(var_cluster, feat_cluster, total_feat_sum_without_zero):
    data_in_clus, feat_in_clus_data = features_in_var_clus(var_cluster)
    clus_prob_dist = joint_prob_dist(var_cluster, feat_cluster)
    cluster_sum = find_sum_cluster(feat_cluster)

    co_cluster_row = []
    for data_index in range(len(var_cluster)):
        temp = []
        for feat_index in range(len(feat_cluster[0])):
            temp.append(clus_prob_dist[data_index][feat_index] *
                        (data_in_clus[data_index] /
                         feat_in_clus_data) *
                        (cluster_sum[feat_index] /
                         total_feat_sum_without_zero))
        co_cluster_row.append(temp)
    return co_cluster_row
