import math

"""
find the aggregate entropy of the clusters
"""
def total_entropy(var_clusters):
    overall_entropy = 0
    for clus_index in range(len(var_clusters)):
        correct_label, total, accuracy = cluster_entropy(var_clusters[clus_index], clus_index)
        if accuracy == 0:
            clus_entropy = 0
        else:
            clus_entropy = - accuracy * math.log(accuracy)
        if total == 0:
            pass
        else:
            overall_entropy += (correct_label / total) * clus_entropy
    return overall_entropy


"""
find the entropy of an individual cluster
"""
def cluster_entropy(var_cluster, clus_index):
    correct_label = 0
    total = 0
    # check if the label matches the cluster the data is in
    for data in var_cluster:
        if data.label == clus_index:
            correct_label += 1
        total += 1
    if total == 0:
        return 0, 0, 0
    else:
        acc = correct_label / total
        return correct_label, total, acc
