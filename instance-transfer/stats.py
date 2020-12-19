from statistics import stdev, mean
from typing import List, Dict
import matplotlib.pyplot as plt


def calculate_stats(arr_of_confusion: List[Dict]):
    accuracies = []
    precisions = []
    recalls = []
    for matrix in arr_of_confusion:
        accuracies.append((matrix["tp"] + matrix["tn"]) / (sum(matrix.values())))
        precisions.append(matrix["tp"] / (matrix["tp"] + matrix["fp"]))
        recalls.append(matrix["tp"] / (matrix["tp"] + matrix["fn"]))

    print_formatted_stats("Accuracy", accuracies)
    print_formatted_stats("Precision", precisions)
    print_formatted_stats("Recall", recalls)


def print_formatted_stats(stat_str, to_stat):
    if len(to_stat) > 1:
        print(
            "{}: {}, {}".format(
                stat_str, round(mean(to_stat), 3), round(stdev(to_stat), 3)
            )
        )
    else:
        print("{}: {}".format(stat_str, round(to_stat[0], 3)))


def calculate_aroc(arr_of_confidence):
    sort_conf = sorted(arr_of_confidence, key=lambda conf: conf[1], reverse=True)
    total_positive = len([x for x in sort_conf if x[0]])
    total_negative = len(sort_conf) - total_positive

    roc_points = []
    fp, tp = 0, 0
    for label, confidence in sort_conf:
        if label:
            tp += 1
        else:
            fp += 1
        fpr = fp / total_negative if total_negative > 0 else 0
        tpr = tp / total_positive if total_positive > 0 else 0
        roc_points.append([fpr, tpr])
    X = [r[0] for r in roc_points]
    y = [r[1] for r in roc_points]

    print("Area under ROC: {0:0.3f}".format(calculate_integral(roc_points)))


def calculate_integral(points):
    area = 0

    pt_idx = 1
    while pt_idx < len(points):
        dx = points[pt_idx][0] - points[pt_idx - 1][0]
        area += ((points[pt_idx][1] + points[pt_idx - 1][1]) / 2) * dx
        pt_idx += 1

    return area
