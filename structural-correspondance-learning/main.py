import data
import argparse as argp


LAMBDA = 1e-3
MU = 1e-1


def source_loss():
    pass


def correct_misalignments():
    pass


def alpha_dist():
    pass


def select_pivots(num_pivots, labeled_source, unlabeled_source, unlabeled_target):
    # want to choose the num_pivots features with the highest mutual information gain to the source label
    # sort the features according to how many times they occur in both teh source and target domains
    # then, choose the num_pivots
    pass


def calc_mutual_info():
    pass


def scl():
    pass


def main():
    ap = argp.ArgumentParser()
    ap.add_argument("-s", "--source", required=True, help="Source domain")
    ap.add_argument("-t", "--target", required=True, help="Target domain")
    args = vars(ap.parse_args())
    labeled_source, unlabeled_source = data.collect_review_data(args["source"])
    _, unlabeled_target = data.collect_review_data(args["target"])

    # split source and target datasets into training and testing data
    #baseline classifier


    #classifier trained in domain

    #classifiers trained on new domains


if __name__ == "__main__":
    main()
