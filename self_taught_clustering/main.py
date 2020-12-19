import random
import sys
from random import shuffle
from STC import update_clusters, init_feat_and_var
from data import collect_review_data


random.seed(12345)


def main():
    if len(sys.argv) != 3:
        print (len(sys.argv))
        raise ValueError("Error: Incorrect Number of Args!")
    iterations = int(sys.argv[1])
    num_features = int(sys.argv[2])

    init_num_targ_clus = 2
    init_num_feat_clus = 5

    # get review data from dvd database
    targ_reviews, aux_reviews = collect_review_data("dvd", "books", num_features)

    # randomize the data sets
    shuffle(targ_reviews)

    shuffle(aux_reviews)

    # initialize the target data
    init_targ_clus, init_targ_feat_clus = init_feat_and_var(targ_reviews, num_features, init_num_targ_clus,
                                                            init_num_feat_clus)

    # initialize the auxiliary data
    init_aux_clus, init_aux_feat_clus = init_feat_and_var(aux_reviews, num_features, init_num_targ_clus,
                                                          init_num_feat_clus)

    total_feat_clus = []

    for feat_clus in init_targ_feat_clus:
        total_feat_clus.append(feat_clus)

    for feat_clus in init_aux_feat_clus:
        total_feat_clus.append(feat_clus)

    # run the method for updating clusters
    update_clusters(init_targ_clus, init_aux_clus, total_feat_clus,
                    iterations)




if __name__ == '__main__':
    main()



