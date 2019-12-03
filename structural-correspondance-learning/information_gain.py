import math


def calc_mutual_info(source_data, feature):
    pos_and_feature = 0
    neg_and_feature = 0
    pos_not_feature = 0
    neg_not_feature = 0
    for example in source_data:
        if example.label == 1:
            if feature in example.words.keys():
                pos_and_feature += 1
            else:
                pos_not_feature += 1
        else:
            if feature in example.words.keys():
                neg_and_feature += 1
            else:
                neg_not_feature += 1

    label_entropy = calc_entropy((pos_and_feature + neg_and_feature) / len(source_data),
                                 (pos_not_feature + neg_not_feature) / len(source_data))

    pos_cond_entropy = calc_entropy(pos_and_feature / (pos_and_feature + pos_not_feature),
                                    pos_not_feature / (pos_and_feature + pos_not_feature))
    neg_cond_entropy = calc_entropy(neg_not_feature / (neg_not_feature + neg_and_feature),
                                    neg_and_feature / (neg_not_feature + neg_and_feature))

    cond_entropy = ((pos_and_feature + pos_not_feature) / len(source_data)) * pos_cond_entropy + \
                   ((neg_and_feature + neg_not_feature) / len(source_data)) * neg_cond_entropy

    return label_entropy - cond_entropy


def calc_entropy(pos, neg):
    if pos != 0 and neg != 0:
        return -(pos * math.log2(pos) + neg * math.log2(neg))
    else:
        return 0
