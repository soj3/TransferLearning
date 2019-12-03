import random
random.seed(12345)


def merge_list(list1, list2):
    dict1 = dict(list1)
    dict2 = dict(list2)
    dict3 = {**dict1, **dict2}
    for key in dict3.keys():
        if key in dict1.keys() and key in dict2.keys():
            dict3[key] = dict1[key] + dict2[key]
    return list(dict3.items())


def merge_pivots_and_vocab(vocab, pivots):
    keys = {k for k, _ in vocab}
    for pivot in pivots:
        if pivot not in keys:
            vocab.insert(0, (pivot, 50))
            keys.add(pivot)
    return vocab


def split_data(data):
    train_features, train_labels, test_features, test_labels = [], [], [], []

    pos, neg = [], []
    for ex in data:
        (pos if ex.label == 1 else neg).append(ex)

    pos_proportion = len(pos)/(len(pos) + len(neg))
    neg_proportion = 1 - pos_proportion

    num_pos_train = int(pos_proportion * 1600)
    num_neg_train = int(neg_proportion * 1600)

    num_pos_test = int(pos_proportion * 400)
    num_neg_test = int(neg_proportion * 400)

    train_examples = []
    test_examples = []

    random.shuffle(pos)
    random.shuffle(neg)

    for i in range(num_pos_train):
        train_examples.append(pos.pop())

    for i in range(num_neg_train):
        train_examples.append(neg.pop())

    for i in range(num_pos_test):
        test_examples.append(pos.pop())

    for i in range(num_neg_test):
        test_examples.append(neg.pop())

    for ex in train_examples:
        train_features.append(ex.features)
        train_labels.append(ex.label)

    for ex in test_examples:
        test_features.append(ex.features)
        test_labels.append(ex.label)

    return train_features, train_labels, test_features, test_labels
