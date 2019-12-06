import random
from sklearn.feature_extraction.text import CountVectorizer
random.seed(12345)


def merge_list(list1, list2):
    dict1 = dict(list1)
    dict2 = dict(list2)
    dict3 = {**dict1, **dict2}
    for key in dict3.keys():
        if key in dict1.keys() and key in dict2.keys():
            dict3[key] = dict1[key] + dict2[key]
    return list(dict3.items())


def merge_pivots_and_vocab(vocab, pivots, NUM_FEATURES):
    keys = [k for k, _ in vocab]
    for pivot in pivots:
        if pivot not in keys[:NUM_FEATURES]:
            vocab.insert(0, (pivot, 50))
    keys = [k for k, _ in vocab]
    for pivot in pivots:
        assert pivot in keys
    return vocab


def split_data(pos, neg):

    pos_proportion = len(pos)/(len(pos) + len(neg))
    neg_proportion = 1 - pos_proportion

    num_pos_train = int(pos_proportion * 1600)
    num_neg_train = int(neg_proportion * 1600)

    num_pos_test = int(pos_proportion * 400)
    num_neg_test = int(neg_proportion * 400)

    train_examples = []
    test_examples = []
    train_labels, test_labels = [], []

    random.shuffle(pos)
    random.shuffle(neg)

    for i in range(num_pos_train):
        train_examples.append(pos.pop())
        train_labels.append(1)

    for i in range(num_neg_train):
        train_examples.append(neg.pop())
        train_labels.append(0)

    for i in range(num_pos_test):
        test_examples.append(pos.pop())
        test_labels.append(1)

    for i in range(num_neg_test):
        test_examples.append(neg.pop())
        test_labels.append(0)

    return train_examples, train_labels, test_examples, test_labels


def get_dicts_and_train_sets(train_source, train_and_unlabeled, unlabeled, target_un):
    dicts = []
    train_sets = []

    dict1 = CountVectorizer(binary=True, min_df=5)
    x_train = dict1.fit_transform(train_source).toarray()

    dicts.append(dict1)
    train_sets.append(x_train)

    source_dict = CountVectorizer(binary=True, min_df=20)
    x_train_source = source_dict.fit_transform(train_and_unlabeled).toarray()

    dicts.append(source_dict)
    train_sets.append(x_train_source)

    unlabeled_dict = CountVectorizer(binary=True, min_df=40)
    x_train_unlabeled = unlabeled_dict.fit_transform(unlabeled).toarray()

    dicts.append(unlabeled_dict)
    train_sets.append(x_train_unlabeled)

    target_dict = CountVectorizer(binary=True, min_df=20)
    x_train_target = target_dict.fit_transform(target_un).toarray()

    dicts.append(target_dict)
    train_sets.append(x_train_target)

    return dicts, train_sets
