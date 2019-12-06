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


def split_data(pos, neg, num_folds, examples_per_fold):

    data_len = len(pos) + len(neg)
    pos_proportion = len(pos)/data_len
    neg_proportion = 1 - pos_proportion
    data_folds = []
    data_fold_labels = []
    for i in range(num_folds):
        fold = []
        fold_labels = []
        num_pos = int(pos_proportion * examples_per_fold)
        num_neg = int(neg_proportion * examples_per_fold)

        random.shuffle(pos)
        random.shuffle(neg)

        for j in range(num_pos):
            fold.append(pos.pop())
            fold_labels.append(1)

        for j in range(num_neg):
            fold.append(neg.pop())
            fold_labels.append(-1)

        data_folds.append(fold)
        data_fold_labels.append(fold_labels)

    return data_folds, data_fold_labels


def get_dicts_and_train_sets(train_source, train_and_unlabeled, unlabeled, target_un):
    dicts = []
    train_sets = []

    dict1 = CountVectorizer(binary=True, min_df=30)
    x_train = dict1.fit_transform(train_source).toarray()

    dicts.append(dict1)
    train_sets.append(x_train)

    source_dict = CountVectorizer(binary=True, min_df=30)
    x_train_source = source_dict.fit_transform(train_and_unlabeled).toarray()

    dicts.append(source_dict)
    train_sets.append(x_train_source)

    unlabeled_dict = CountVectorizer(binary=True, min_df=30)
    x_train_unlabeled = unlabeled_dict.fit_transform(unlabeled).toarray()

    dicts.append(unlabeled_dict)
    train_sets.append(x_train_unlabeled)

    target_dict = CountVectorizer(binary=True, min_df=30)
    x_train_target = target_dict.fit_transform(target_un).toarray()

    dicts.append(target_dict)
    train_sets.append(x_train_target)

    return dicts, train_sets


def huber_loss(predictions, actual):
    loss = 0
    assert len(predictions) == len(actual)
    for i in range(len(predictions)):
        if predictions[i] * actual[i] >= -1:
            loss += max(0, 1 - predictions[i] * actual[i]) ** 2
        else:
            loss += -4 * predictions[i] * actual[i]

    return loss

