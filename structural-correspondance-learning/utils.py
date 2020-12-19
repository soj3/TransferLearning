import random
from sklearn.feature_extraction.text import CountVectorizer
from main import *
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


def split_data(pos_data, neg_data, num_folds, examples_per_fold):
    pos = deepcopy(pos_data)
    neg = deepcopy(neg_data)
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


def get_a_dist(pivot_matrix, pivot_appearances, dicts, source_un, target_un):
    total_num_examples = len(source_un) + len(target_un)
    source_proportion = len(source_un) / total_num_examples
    target_proportion = 1 - source_proportion

    combined_examples = []
    train_domain_labels = []
    test_examples = []
    test_domain_labels = []
    random.shuffle(source_un)
    random.shuffle(target_un)

    for i in range(int(1 / 4 * source_proportion * total_num_examples)):
        combined_examples.append(source_un.pop())
        train_domain_labels.append(1)

    for i in range(int(1 / 4 * target_proportion * total_num_examples)):
        combined_examples.append(target_un.pop())
        train_domain_labels.append(-1)

    for i in range(int(1 / 4 * source_proportion * total_num_examples)):
        combined_examples.append(source_un.pop())
        train_domain_labels.append(1)

    for i in range(int(1 / 4 * target_proportion * total_num_examples)):
        test_examples.append(target_un.pop())
        test_domain_labels.append(-1)

    dict = dicts[2]
    train_data = dict.transform(combined_examples).toarray()
    test_data = dict.transform(test_examples).toarray()

    train_examples_for_adaptation = np.delete(train_data, pivot_appearances, 1)
    adapted_train_examples = train_examples_for_adaptation.dot(pivot_matrix)

    test_examples_for_adaptation = np.delete(test_data, pivot_appearances, 1)
    adapted_test_examples = test_examples_for_adaptation.dot(pivot_matrix)

    classifier = model.SGDClassifier(loss="modified_huber")
    classifier.fit(adapted_train_examples, train_domain_labels)
    predictions = classifier.predict(adapted_test_examples)

    loss = huber_loss(predictions, test_domain_labels)

    loss_per_example = loss / len(test_examples)
    risk = loss_per_example/len(test_examples)

    a_dist = round((1 - risk) * 100, 2)

    return a_dist

