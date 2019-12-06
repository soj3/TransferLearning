from os.path import dirname, abspath, join
from example import SentimentExample
from typing import List

current_folder = dirname(abspath(__file__))
data_folder = join(dirname(current_folder), "data-sets")

# Folders
acl_folder = join(data_folder, "processed_acl")

task_a_lab_folder = join(data_folder, "task_a_lab")
task_b_lab_folder = join(data_folder, "task_b_lab")

data_task_a_folder = join(data_folder, "data_task_a")
data_task_b_folder = join(data_folder, "data_task_b")

newsgroup_folder = join(data_folder, "20news-bydate/workingData")

# ---------------------------------------------- #
#                   Spam Data                    #
# ---------------------------------------------- #
def collect_spam_a_data(num_features=3000):
    domains = [
        "task_a_u00_eval_lab.tf",
        "task_a_u01_eval_lab.tf",
        "task_a_u02_eval_lab.tf",
    ]

    vocab = {}
    spam_domains = []
    for domain in domains:
        filename = join(task_a_lab_folder, domain)
        spam_domains.append(read_spams(domain, vocab, filename))

    vocab = sorted(vocab.items(), key=lambda item: item[1])[::-1]

    for spams in spam_domains:
        for spam in spams:
            spam.create_features(vocab[:num_features])

    return spam_domains


def collect_spam_b_data(num_features=3000):
    domains = [
        "task_b_u00_eval_lab.tf",
        "task_b_u01_eval_lab.tf",
        "task_b_u02_eval_lab.tf",
        "task_b_u03_eval_lab.tf",
        "task_b_u04_eval_lab.tf",
        "task_b_u05_eval_lab.tf",
        "task_b_u06_eval_lab.tf",
        "task_b_u07_eval_lab.tf",
        "task_b_u08_eval_lab.tf",
        "task_b_u09_eval_lab.tf",
        "task_b_u10_eval_lab.tf",
        "task_b_u11_eval_lab.tf",
        "task_b_u12_eval_lab.tf",
        "task_b_u13_eval_lab.tf",
        "task_b_u14_eval_lab.tf",
    ]

    vocab = {}
    spam_domains = []
    for domain in domains:
        filename = join(task_b_lab_folder, domain)
        spam_domains.append(read_spams(domain, vocab, filename))

    vocab = sorted(vocab.items(), key=lambda item: item[1])[::-1]

    for spams in spam_domains:
        for spam in spams:
            spam.create_features(vocab[:num_features])

    return spam_domains


def read_spams(domain, vocab, filename):
    spams = []

    with open(filename, "r") as f:
        for line in f:
            spams.append(parse_spam(line, vocab))

    return spams


def parse_spam(line, vocab):
    words_dict = {}
    split = line.split(" ")

    for word_count in split[1:]:
        word, count_str = word_count.split(":")
        count = int(count_str)

        vocab[word] = count if word not in vocab else vocab[word] + count
        words_dict[word] = count

    label = int(split[0])
    bool_label = 1 if label == 1 else 0
    return SentimentExample(words_dict, bool_label)


# ---------------------------------------------- #
#                News Group Data                 #
# ---------------------------------------------- #
def collect_newsgroup_data(num_features=3000):
    domains = ["trainCompletevBinary.txt", "testCompletevBinary.txt"]

    vocab = {}
    news_domains = []
    for domain in domains:
        filename = join(newsgroup_folder, domain)
        news_domains.append(read_newsgroups(domain, vocab, filename))

    vocab = sorted(vocab.items(), key=lambda item: item[1])[::-1]

    for news in news_domains:
        for new in news:
            new.create_features(vocab[:num_features])

    return news_domains


def read_newsgroups(domain, vocab, filename):
    spams = []

    with open(filename, "r") as f:
        for line in f:
            spams.append(parse_newsgroup(line, vocab))

    return spams


def parse_newsgroup(line, vocab):
    words_dict = {}
    split = line.split(" ")

    for word_count in split[:-1]:
        word, count_str = word_count.split(":")
        count = int(count_str)

        vocab[word] = count if word not in vocab else vocab[word] + count
        words_dict[word] = count

    label = int(split[-1])
    bool_label = 1 if label == 1 else 0
    return SentimentExample(words_dict, bool_label)


# ---------------------------------------------- #
#                 Review Data                    #
# ---------------------------------------------- #
def collect_review_data(num_features=3000):
    domains = ["books", "dvd", "electronics", "kitchen"]

    vocab = {}
    reviews_domains = []
    for domain in domains:
        reviews_domains.append(read_reviews(domain, vocab))

    vocab = sorted(vocab.items(), key=lambda item: item[1])[::-1]

    for reviews in reviews_domains:
        for r in reviews:
            r.create_features(vocab[:num_features])

    return reviews_domains


def read_reviews(domain, vocab):
    domain_folder = join(acl_folder, domain)
    file_names = ["negative.review", "positive.review", "unlabeled.review"]

    reviews = []

    for file_name in file_names:
        with open(join(domain_folder, file_name), "r") as f:
            for line in f:
                reviews.append(parse_review(line, vocab))

    return reviews


def parse_review(line, vocab):
    words_dict = {}
    split = line.split(" ")

    for word_count in split[:-1]:
        word, count_str = word_count.split(":")
        count = int(count_str)

        vocab[word] = count if word not in vocab else vocab[word] + count
        words_dict[word] = count

    label = split[-1].split(":")[1][:-1]
    bool_label = 1 if label == "positive" else 0
    return SentimentExample(words_dict, bool_label)
