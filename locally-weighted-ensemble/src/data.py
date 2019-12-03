from os.path import dirname, abspath, join
from example import SentimentExample
from typing import List

root_dir = dirname(abspath(f"{__file__}/.."))
data_folder = join(dirname(root_dir), "data-sets")
acl_folder = join(data_folder, "processed_acl")


def collect_review_data(num_features: int = 3000):
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
                reviews.append(parse_line(line, vocab))

    return reviews


def parse_line(line, vocab):
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

