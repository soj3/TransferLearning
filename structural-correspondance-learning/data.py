from os.path import dirname, abspath, join
from example import SentimentExample


current_folder = dirname(abspath(__file__))
data_folder = join(dirname(current_folder), "data-sets")
acl_folder = join(data_folder, "processed_acl")


def collect_review_data(domain, num_features=3000):
    labeled_reviews, unlabeled_reviews, vocab = read_reviews(domain)

    for review in labeled_reviews:
        review.create_features(vocab[:num_features])

    for review in unlabeled_reviews:
        review.create_features(vocab[:num_features])

    return labeled_reviews, unlabeled_reviews


def read_reviews(domain):
    domain_folder = join(acl_folder, domain)
    file_names = ["negative.review", "positive.review"]

    labeled_reviews = []
    unlabeled_reviews = []
    vocab = {}

    for file_name in file_names:
        with open(join(domain_folder, file_name), "r") as f:
            for line in f:
                labeled_reviews.append(parse_line(line, vocab))

    with open(join(domain_folder, "unlabeled.review"), "r") as f:
        for line in f:
            unlabeled_reviews.append(parse_line(line, vocab))

    sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])[::-1]

    return labeled_reviews, unlabeled_reviews, sorted_vocab


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


