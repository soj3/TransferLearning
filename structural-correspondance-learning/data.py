from os.path import dirname, abspath, join
from sklearn.feature_extraction.text import CountVectorizer


current_folder = dirname(abspath(__file__))
data_folder = join(dirname(current_folder), "data-sets")
acl_folder = join(data_folder, "processed_acl")


def parse_line(line):
    split = line.split(" ")
    text = ""
    for word_count in split[:-1]:
        word, count_str = word_count.split(":")
        count = int(count_str)

        text = text + word + " "
    return text


def get_reviews(domain):
    pos_reviews = []
    neg_reviews = []
    un_reviews = []

    domain_folder = join(acl_folder, domain)

    with open(join(domain_folder, "positive.review"), "r") as f:
        for line in f:
            pos_reviews.append(parse_line(line))

    with open(join(domain_folder, "negative.review"), "r") as f:
        for line in f:
            neg_reviews.append(parse_line(line))

    with open(join(domain_folder, "unlabeled.review"), "r") as f:
        for line in f:
            un_reviews.append(parse_line(line))

    return pos_reviews, neg_reviews, un_reviews
