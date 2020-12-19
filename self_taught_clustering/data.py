from os.path import dirname, abspath, join

from example import SentimentExample

current_folder = dirname(abspath(__file__))
data_folder = join(current_folder, "data-sets")
acl_folder = join(data_folder, "processed_acl")

"""
preprocessing for the review data
"""


def collect_review_data(domain, aux, num_features):
    targ_reviews, targ_vocab = read_reviews(domain)
    aux_reviews, aux_vocab = read_reviews(aux)

    for review in targ_reviews:
        review.create_features(targ_vocab[:num_features])

    for review in aux_reviews:
        review.create_features(aux_vocab[:num_features])

    return targ_reviews, aux_reviews


"""
takes in the files to be transcribed to data and outputs the data itself
"""


def read_reviews(domain):
    domain_folder = join(acl_folder, domain)
    file_names = ["negative.review", "positive.review", "unlabeled.review"]
    reviews = []
    vocab = {}

    # extract the data from the document
    for file_name_index in range(2):

        with open(join(domain_folder, file_names[file_name_index]), "r") as f:

            index = 0

            for line in f:

                if index < 25:
                    reviews.append(parse_line(line, vocab))

                index += 1

    sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])[::-1]

    return reviews, sorted_vocab


# input a line and the vocabulary for the document and output  the line as type SentimentExample
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
