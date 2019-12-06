from data import collect_review_data
from sklearn.linear_model import LogisticRegression
from lwe import lwe, score
import random
import numpy as np

b_data, d_data, e_data, k_data = collect_review_data(500)
random.shuffle(b_data)
book_classifier = LogisticRegression(solver="liblinear")
book_classifier.fit(b_data, [x.label for x in b_data])
random.shuffle(d_data)
print(score([x.label for x in k_data], book_classifier.predict(d_data)))

# print(
#     score([x.label for x in k_data], lwe([b_data], [book_classifier], k_data, 0.5, 2))
# )
