import random
from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from data import collect_review_data, collect_spam_a_data, collect_spam_b_data
from example import Example
from lwe import lwe, score, labels


# reviews
b_data, d_data, e_data, k_data = collect_review_data(5000)

# # Spam Task A Data
# sp1, sp2, sp3 = collect_spam_a_data(3000)

# # Spam Task B Data
# sps15 = collect_spam_b_data(3000)

N = 1000

train = [random.sample(d_data, N), random.sample(b_data, N)]
models = [LogisticRegression(solver="lbfgs", max_iter=200) for data in train]

test = random.sample(k_data, N)


baseline = LogisticRegression(solver="lbfgs", max_iter=200)
baseline_train = random.sample(d_data, N)
baseline.fit(baseline_train, labels(baseline_train))
print(f"Baseline Accuracy: {score(baseline.predict(test), labels(test))}")

for i, model, data in zip(range(len(models)), models, train):
    print(f"Fitting Model {i}")
    model.fit(data, labels(data))
print(f"LWE Accuracy: {score(lwe(train, models, test, 0.5, 2), labels(test))}")
