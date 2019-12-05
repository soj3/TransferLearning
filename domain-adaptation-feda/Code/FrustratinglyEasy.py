import numpy as np
import random
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=50)
# Change parameters to determine
# A. Size of the sample that we take as "labeled" from the target domain
# B. THe number of tests we want to run
test_size = 100
n = 50
# For the amazon dataset, what are the source and target domains
src = 'dvd'
tgt = 'kitchen'

# a simple similarity metric implemented simply. jaccard similarity is just the
# size of the intersection divided by the size of the union
def jaccard_similarity(domain1_labels,domain2_labels):
    print(np.array(domain1_labels))
    intersect = np.intersect1d(np.array(domain1_labels), np.array(domain2_labels))
    union = np.union1d(domain1_labels,domain2_labels)
    print(intersect)
    return intersect.size / union.size

def load_data_sent(src, tgt):
    src_features_neg, src_labels_neg, a = read_data_sentiment('processed_acl/'+ src + '/negative.review')
    src_features_pos, src_labels_pos, a = read_data_sentiment('processed_acl/'+ src + '/positive.review')
    tgt_features_neg, tgt_labels_neg, a = read_data_sentiment('processed_acl/'+ tgt + '/negative.review')
    tgt_features_pos, tgt_labels_pos, a = read_data_sentiment('processed_acl/'+ tgt + '/positive.review')
    tgt_features = tgt_features_pos + tgt_features_neg
    tgt_labels = tgt_labels_pos + tgt_labels_neg
    src_features = src_features_pos + src_features_neg
    src_labels = src_labels_pos + src_labels_neg
    return src_features, src_labels, tgt_features, tgt_labels


def load_data_spam():
    src_features, src_labels , a = read_data_spam('evaluation_data_labeled/task_a_lab/task_a_u00_eval_lab.tf')
    tgt_features, tgt_labels , a = read_data_spam('evaluation_data_labeled/task_a_lab/task_a_u01_eval_lab.tf')
    return src_features, src_labels, tgt_features, tgt_labels

def load_data_test():
    src_features, src_labels , a = read_data_spam('evaluation_data_labeled/task_a_lab/test1.tf')
    tgt_features, tgt_labels , a = read_data_spam('evaluation_data_labeled/task_a_lab/test2.tf')
    return src_features, src_labels, tgt_features, tgt_labels

def read_data_sentiment(filename):
    features= []
    labels = []
    all_features = {}
    with open(filename) as f:
        for line in f:
            example = line.strip().split(' ')
            example_features = {}
            if example[-1] == '#label#:positive':
                labels.append(1)
            else:
                labels.append(-1)
            for j in range(len(example)-1):
                word, freq = example[j].split(':')
                example_features[word] = int(freq)
                all_features[word] = int(freq)
            features.append(example_features)
    return features, labels, all_features

def read_data_newsgroup(filename_data,filename_labels):
    features= []
    labels = []
    all_features = {}
    with open(filename_data) as f:
        for line in f:
            example = line.strip().split(' ')
            example_features = {}
            if example[-1] == '#label#:positive':
                labels.append(1)
            else:
                labels.append(-1)
            for j in range(len(example)-1):
                word, freq = example[j].split(':')
                example_features[word] = int(freq)
                all_features[word] = int(freq)
            features.append(example_features)
    return features, labels, all_features


def read_data_spam(filename):
    features = []
    labels = []
    all_features = {}
    with open(filename) as f:
        for line in f:
            example = line.strip().split(' ')
            example_features = {}
            if example[0] == '1':
                labels.append(1)
            else:
                labels.append(-1)
            for j in range(1,len(example)):
                word, freq = example[j].split(':')
                example_features[word] = int(freq)
                all_features[word] = int(freq)
            features.append(example_features)
    return features, labels, all_features


# pre preprocess because feda is also a preprocessing step. Prepreprocessing takes care of arranging the features into
# the same order and actually indluding all of the feature space from both the source and target domain as they may differ
# reduces the number of features to the most important ones as otherwise it takes hours to run
# (takes the n most frequently occuring features from all the domains)
def prepreprocess_data(src_features,tgt_features,n):

    vectorizer = DictVectorizer(sparse=False)
    # I need an entry for EVERY feature so I can't use a sparse matrix. What this is actually doing then is taking
    # all of the dictionary entries in any example and then wrapping them all together in a single feature space, then
    # enetering in that feature space the
    # print(src_features + tgt_features)
    completed_features = vectorizer.fit_transform(src_features + tgt_features)
    feature_count = np.sum(completed_features,0)
    sorted_features = np.argsort(feature_count)
   # print(completed_features)
    src_features = completed_features[:len(src_features), sorted_features[-n:]]
    tgt_features = completed_features[len(src_features):, sorted_features[-n:]]
    print(src_features.shape, tgt_features.shape)
    return src_features, tgt_features


def tgt_learner_svm(src_domain, src_domain_labels, eval_features, train_features, eval_labels, train_labels):

    tgt_learner = LinearSVC(C=1,max_iter=2000)
    tgt_learner.fit(train_features, train_labels)
    score = tgt_learner.score(eval_features, eval_labels)
    return score


def all_learner_svm(src_domain, src_domain_labels, eval_features, train_features, eval_labels, train_labels):

    all_features = np.concatenate((src_domain, train_features))
    all_labels = np.concatenate((src_domain_labels, train_labels))
    tgt_learner = LinearSVC(max_iter=2000)
    tgt_learner.fit(all_features, all_labels)
    score = tgt_learner.score(eval_features, eval_labels)
    return score


def feda_svm(src_domain, src_domain_labels, eval_features, train_features, eval_labels, train_labels):
    all_labels = np.concatenate((src_domain_labels, train_labels))
    aug_features_src = np.concatenate((src_domain, src_domain, np.zeros(src_domain.shape)),axis =1)
    aug_features_train = np.concatenate((train_features, np.zeros(train_features.shape), train_features), axis=1)
    aug_features_eval = np.concatenate((eval_features, np.zeros(eval_features.shape), eval_features), axis=1)
    aug_features = np.concatenate((aug_features_src, aug_features_train))
    feda_learner = LinearSVC(max_iter=2000) # feda is agnostic to the underlying
    # learning algorithm. As such i used the sklearn library to actually make sure that my implementation worked.
    # The maximum iterations were increased to have a better result with the spam dataset as it failed to converge
    feda_learner.fit(aug_features, all_labels)
    score = feda_learner.score(aug_features_eval, eval_labels)
    return score


def tgt_learner_log_reg(src_domain, src_domain_labels, eval_features, train_features, eval_labels, train_labels):
    tgt_learner = LogisticRegression(solver='liblinear', max_iter= 200)
    tgt_learner.fit(train_features, train_labels)
    score = tgt_learner.score(eval_features, eval_labels)
    return score


def all_learner_log_reg(src_domain, src_domain_labels, eval_features, train_features, eval_labels, train_labels):

    all_features = np.concatenate((src_domain, train_features))
    all_labels = np.concatenate((src_domain_labels, train_labels))
    all_learner = LogisticRegression(solver='liblinear', max_iter= 200)
    all_learner.fit(all_features, all_labels)
    score = all_learner.score(eval_features, eval_labels)
    return score


def feda_log_reg(src_domain, src_domain_labels, eval_features, train_features, eval_labels, train_labels):

    all_labels = np.concatenate((src_domain_labels, train_labels))
    aug_features_src = np.concatenate((src_domain, src_domain, np.zeros(src_domain.shape)),axis=1)
    aug_features_train = np.concatenate((train_features, np.zeros(train_features.shape), train_features), axis=1)
    aug_features_eval = np.concatenate((eval_features, np.zeros(eval_features.shape), eval_features), axis=1)
    aug_features = np.concatenate((aug_features_src, aug_features_train))
    feda_learner = LogisticRegression(solver='liblinear', max_iter= 200)
    feda_learner.fit(aug_features, all_labels)
    score = feda_learner.score(aug_features_eval, eval_labels)
    return score


def feda_log_reg_mod(src_domain,src_domain_labels, eval_features, train_features, eval_labels, train_labels, similarity):

    all_labels = np.concatenate((src_domain_labels, train_labels))
    weights = np.full(len(src_domain_labels),similarity)
    train_weights = np.ones(len(train_labels))
    all_weights = np.concatenate((weights,train_weights))
    aug_features_src1 = np.concatenate((src_domain, src_domain, np.zeros(src_domain.shape)), axis=1)
    aug_features_train = np.concatenate((train_features, np.zeros(train_features.shape), train_features),axis=1)
    aug_features_eval = np.concatenate((eval_features, np.zeros(eval_features.shape), np.zeros(eval_features.shape), eval_features),axis=1)
    aug_features = np.concatenate((aug_features_src1, aug_features_train))
    feda_learner = LogisticRegression(C=1, solver='liblinear', max_iter= 200) # feda is agnostic to the underlying
    # learning algorithm.
    feda_learner.fit(aug_features, all_labels, sample_weight=all_weights)
    score = feda_learner.score(aug_features_eval, eval_labels)
    return score

def training_samples(tgt_features, tgt_labels, test_size):
    eval_features = []
    train_features = []
    eval_labels = []
    train_labels = []
    train_set = random.sample(range(0,len(tgt_features)), test_size)
    for i in range(0,len(tgt_features)):
        if(i in train_set):
            train_features.append(tgt_features[i])
            train_labels.append(tgt_labels[i])
        else:
            eval_features.append(tgt_features[i])
            eval_labels.append(tgt_labels[i])

    return np.vstack(eval_features), np.vstack(train_features), eval_labels, train_labels

# def create_feature_space(features):
#     feature_language = []
#     for x in features:
#         for f, v in x.items():
#
#             if f in feature_language:
#
#             else:
#                 feature_language.append(f)



# src_features, src_labels, tgt_features, tgt_labels = load_data_test()

# src_features, tgt_features = prepreprocess_data(src_features, tgt_features, 10000)


src_features, src_labels, tgt_features, tgt_labels = load_data_sent(src, tgt)


src_features, tgt_features = prepreprocess_data(src_features, tgt_features, 10000)

print('{} to {} using {} target examples'.format(src, tgt, test_size))
f_log_reg = []
f_svm = []
t_log_reg = []
t_svm = []
a_log_reg = []
a_svm = []
for i in range(n):
    eval_features, train_features, eval_labels, train_labels = training_samples(tgt_features, tgt_labels, test_size)
    f_log_reg.append(feda_log_reg(src_features, src_labels, eval_features, train_features, eval_labels, train_labels))
    t_log_reg.append(tgt_learner_log_reg(src_features, src_labels, eval_features, train_features, eval_labels, train_labels))
    a_log_reg.append(all_learner_log_reg(src_features, src_labels, eval_features, train_features, eval_labels, train_labels))
    f_svm.append(feda_svm(src_features, src_labels, eval_features, train_features, eval_labels, train_labels))
    t_svm.append(tgt_learner_svm(src_features, src_labels, eval_features, train_features, eval_labels, train_labels))
    a_svm.append(all_learner_svm(src_features, src_labels, eval_features, train_features, eval_labels, train_labels))


print('Log Reg FEDA:', f_log_reg)
print(np.average(f_log_reg))
print(np.std(f_log_reg))
print('Log Reg All:', a_log_reg)
print(np.average(a_log_reg))
print(np.std(a_log_reg))
print('Log Reg Target:', t_log_reg)
print(np.average(t_log_reg))
print(np.std(t_log_reg))
print('SVM FEDA:', f_svm)
print(np.average(f_svm))
print('SVM All:', a_svm)
print(np.average(a_svm))
print('Target SVM:', t_svm)
print(np.average(t_svm))


src_features, src_labels, tgt_features, tgt_labels = load_data_spam()

src_features, tgt_features = prepreprocess_data(src_features, tgt_features, 10000)

print('spam train to spam eval using {} labeled target examples'.format(test_size))
f_svm = []
t_svm = []
a_svm = []

for i in range(0, n):
    eval_features, train_features, eval_labels, train_labels = training_samples(tgt_features, tgt_labels,test_size=test_size)
    f_svm.append(feda_svm(src_features, src_labels, eval_features, train_features, eval_labels, train_labels))
    t_svm.append(tgt_learner_svm(src_features, src_labels, eval_features, train_features, eval_labels, train_labels))
    a_svm.append(all_learner_svm(src_features, src_labels, eval_features, train_features, eval_labels, train_labels))
print('SVM FEDA:', f_svm)
print(np.average(f_svm))
print(np.std(f_svm))
print('SVM All:', a_svm)
print(np.average(a_svm))
print(np.std(a_svm))
print('SVM Target:', t_svm)
print(np.average(t_svm))
print(np.std(t_svm))