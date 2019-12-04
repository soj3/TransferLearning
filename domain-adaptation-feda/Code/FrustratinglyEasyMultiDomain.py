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


def jaccard_similarity(domain1_labels,domain2_labels):
   # print(np.array(domain1_labels))
    intersect = np.intersect1d(np.array(domain1_labels),np.array(domain2_labels))
    union = np.union1d(domain1_labels,domain2_labels)
    # print(intersect)
    return intersect.size / union.size


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

#as all of the inboxes have different domains in order to simplify implementation (because I use labeled data for evaluation) I am taking from two labeled evaluation inboxes and one target
def load_data_spam_multi():
    src1_features, src1_labels ,src1_language= read_data_spam('evaluation_data_labeled/task_b_lab/task_b_u00_eval_lab.tf')
    src2_features, src2_labels,src2_language = read_data_spam('evaluation_data_labeled/task_b_lab/task_b_u01_eval_lab.tf')
    tgt_features, tgt_labels,tgt_language = read_data_spam('evaluation_data_labeled/task_b_lab/task_b_u02_eval_lab.tf')
    return src1_features, src1_labels, src2_features, src2_labels, tgt_features, tgt_labels, src1_language, src2_language, tgt_language

# pre preprocess because feda is also a preprocessing step. Prepreprocessing takes care of arranging the features into
# the same order and actually indluding all of the feature space from both the source and target domain as they may differ
# reduces the number of features to the most important ones as otherwise it takes hours to run
# (takes the n most frequently occuring features from all the domains)
def prepreprocess_data_multi(src1_features, src2_features, tgt_features, n):
    vectorizer = DictVectorizer(sparse=False)
    completed_features = vectorizer.fit_transform(src1_features + src2_features + tgt_features)
    feature_count = np.sum(completed_features, 0)
    sorted_features = np.argsort(feature_count)
    print(completed_features.shape)
    src1_features = completed_features[:len(src1_features), sorted_features[-n:]]
    src2_features = completed_features[len(src1_features):len(src1_features)*2, sorted_features[-n:]]
    tgt_features = completed_features[len(src1_features)*2:, sorted_features[-n:]]
    print(src1_features.shape, src1_features.shape, tgt_features.shape)
    return src1_features, src2_features, tgt_features


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


def tgt_learner_svm_multi(src1_domain, src1_domain_labels, src2_domain, src2_domain_labels, eval_features, train_features, eval_labels, train_labels):

    tgt_learner = LinearSVC(max_iter=2000)
    tgt_learner.fit(train_features, train_labels)
    score = tgt_learner.score(eval_features, eval_labels)
    return score


def all_learner_svm_multi(src1_domain, src1_domain_labels, src2_domain, src2_domain_labels, eval_features, train_features, eval_labels, train_labels):

    all_features = np.concatenate((src1_domain, src2_domain, train_features))
    all_labels = np.concatenate((src1_domain_labels, src2_domain_labels, train_labels))
    tgt_learner = LinearSVC(max_iter=2000)
    tgt_learner.fit(all_features, all_labels)
    score = tgt_learner.score(eval_features, eval_labels)
    return score


def src_learner_svm_multi(src1_domain, src1_domain_labels, src2_domain, src2_domain_labels, eval_features, train_features, eval_labels, train_labels):

    all_features = np.concatenate((src1_domain, src2_domain))
    all_labels = np.concatenate((src1_domain_labels, src2_domain_labels))
    tgt_learner = LinearSVC(max_iter=2000)
    tgt_learner.fit(all_features, all_labels)
    score = tgt_learner.score(eval_features, eval_labels)
    return score


def feda_svm_multi(src1_domain, src1_domain_labels, src2_domain, src2_domain_labels, eval_features, train_features, eval_labels, train_labels):

    all_labels = np.concatenate((src1_domain_labels, src2_domain_labels, train_labels))
    aug_features_src1 = np.concatenate((src1_domain, src1_domain, np.zeros(src1_domain.shape), np.zeros(src1_domain.shape)), axis=1)
    aug_features_src2 = np.concatenate((src2_domain, np.zeros(src2_domain.shape), src2_domain, np.zeros(src2_domain.shape)),axis=1)
    aug_features_train = np.concatenate((train_features, np.zeros(train_features.shape), np.zeros(train_features.shape), train_features),axis=1)
    aug_features_eval = np.concatenate((eval_features, np.zeros(eval_features.shape), np.zeros(eval_features.shape), eval_features),axis=1)
    aug_features = np.concatenate((aug_features_src1, aug_features_src2, aug_features_train))
    feda_learner = LinearSVC(max_iter=2000) # feda is agnostic to the underlying
    # learning algorithm.
    feda_learner.fit(aug_features, all_labels)
    score = feda_learner.score(aug_features_eval, eval_labels)
    return score


def feda_svm_mod_multi(src1_domain, src1_domain_labels, src2_domain, src2_domain_labels, eval_features, train_features, eval_labels, train_labels, src1_similarity, src2_similarity):

    all_labels = np.concatenate((src1_domain_labels, src2_domain_labels, train_labels))
    src1_weights = np.full(len(src1_domain_labels),src1_similarity)
    src2_weights = np.full(len(src2_domain_labels), src2_similarity)
    train_weights = np.ones(len(train_labels))
    all_weights = np.concatenate((src1_weights,src2_weights,train_weights))
    # print(all_weights)
    aug_features_src1 = np.concatenate((src1_domain, src1_domain, np.zeros(src1_domain.shape), np.zeros(src1_domain.shape)), axis=1)
    aug_features_src2 = np.concatenate((src2_domain, np.zeros(src2_domain.shape), src2_domain, np.zeros(src2_domain.shape)),axis=1)
    aug_features_train = np.concatenate((train_features, np.zeros(train_features.shape), np.zeros(train_features.shape), train_features),axis=1)
    aug_features_eval = np.concatenate((eval_features, np.zeros(eval_features.shape), np.zeros(eval_features.shape), eval_features),axis=1)
    aug_features = np.concatenate((aug_features_src1, aug_features_src2, aug_features_train))
    feda_learner = LinearSVC(max_iter=2000) # feda is agnostic to the underlying
    # learning algorithm.
    feda_learner.fit(aug_features, all_labels, sample_weight=all_weights)
    score = feda_learner.score(aug_features_eval, eval_labels)
    return score

def feda_log_reg_multi(src1_domain, src1_domain_labels, src2_domain, src2_domain_labels, eval_features, train_features, eval_labels, train_labels):

    all_labels = np.concatenate((src1_domain_labels, src2_domain_labels, train_labels))
    aug_features_src1 = np.concatenate((src1_domain, src1_domain, np.zeros(src1_domain.shape), np.zeros(src1_domain.shape)), axis=1)
    aug_features_src2 = np.concatenate((src2_domain, np.zeros(src2_domain.shape), src2_domain, np.zeros(src2_domain.shape)),axis=1)
    aug_features_train = np.concatenate((train_features, np.zeros(train_features.shape), np.zeros(train_features.shape), train_features),axis=1)
    aug_features_eval = np.concatenate((eval_features, np.zeros(eval_features.shape), np.zeros(eval_features.shape), eval_features),axis=1)
    aug_features = np.concatenate((aug_features_src1, aug_features_src2, aug_features_train))
    feda_learner = LogisticRegression(solver='liblinear', max_iter= 200) # feda is agnostic to the underlying
    # learning algorithm.
    feda_learner.fit(aug_features, all_labels)
    score = feda_learner.score(aug_features_eval, eval_labels)
    return score


def feda_log_reg_mod_multi(src1_domain, src1_domain_labels, src2_domain, src2_domain_labels, eval_features, train_features, eval_labels, train_labels, src1_similarity, src2_similarity):

    all_labels = np.concatenate((src1_domain_labels, src2_domain_labels, train_labels))
    src1_weights = np.full(len(src1_domain_labels),src1_similarity)
    src2_weights = np.full(len(src2_domain_labels), src2_similarity)
    train_weights = np.ones(len(train_labels))
    all_weights = np.concatenate((src1_weights,src2_weights,train_weights))
    aug_features_src1 = np.concatenate((src1_domain, src1_domain, np.zeros(src1_domain.shape), np.zeros(src1_domain.shape)), axis=1)
    aug_features_src2 = np.concatenate((src2_domain, np.zeros(src2_domain.shape), src2_domain, np.zeros(src2_domain.shape)),axis=1)
    aug_features_train = np.concatenate((train_features, np.zeros(train_features.shape), np.zeros(train_features.shape), train_features),axis=1)
    aug_features_eval = np.concatenate((eval_features, np.zeros(eval_features.shape), np.zeros(eval_features.shape), eval_features),axis=1)
    aug_features = np.concatenate((aug_features_src1, aug_features_src2, aug_features_train))
    feda_learner = LogisticRegression(C=1, solver='liblinear', max_iter= 200) # feda is agnostic to the underlying
    # learning algorithm.
    feda_learner.fit(aug_features, all_labels, sample_weight=all_weights)
    score = feda_learner.score(aug_features_eval, eval_labels)
    return score


src1_features, src1_labels, src2_features, src2_labels, tgt_features, tgt_labels, src1_language, src2_language, tgt_language = load_data_spam_multi()
src1_similarity = jaccard_similarity(list(map(int,src1_language.keys())),list(map(int,tgt_language.keys())))
src2_similarity = jaccard_similarity(list(map(int,src2_language.keys())),list(map(int,tgt_language.keys())))
print(src1_similarity)
print(src2_similarity)
src1_features, src2_features, tgt_features = prepreprocess_data_multi(src1_features, src2_features, tgt_features, 10000)


print('spam train to spam eval using {} labeled target examples'.format(test_size))
fm_svm = []
fm_mod_svm = []
tm_svm = []
am_svm = []
sm_svm = []
fm_logreg = []
fm_mod_logreg = []
for i in range(0, n):
    eval_features, train_features, eval_labels, train_labels = training_samples(tgt_features, tgt_labels,test_size)  # literally just a utility to make the test selections.
    fm_svm.append(feda_svm_multi(src1_features, src1_labels, src2_features, src2_labels, eval_features, train_features, eval_labels, train_labels))
    fm_mod_svm.append(feda_svm_mod_multi(src1_features, src1_labels, src2_features, src2_labels, eval_features, train_features,eval_labels, train_labels, src1_similarity, src2_similarity))
    fm_logreg.append(feda_log_reg_multi(src1_features, src1_labels, src2_features, src2_labels, eval_features, train_features, eval_labels, train_labels))
    fm_mod_logreg.append(feda_log_reg_mod_multi(src1_features, src1_labels, src2_features, src2_labels, eval_features, train_features,eval_labels, train_labels, src1_similarity, src2_similarity))
    tm_svm.append(tgt_learner_svm_multi(src1_features, src1_labels, src2_features, src2_labels, eval_features, train_features, eval_labels, train_labels))
    am_svm.append(all_learner_svm_multi(src1_features, src1_labels, src2_features, src2_labels, eval_features, train_features, eval_labels, train_labels))
    sm_svm.append(src_learner_svm_multi(src1_features, src1_labels, src2_features, src2_labels, eval_features, train_features,eval_labels, train_labels))
print('SVM FEDA:', fm_svm)
print(np.average(fm_svm))
print(np.std(fm_svm))
print('SVM FEDA mod:', fm_mod_svm)
print(np.average(fm_mod_svm))
print(np.std(fm_mod_svm))
print('LR FEDA:', fm_logreg)
print(np.average(fm_logreg))
print(np.std(fm_logreg))
print('LR FEDA mod:', fm_mod_logreg)
print(np.average(fm_mod_logreg))
print(np.std(fm_mod_logreg))
print('SVM All:', am_svm)
print(np.average(am_svm))
print(np.std(am_svm))
print('SVM Target:', tm_svm)
print(np.average(tm_svm))
print(np.std(tm_svm))
print('SVM Source:', sm_svm)
print(np.average(sm_svm))
print(np.std(sm_svm))