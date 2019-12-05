import numpy as np
import random
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import warnings


def rewrite_data_newsgroup(filename):
    newfile = open("20news-bydate/matlab/reformedTrainData.txt","w+")
    curLine= ""
    curLabel = '-1'
    with open(filename) as f:
        for line in f:
            example = line.strip().split(' ')
            if curLabel == example[0]:
                curLine+= example[1] + ":" + example[2] + " "
            else:
                newfile.write(curLine)
                newfile.write('\n')
                curLabel = example[0]
                curLine = example[1] + ":" + example[2] + " "

def rewrite_with_labels(filename, labels, newName):
    newfile = open(newName,"w+")
    with open(filename) as f:
        examples = f.readlines()
    with open(labels) as l:
        labels = l.readlines()
    for i in range(len(examples)):
        line = examples[i].strip()
        label = labels[i].strip()
        newfile.write(label + ' ' + line)
        newfile.write('\n')


def binarize(filename,newName):
    newfile = open(newName,"w+")
    with open(filename) as f:
        examples = f.readlines()
    for i in range(len(examples)):
        line = examples[i].strip().split(' ')
        if (line[-1] == '4') | (line[-1] == '5'):
            newfile.write(examples[i])
            newfile.write('\n')
# rewrite_data_newsgroup('20news-bydate/matlab/train.data')
rewrite_with_labels('20news-bydate/matlab/reformedTrainData.txt','20news-bydate/matlab/train.label','20news-bydate/matlab/trainComplete.txt')
rewrite_with_labels('20news-bydate/matlab/reformedTestData.txt','20news-bydate/matlab/test.label','20news-bydate/matlab/testComplete.txt')