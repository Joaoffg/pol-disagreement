# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

texts = []
labels = []
test_texts = []
test_labels = []

df = pd.read_csv('train.csv',
                 header=None,
                 names=['label', 'text'],
                 nrows=50000,
                 encoding='UTF-8')

testdf = pd.read_csv('test.csv',
                     header=None,
                     names=['label', 'text'],
                     nrows=50000,
                     encoding='UTF-8')

for i in range(len(df.label)):
    texts.append(df['text'][i])
    labels.append(df['label'][i])

for i in range(len(testdf.label)):
    test_texts.append(testdf['text'][i])
    test_labels.append(testdf['label'][i])

classifier = Pipeline([('vect', TfidfVectorizer(analyzer='char',
                                                encoding='UTF-8',
                                                strip_accents='unicode',
                                                ngram_range=(1, 7),
                                                min_df=3)),
                       ('clf', LinearSVC(C=1.0))])

classifier = classifier.fit(df.text, df.label)

filename = 'SVM_politics_classifier.sav'
pickle.dump(classifier, open(filename, 'wb'))

predicted_svm = classifier.predict(testdf.text)

print(np.mean(predicted_svm == testdf.label))

np.savetxt('predictions.csv', predicted_svm, fmt='%d', delimiter=',')
