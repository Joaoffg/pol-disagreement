# -*- coding: UTF-8 -*-

from collections import Counter
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
import sklearn.metrics
import nltk
import pickle

texts = []
labels = []
test_texts = []
test_labels = []

df = pd.read_csv('train.csv',
    header=None,
    # names=['label', 'text'],
    names=['label', 'text'],
    nrows=50000,
    encoding='UTF-8')

testdf = pd.read_csv('test.csv',
    header=None,
    # names=['label', 'text'],
    names=['label', 'text'],
    nrows=50000,
    encoding='UTF-8')


for i in range (len(df.label)):
    texts.append(df['text'][i])
    labels.append(df['label'][i])

for i in range (len(testdf.label)):
    test_texts.append(testdf['text'][i])
    test_labels.append(testdf['label'][i])
'''
stemmer = nltk.stem.RSLPStemmer()
texts_stemmed = stemmer.stem(texts)
'''

'''
train_y, train_x = load_data_frame('train.csv')

test_y, test_x = load_data_frame('val.csv')
'''
t1 = datetime.now()
vectorizer = TfidfVectorizer(analyzer= 'char', encoding='UTF-8', strip_accents='unicode', ngram_range=(1, 7), min_df=3)
classifier = LinearSVC()
Xs = vectorizer.fit_transform(texts)

test_Xs = vectorizer.fit_transform(test_texts)

print(datetime.now() - t1)
print(Xs.shape)

# sklearn.metrics.make_scorer(sklearn.metrics.cohen_kappa_score)

score = cross_val_score(classifier, Xs, labels, scoring='accuracy', cv=10, n_jobs=1)

print(score)

#model = classifier.fit(Xs, labels)

#predictions = model.predict(test_Xs)

#print(np.mean(predictions == test_labels))



'''
print(datetime.now() - t1)
print(score)
print(sum(score) / len(score))
'''