# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 23:29:58 2018

@author: pig84
"""

from sklearn.feature_extraction.text import CountVectorizer
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd

def read_file(filename):
    data = []
    target = []
    with open(filename, newline = '', encoding = "utf-8", errors='ignore') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            item = ""
            for i in range(len(row) - 2):
                item = item + row[2 + i]
            data.append(item)
            target.append(row[1])
    
    return data[1:], target[1:]

def main():
    data, target = read_file('./processed.csv')
    print(type(data))
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(data)
    print(X_train_counts.shape)
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    print(type(X_train_counts))
    df = pd.DataFrame(X_train_tf.toarray())
    df.to_csv('./for_lstm', sep=',')

if __name__ == '__main__':
    main()