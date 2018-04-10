# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 15:03:19 2018

@author: pig84
"""

import xml.etree.ElementTree as ET
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def main():
    
    train = ET.parse('Restaurants_Train.xml')
    sentences = train.getroot()
    
    #subtask 1
    #text
    test = ET.parse('Restaurants_Test_Data_PhaseA.xml')
    test_sentences = test.getroot()
    
    text = np.array([])
    test_text = np.array([])
    term = []
    print('Creating term dictionary...')
    #create dict
    for sentence in sentences.findall('sentence'):
        for aspectTerms in sentence.findall('aspectTerms'):
            for aspectTerm in aspectTerms.findall('aspectTerm'):
                if aspectTerm.attrib['term'] not in term:
                    term.append(aspectTerm.attrib['term'])
    print('Dictionary lenth :', len(term))
    print('Create training matrix...')
    #create training matrix
    train_term_matrix = np.empty((0, len(term)))
    train_polarity = np.empty((0, 1), dtype = str)
    for sentence in sentences.findall('sentence'):
        for aspectTerms in sentence.findall('aspectTerms'):
            for aspectTerm in aspectTerms.findall('aspectTerm'):
                for i in range(len(term)):
                    if term[i] == aspectTerm.attrib['term']:
                        text = np.append(text, sentence.find('text').text)
                        zeros = np.zeros([len(term)])
                        zeros[i] = 1
                        zeros = zeros.reshape(1, -1)
                        train_term_matrix = np.append(train_term_matrix, zeros, axis = 0)
                        train_polarity = np.append(train_polarity, aspectTerm.attrib['polarity'])
    '''
    print(text.shape)
    print(train_term_matrix.shape)
    print(train_polarity.shape)
    '''
    
    print('Training...')
    #training
    count_vect = CountVectorizer()
    X_train_counts = np.asarray(count_vect.fit_transform(text).todense())
    train_matrix = np.concatenate((X_train_counts, train_term_matrix), axis = 1)
    clf = MultinomialNB().fit(train_matrix, train_polarity)
    
    print('Extracting test data term...')
    #test extract term
    test_term_matrix = np.empty((0, len(term)))
    for sentence in test_sentences.findall('sentence'):
        sentence_split = sentence.find('text').text.replace('.', '').split()
        for i in range(len(sentence_split)):
            if sentence_split[i] in term:
                test_text = np.append(test_text, sentence.find('text').text)
                zeros = np.zeros([len(term)])
                zeros[term.index(sentence_split[i])] = 1
                zeros = zeros.reshape(1, -1)
                test_term_matrix = np.append(test_term_matrix, zeros, axis = 0)
    '''
    print(test_text.shape)
    print(test_term_matrix.shape)
    '''
    print('Predicting...')
    X_test_counts = np.asarray(count_vect.transform(test_text).todense())
    test_matrix = np.concatenate((X_test_counts, test_term_matrix), axis = 1)
    prediction = clf.predict(test_matrix)
    
    print('Calculating accuracy...')
    test_gold = ET.parse('Restaurants_Test_Gold.xml')
    test_gold_sentences = test_gold.getroot()
    
    prediction_index = 0
    correct_count = 0
    wrong_count = 0
    for sentence in test_gold_sentences.findall('sentence'):
        for aspectTerms in sentence.findall('aspectTerms'):
            for aspectTerm in aspectTerms.findall('aspectTerm'):
                if aspectTerm.attrib['term'] in term:
                    if aspectTerm.attrib['polarity'] == prediction[prediction_index]:
                        correct_count += 1
                    else:
                        wrong_count += 1
                    prediction_index += 1
                else:
                    wrong_count += 1
    print('Accuracy :', round((correct_count/(correct_count + wrong_count)*100), 2), '%')
    
if __name__ == '__main__':
    main()