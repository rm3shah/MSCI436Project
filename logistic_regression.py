# import libraries
import pandas as pd
import numpy as np
import nltk
import re
import csv
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression


def lemmatize_data(train, test):
    # lemmatize ingredients  
    train['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['ingredients']]       
    test['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in test['ingredients']]  

def preprocessing():
    # split data into training and testing dataset
    data = pd.read_json("../data/train.json")
    train, test = train_test_split(data, test_size=0.2, random_state=4381)
    
    # ground truth
    true_val = test['cuisine']
    
    # call function to clean data
    lemmatize_data(train, test)
    return train, test, true_val

def vectorize(train, test):
    # create corpa based clean data
    train_corpus = train['ingredients_string']
    test_corpus = test['ingredients_string']
    
    # convert ingredients to matrix of TF-IDF features
    # ngram_range = # of words in a sequence
    # max_df = max document frequency, ignore words that exceed this frequency
    # token pattern = regexp used, mandatory if analyzer='word'
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range = ( 1 , 1 ),analyzer="word", max_df = .5, token_pattern=r'\w+')

    # return document term matrices fit on respective corpa
    train_tfidf = vectorizer.fit_transform(train_corpus).todense()
    test_tfidf = vectorizer.transform(test_corpus)
    
    return train_tfidf, test_tfidf

def logistic_regression(train_predictor, train_target, test_predictor, true_val):
    model = LogisticRegression()

    # process exhaustive search over specified parameter values for the model
    # do for num_folds
    num_folds = 10
    parameters = {'C':[1, 10]}
    classifier = GridSearchCV(model, parameters, cv=num_folds)

    # fit classification model to data
    classifier = classifier.fit(train_predictor,train_target)

    # make prediction
    prediction = classifier.predict(test_predictor)
    
    # test model accuracy
    print(accuracy_score(true_val, prediction))
    print(classification_report(true_val, prediction))

    # create confusion matrix
    with open('confusion_matrix.csv', 'w') as f:
        f.write(np.array2string(confusion_matrix(true_val, prediction), separator=', '))

# run all functions
train, test, true_val = preprocessing()
train_predictor, test_predictor = vectorize(train, test)
train_target = train['cuisine']
logistic_regression(train_predictor, train_target, test_predictor, true_val)