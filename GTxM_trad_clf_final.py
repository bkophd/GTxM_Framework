# Decription: Ground Truth data experiments with traditional classifers: SVM, XGBoost, RandomForest and NaiveBayes
# Created by: Kazeem Oladejo
# Last updated: 04-Feb-2023
# version 1.1
# Description: uses best cross validation model as the production release model, without additional testing cycle

import re
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from pyinstrument import Profiler
import csv
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=sklearn.exceptions.UndefinedMetricWarning)

# parse pyinstrument profiler text for timing
def getTiming(timingText):
    m = re.search('Duration: ', timingText)
    return (timingText[m.end(): m.end()+5])

def runClassifier(exp_name, classifier, data1, data2, data3, data4, ngram1, ngram2, maxFeat, vec):
    # define execution timing profile from the pyInstrument library
    print ('Working on: ' + exp_name + ' ' +classifier+ '...')
    timing = Profiler()

    scoring = {'acc': 'accuracy',
               'prec': 'precision_weighted',
               'recall': 'recall_weighted',
               'f1': 'f1_weighted'}

    test_ratio = 0.20
    corpus = data[data1] #note: data is a global variable from __main__
    if data2 != '':
        corpus = data[data1] + data[data2]
    if data3 != '':
        corpus = data[data1] + data[data2] + data[data3]
    if data4 != '':
        corpus = data[data1] + data[data2] + data[data3] + data[data4]
    corpus = corpus.fillna(value='')
    if vec == 'TFIDF':
        vectorizer = TfidfVectorizer(min_df=2,ngram_range=(ngram1,ngram2),max_features=maxFeat)
    else:
        vectorizer = CountVectorizer(min_df=2,ngram_range=(ngram1,ngram2),max_features=maxFeat)
    data_vec = vectorizer.fit_transform(corpus)
    vec_dtm = pd.DataFrame(data_vec.toarray(), columns=vectorizer.get_feature_names())
    vec_dtm.index = corpus.index
    y = data['target']
    X = vec_dtm
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, stratify=y, random_state=1)

    timing.start()
    if classifier == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    if classifier == 'LinearSVM':
        clf = SVC(kernel='linear')
    if classifier == 'GaussianNB':
        clf = GaussianNB()
    if classifier == 'XGBoost':
        clf = XGBClassifier(objective="multi:softprob")

    # execute test runs
    val_scores = cross_validate(clf, X_train, y_train, scoring=scoring, cv=5, return_train_score=True)

    clf.fit(X_train, y_train)
    test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_prec_recall_f1 = precision_recall_fscore_support(y_test, test_pred, average = 'weighted')

    timing.stop()

    result = [exp_name, classifier, getTiming(timing.output_text()),
                val_scores['test_acc'].mean()*100, val_scores['test_prec'].mean()*100,
                val_scores['test_recall'].mean()*100, val_scores['test_f1'].mean()*100,
                test_acc*100, test_prec_recall_f1[0]*100, test_prec_recall_f1[1]*100, test_prec_recall_f1[2]*100]

    time.sleep(1)
    print ('... done.')
    return result

if __name__ == "__main__":
    # load data
    # data = pd.read_csv('data/GroundTruthTokens.csv')
    # data = pd.read_csv('data/reGTr/reGroundTruthTokens.csv')
    # CGT
    # data = pd.read_csv('data/CGT2/CGT_Token_Count.csv')
    # data = pd.read_csv('data/CGT2/CGT_Token_Score.csv')
    # data = pd.read_csv('data/CGT2/CGT_Token_MixOpt_noHealth.csv')
    # data = pd.read_csv('data/CGT2/CGT_Token_MixOpt_noHealth_FlatPolitics.csv')
    data = pd.read_csv('data/GroundTruthTokens.csv', encoding='ISO-8859-1')
    df = pd.DataFrame(columns=['experiment','classifier','exec_time','val_acc','val_prec','val_recall',
                               'val_f1','test_acc','test_prec','test_recall','test_f1'])

    # df.loc[len(df)] = runClassifier('A1. All Tokens (baseline)','RandomForest','cleanSMRText','','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A1. All Tokens (baseline)','LinearSVM','cleanSMRText','','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A1. All Tokens (baseline)','GaussianNB','cleanSMRText','','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A1. All Tokens (baseline)','XGBoost','cleanSMRText','','','',1,1,20000,'Count')
    #
    # df.loc[len(df)] = runClassifier('A2. Nouns','RandomForest','smrNouns','','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A2. Nouns','LinearSVM','smrNouns','','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A2. Nouns','GaussianNB','smrNouns','','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A2. Nouns','XGBoost','smrNouns','','','',1,1,20000,'Count')
    #
    # df.loc[len(df)] = runClassifier('A3. Verbs','RandomForest','smrVerbs','','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A3. Verbs','LinearSVM','smrVerbs','','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A3. Verbs','GaussianNB','smrVerbs','','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A3. Verbs','XGBoost','smrVerbs','','','',1,1,20000,'Count')
    #
    # df.loc[len(df)] = runClassifier('A4. Named Entities','RandomForest','smrNER','','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A4. Named Entities','LinearSVM','smrNER','','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A4. Named Entities','GaussianNB','smrNER','','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A4. Named Entities','XGBoost','smrNER','','','',1,1,20000,'Count')
    #
    # df.loc[len(df)] = runClassifier('A5. Hashtags','RandomForest','smrHashtags','','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A5. Hashtags','LinearSVM','smrHashtags','','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A5. Hashtags','GaussianNB','smrHashtags','','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A5. Hashtags','XGBoost','smrHashtags','','','',1,1,20000,'Count')
    #
    # df.loc[len(df)] = runClassifier('A6. Mentions','RandomForest','smrMentions','','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A6. Mentions','LinearSVM','smrMentions','','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A6. Mentions','GaussianNB','smrMentions','','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A6. Mentions','XGBoost','smrMentions','','','',1,1,20000,'Count')
    #
    # df.loc[len(df)] = runClassifier('A6. Mentions','RandomForest','smrMentions','','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A6. Mentions','LinearSVM','smrMentions','','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A6. Mentions','GaussianNB','smrMentions','','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A6. Mentions','XGBoost','smrMentions','','','',1,1,20000,'Count')
    #
    # df.loc[len(df)] = runClassifier('A7. Nouns+Verbs','RandomForest','smrNouns','smrVerbs','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A7. Nouns+Verbs','LinearSVM','smrNouns','smrVerbs','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A7. Nouns+Verbs','GaussianNB','smrNouns','smrVerbs','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A7. Nouns+Verbs','XGBoost','smrNouns','smrVerbs','','',1,1,20000,'Count')
    #
    # df.loc[len(df)] = runClassifier('A8. Mentions+Nouns','RandomForest','smrMentions','smrNouns','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A8. Mentions+Nouns','LinearSVM','smrMentions','smrNouns','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A8. Mentions+Nouns','GaussianNB','smrMentions','smrNouns','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A8. Mentions+Nouns','XGBoost','smrMentions','smrNouns','','',1,1,20000,'Count')
    #
    # df.loc[len(df)] = runClassifier('A9. Mentions+NamedEntities','RandomForest','smrMentions','smrNER','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A9. Mentions+NamedEntities','LinearSVM','smrMentions','smrNER','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A9. Mentions+NamedEntities','GaussianNB','smrMentions','smrNER','','',1,1,20000,'Count')
    # df.loc[len(df)] = runClassifier('A9. Mentions+NamedEntities','XGBoost','smrMentions','smrNER','','',1,1,20000,'Count')
    #
    # df.loc[len(df)] = runClassifier('B1. 1-gram+2-gram','RandomForest','smrMentions','smrNouns','','',1,2,20000,'Count')
    # df.loc[len(df)] = runClassifier('B1. 1-gram+2-gram','LinearSVM','smrMentions','smrNouns','','',1,2,20000,'Count')
    # df.loc[len(df)] = runClassifier('B1. 1-gram+2-gram','GaussianNB','smrMentions','smrNouns','','',1,2,20000,'Count')
    # df.loc[len(df)] = runClassifier('B1. 1-gram+2-gram','XGBoost','smrMentions','smrNouns','','',1,2,20000,'Count')
    #
    # df.loc[len(df)] = runClassifier('B2. 1-gram+2-gram+3-gram','RandomForest','smrMentions','smrNouns','','',1,3,20000,'Count')
    # df.loc[len(df)] = runClassifier('B2. 1-gram+2-gram+3-gram','LinearSVM','smrMentions','smrNouns','','',1,3,20000,'Count')
    # df.loc[len(df)] = runClassifier('B2. 1-gram+2-gram+3-gram','GaussianNB','smrMentions','smrNouns','','',1,3,20000,'Count')
    # df.loc[len(df)] = runClassifier('B2. 1-gram+2-gram+3-gram','XGBoost','smrMentions','smrNouns','','',1,3,20000,'Count')
    #
    # df.loc[len(df)] = runClassifier('B3. 2-gram','RandomForest','smrMentions','smrNouns','','',2,2,20000,'Count')
    # df.loc[len(df)] = runClassifier('B3. 2-gram','LinearSVM','smrMentions','smrNouns','','',2,2,20000,'Count')
    # df.loc[len(df)] = runClassifier('B3. 2-gram','GaussianNB','smrMentions','smrNouns','','',2,2,20000,'Count')
    # df.loc[len(df)] = runClassifier('B3. 2-gram','XGBoost','smrMentions','smrNouns','','',2,2,20000,'Count')
    #
    # df.loc[len(df)] = runClassifier('B4. 2-gram+3-gram','RandomForest','smrMentions','smrNouns','','',2,3,20000,'Count')
    # df.loc[len(df)] = runClassifier('B4. 2-gram+3-gram','LinearSVM','smrMentions','smrNouns','','',2,3,20000,'Count')
    # df.loc[len(df)] = runClassifier('B4. 2-gram+3-gram','GaussianNB','smrMentions','smrNouns','','',2,3,20000,'Count')
    # df.loc[len(df)] = runClassifier('B4. 2-gram+3-gram','XGBoost','smrMentions','smrNouns','','',2,3,20000,'Count')
    #
    # df.loc[len(df)] = runClassifier('C1. TFIDF','RandomForest','smrMentions','smrNouns','','',1,1,20000,'TFIDF')
    # df.loc[len(df)] = runClassifier('C1. TFIDF','LinearSVM','smrMentions','smrNouns','','',1,1,20000,'TFIDF')
    # df.loc[len(df)] = runClassifier('C1. TFIDF','GaussianNB','smrMentions','smrNouns','','',1,1,20000,'TFIDF')
    # df.loc[len(df)] = runClassifier('C1. TFIDF','XGBoost','smrMentions','smrNouns','','',1,1,20000,'TFIDF')
    #
    # df.loc[len(df)] = runClassifier('D1. 10k Features','RandomForest','smrMentions','smrNouns','','',1,1,10000,'Count')
    # df.loc[len(df)] = runClassifier('D1. 10k Features','LinearSVM','smrMentions','smrNouns','','',1,1,10000,'Count')
    # df.loc[len(df)] = runClassifier('D1. 10k Features','GaussianNB','smrMentions','smrNouns','','',1,1,10000,'Count')
    # df.loc[len(df)] = runClassifier('D1. 10k Features','XGBoost','smrMentions','smrNouns','','',1,1,10000,'Count')
    #
    # df.loc[len(df)] = runClassifier('D2. 5k Features','RandomForest','smrMentions','smrNouns','','',1,1,5000,'Count')
    # df.loc[len(df)] = runClassifier('D2. 5k Features','LinearSVM','smrMentions','smrNouns','','',1,1,5000,'Count')
    # df.loc[len(df)] = runClassifier('D2. 5k Features','GaussianNB','smrMentions','smrNouns','','',1,1,5000,'Count')
    # df.loc[len(df)] = runClassifier('D2. 5k Features','XGBoost','smrMentions','smrNouns','','',1,1,5000,'Count')
    #
    # df.loc[len(df)] = runClassifier('E1. TFIDF 10k Features, 1ng_2ng_3ng','RandomForest','smrMentions','smrNouns','','',1,3,10000,'TFIDF')
    # df.loc[len(df)] = runClassifier('E1. TFIDF 10k Features, 1ng_2ng_3ng','LinearSVM','smrMentions','smrNouns','','',1,3,10000,'TFIDF')
    # df.loc[len(df)] = runClassifier('E1. TFIDF 10k Features, 1ng_2ng_3ng','GaussianNB','smrMentions','smrNouns','','',1,3,10000,'TFIDF')
    # df.loc[len(df)] = runClassifier('E1. TFIDF 10k Features, 1ng_2ng_3ng','XGBoost','smrMentions','smrNouns','','',1,3,10000,'TFIDF')
    #
    # df.loc[len(df)] = runClassifier('E2. TFIDF 5k Features, 1ng_2ng_3ng','RandomForest','smrMentions','smrNouns','','',1,3,5000,'TFIDF')
    # df.loc[len(df)] = runClassifier('E2. TFIDF 5k Features, 1ng_2ng_3ng','LinearSVM','smrMentions','smrNouns','','',1,3,5000,'TFIDF')
    # df.loc[len(df)] = runClassifier('E2. TFIDF 5k Features, 1ng_2ng_3ng','GaussianNB','smrMentions','smrNouns','','',1,3,5000,'TFIDF')
    # df.loc[len(df)] = runClassifier('E2. TFIDF 5k Features, 1ng_2ng_3ng','XGBoost','smrMentions','smrNouns','','',1,3,5000,'TFIDF')
    #
    # df.loc[len(df)] = runClassifier('E3. TFIDF 20k Features, 1ng_2ng_3ng','RandomForest','smrMentions','smrNouns','','',1,3,20000,'TFIDF')
    # df.loc[len(df)] = runClassifier('E3. TFIDF 20k Features, 1ng_2ng_3ng','LinearSVM','smrMentions','smrNouns','','',1,3,20000,'TFIDF')
    # df.loc[len(df)] = runClassifier('E3. TFIDF 20k Features, 1ng_2ng_3ng','GaussianNB','smrMentions','smrNouns','','',1,3,20000,'TFIDF')
    # df.loc[len(df)] = runClassifier('E3. TFIDF 20k Features, 1ng_2ng_3ng','XGBoost','smrMentions','smrNouns','','',1,3,20000,'TFIDF')
    #
    # df.loc[len(df)] = runClassifier('F1. TFIDF 5k Features, 1ng_2ng','RandomForest','smrMentions','smrNouns','','',1,2,5000,'TFIDF')
    # df.loc[len(df)] = runClassifier('F1. TFIDF Ment+Nouns 5k Feat 1ng_2ng','LinearSVM','smrMentions','smrNouns','','',1,2,5000,'TFIDF')
    # df.loc[len(df)] = runClassifier('F1. TFIDF 5k Features, 1ng_2ng','GaussianNB','smrMentions','smrNouns','','',1,2,5000,'TFIDF')
    # df.loc[len(df)] = runClassifier('F1. TFIDF Ment+Nouns 5k Feat 1ng_2ng','XGBoost','smrMentions','smrNouns','','',1,2,5000,'TFIDF')
    #
    # df.loc[len(df)] = runClassifier('F2. TFIDF 10k Features, 1ng_2ng','RandomForest','smrMentions','smrNouns','','',1,2,10000,'TFIDF')
    # df.loc[len(df)] = runClassifier('F2. TFIDF 10k Features, 1ng_2ng','LinearSVM','smrMentions','smrNouns','','',1,2,10000,'TFIDF')
    # df.loc[len(df)] = runClassifier('F2. TFIDF 10k Features, 1ng_2ng','GaussianNB','smrMentions','smrNouns','','',1,2,10000,'TFIDF')
    # df.loc[len(df)] = runClassifier('F2. TFIDF 10k Features, 1ng_2ng','XGBoost','smrMentions','smrNouns','','',1,2,10000,'TFIDF')
    #
    # df.loc[len(df)] = runClassifier('F3. TFIDF 20k Features, 1ng_2ng','RandomForest','smrMentions','smrNouns','','',1,2,20000,'TFIDF')
    # df.loc[len(df)] = runClassifier('F3. TFIDF 20k Features, 1ng_2ng','LinearSVM','smrMentions','smrNouns','','',1,2,20000,'TFIDF')
    # df.loc[len(df)] = runClassifier('F3. TFIDF 20k Features, 1ng_2ng','GaussianNB','smrMentions','smrNouns','','',1,2,20000,'TFIDF')
    # df.loc[len(df)] = runClassifier('F3. TFIDF 20k Features, 1ng_2ng','XGBoost','smrMentions','smrNouns','','',1,2,20000,'TFIDF')
    #
    # df.loc[len(df)] = runClassifier('G1. TFIDF Ment+Nouns+Adverbs','LinearSVM','smrMentions','smrNouns','smrAdverbs','',1,2,10000,'TFIDF')
    # df.loc[len(df)] = runClassifier('G2. TFIDF Ment+Nouns+Adjectives','LinearSVM','smrMentions','smrNouns','smrAdjectives','',1,2,10000,'TFIDF')
    # df.loc[len(df)] = runClassifier('G3. TFIDF Ment+NER+Adverbs','LinearSVM','smrMentions','smrNER','smrAdverbs','',1,2,10000,'TFIDF')
    # df.loc[len(df)] = runClassifier('G4. TFIDF Ment+NER+Adjectives','LinearSVM','smrMentions','smrNER','smrAdjectives','',1,2,10000,'TFIDF')
    #
    # df.loc[len(df)] = runClassifier('H1. TFIDF Hash+Nouns+Adverbs','LinearSVM','smrHashtags','smrNouns','smrAdverbs','',1,2,10000,'TFIDF')
    # df.loc[len(df)] = runClassifier('H2. TFIDF Hash+Nouns+Adjectives','LinearSVM','smrHashtags','smrNouns','smrAdjectives','',1,2,10000,'TFIDF')
    # df.loc[len(df)] = runClassifier('H3. TFIDF Hash+NER+Adverbs','LinearSVM','smrHashtags','smrNER','smrAdverbs','',1,2,10000,'TFIDF')
    # df.loc[len(df)] = runClassifier('H4. TFIDF Hash+NER+Adjectives','LinearSVM','smrHashtags','smrNER','smrAdjectives','',1,2,10000,'TFIDF')
    #111

# Baseline
    df.loc[len(df)] = runClassifier('I1. TFIDF Nouns+Adverbs 5k Feat 1-2ng','LinearSVM','smrNouns','smrAdverbs','','',1,2,5000,'TFIDF')
    df.loc[len(df)] = runClassifier('I1. TFIDF Nouns+Adverbs 5k Feat 1-2ng','RandomForest','smrNouns','smrAdverbs','','',1,2,5000,'TFIDF')
    df.loc[len(df)] = runClassifier('I1. TFIDF Nouns+Adverbs 5k Feat 1-2ng','GaussianNB','smrNouns','smrAdverbs','','',1,2,5000,'TFIDF')
    df.loc[len(df)] = runClassifier('I1. TFIDF Nouns+Adverbs 5k Feat 1-2ng','XGBoost','smrNouns','smrAdverbs','','',1,2,5000,'TFIDF')

    df.loc[len(df)] = runClassifier('I2. TFIDF Nouns+Adjectives 5k Feat 1-2ng','LinearSVM','smrNouns','smrAdjectives','','',1,2,5000,'TFIDF')
    df.loc[len(df)] = runClassifier('I2. TFIDF Nouns+Adjectives 5k Feat 1-2ng','RandomForest','smrNouns','smrAdjectives','','',1,2,5000,'TFIDF')
    df.loc[len(df)] = runClassifier('I2. TFIDF Nouns+Adjectives 5k Feat 1-2ng','GaussianNB','smrNouns','smrAdjectives','','',1,2,5000,'TFIDF')
    df.loc[len(df)] = runClassifier('I3. TFIDF Nouns+Adjectives 5k Feat 1-2ng','XGBoost','smrNouns','smrAdjectives','','',1,2,5000,'TFIDF')

    df.loc[len(df)] = runClassifier('J1. TFIDF NER+Adverbs 5k Feat 1-2ng','LinearSVM','smrNER','smrAdverbs','','',1,2,5000,'TFIDF')
    df.loc[len(df)] = runClassifier('J1. TFIDF NER+Adverbs 5k Feat 1-2ng','RandomForest','smrNER','smrAdverbs','','',1,2,5000,'TFIDF')
    df.loc[len(df)] = runClassifier('J1. TFIDF NER+Adverbs 5k Feat 1-2ng','GaussianNB','smrNER','smrAdverbs','','',1,2,5000,'TFIDF')
    df.loc[len(df)] = runClassifier('J1. TFIDF NER+Adverbs 5k Feat 1-2ng','XGBoost','smrNER','smrAdverbs','','',1,2,5000,'TFIDF')

    df.loc[len(df)] = runClassifier('J2. TFIDF NER+Adjectives 5k Feat 1-2ng','LinearSVM','smrNER','smrAdjectives','','',1,2,5000,'TFIDF')
    df.loc[len(df)] = runClassifier('J2. TFIDF NER+Adjectives 5k Feat 1-2ng','RandomForest','smrNER','smrAdjectives','','',1,2,5000,'TFIDF')
    df.loc[len(df)] = runClassifier('J2. TFIDF NER+Adjectives 5k Feat 1-2ng','GaussianNB','smrNER','smrAdjectives','','',1,2,5000,'TFIDF')
    df.loc[len(df)] = runClassifier('J2. TFIDF NER+Adjectives 5k Feat 1-2ng','XGBoost','smrNER','smrAdjectives','','',1,2,5000,'TFIDF')

    df.loc[len(df)] = runClassifier('N1. TFIDF Nouns+Adverbs 10k Feat 1-2ng','LinearSVM','smrNouns','smrAdverbs','','',1,2,10000,'TFIDF')
    df.loc[len(df)] = runClassifier('N1. TFIDF Nouns+Adverbs 10k Feat 1-2ng','RandomForest','smrNouns','smrAdverbs','','',1,2,10000,'TFIDF')
    df.loc[len(df)] = runClassifier('N1. TFIDF Nouns+Adverbs 10k Feat 1-2ng','GaussianNB','smrNouns','smrAdverbs','','',1,2,10000,'TFIDF')
    df.loc[len(df)] = runClassifier('N1. TFIDF Nouns+Adverbs 10k Feat 1-2ng','XGBoost','smrNouns','smrAdverbs','','',1,2,10000,'TFIDF')

    df.loc[len(df)] = runClassifier('N2. TFIDF Nouns+Adjectives 10k Feat 1-2ng','LinearSVM','smrNouns','smrAdjectives','','',1,2,10000,'TFIDF')
    df.loc[len(df)] = runClassifier('N2. TFIDF Nouns+Adjectives 10k Feat 1-2ng','RandomForest','smrNouns','smrAdjectives','','',1,2,10000,'TFIDF')
    df.loc[len(df)] = runClassifier('N2. TFIDF Nouns+Adjectives 10k Feat 1-2ng','GaussianNB','smrNouns','smrAdjectives','','',1,2,10000,'TFIDF')
    df.loc[len(df)] = runClassifier('N2. TFIDF Nouns+Adjectives 10k Feat 1-2ng','XGBoost','smrNouns','smrAdjectives','','',1,2,10000,'TFIDF')

    df.loc[len(df)] = runClassifier('N3. TFIDF NER+Adverbs 10k Feat 1-2ng','LinearSVM','smrNER','smrAdverbs','','',1,2,10000,'TFIDF')
    df.loc[len(df)] = runClassifier('N3. TFIDF NER+Adverbs 10k Feat 1-2ng','RandomForest','smrNER','smrAdverbs','','',1,2,10000,'TFIDF')
    df.loc[len(df)] = runClassifier('N3. TFIDF NER+Adverbs 10k Feat 1-2ng','GaussianNB','smrNER','smrAdverbs','','',1,2,10000,'TFIDF')
    df.loc[len(df)] = runClassifier('N3. TFIDF NER+Adverbs 10k Feat 1-2ng','XGBoost','smrNER','smrAdverbs','','',1,2,10000,'TFIDF')

    df.loc[len(df)] = runClassifier('N4. TFIDF NER+Adjectives 10k Feat 1-2ng','LinearSVM','smrNER','smrAdjectives','','',1,2,10000,'TFIDF')
    df.loc[len(df)] = runClassifier('N4. TFIDF NER+Adjectives 10k Feat 1-2ng','RandomForest','smrNER','smrAdjectives','','',1,2,10000,'TFIDF')
    df.loc[len(df)] = runClassifier('N4. TFIDF NER+Adjectives 10k Feat 1-2ng','GaussianNB','smrNER','smrAdjectives','','',1,2,10000,'TFIDF')
    df.loc[len(df)] = runClassifier('N4. TFIDF NER+Adjectives 10k Feat 1-2ng','XGBoost','smrNER','smrAdjectives','','',1,2,10000,'TFIDF')

    #
    # df.loc[len(df)] = runClassifier('K1. TFIDF Ment+Hash+Nouns','LinearSVM','smrMentions','smrHashtags','smrNouns','',1,2,10000,'TFIDF')
    # df.loc[len(df)] = runClassifier('K2. TFIDF Ment+Hash+Nouns+Adv','LinearSVM','smrMentions','smrHashtags','smrNouns','smrAdverbs',1,2,10000,'TFIDF')
    # df.loc[len(df)] = runClassifier('K3. TFIDF Ment+Hash+Nouns+Adj','LinearSVM','smrMentions','smrHashtags','smrNouns','smrAdjectives',1,2,10000,'TFIDF')
    #
    # df.loc[len(df)] = runClassifier('L1. TFIDF Ment+Hash+NER','LinearSVM','smrMentions','smrHashtags','smrNER','',1,2,10000,'TFIDF')
    # df.loc[len(df)] = runClassifier('L2. TFIDF Ment+Hash+NER+Adv','LinearSVM','smrMentions','smrHashtags','smrNER','smrAdverbs',1,2,10000,'TFIDF')
    # df.loc[len(df)] = runClassifier('L3. TFIDF Ment+Hash+NER+Adj','LinearSVM','smrMentions','smrHashtags','smrNER','smrAdjectives',1,2,10000,'TFIDF')

    # df.loc[len(df)] = runClassifier('M1. TFIDF NER+Nouns+Adv 10k Feat 1-2ng','LinearSVM','smrNER','smrNouns','smrAdverbs','',1,2,10000,'TFIDF')
    # df.loc[len(df)] = runClassifier('M2. TFIDF NER+Nouns+Adj 10k Feat 1-2ng','LinearSVM','smrNER','smrNouns','smrAdjectives','',1,2,10000,'TFIDF')
    # df.loc[len(df)] = runClassifier('M3. TFIDF NER+Nouns+Adv 5k Feat 1-2ng','LinearSVM','smrNER','smrNouns','smrAdverbs','',1,2,5000,'TFIDF')
    # df.loc[len(df)] = runClassifier('M4. TFIDF NER+Nouns+Adj 5k Feat 1-2ng','LinearSVM','smrNER','smrNouns','smrAdjectives','',1,2,5000,'TFIDF')
    # df.loc[len(df)] = runClassifier('M5. TFIDF NER+Nouns+Adj 5k Feat 1-2ng','XGBoost','smrNER','smrNouns','smrAdjectives','',1,2,5000,'TFIDF')

    df.to_csv("results/GroundTruth_Traditional/GTr_results_final.csv")
    # df.to_csv("results/reGroundTruth/reGTr_results_final.csv")
    # CGT
    # df.to_csv("results/CGT2_Traditional/CGT_results_count.csv")
    # df.to_csv("results/CGT2_Traditional/CGT_result_MixOpt_noHealth_FlatPolitics.csv")

    print("All done.")
