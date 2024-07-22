# -*- coding: utf-8 -*-
"""
Created on Apr 25, 2020
@author: Babatunde Kazeem Oladejo
Description: Generate features (V2)
Last updated on Sep 2, 2020
Note: This version is for generating Master Tokens CSV
"""
print('Preparing the program environment...')
#import os
import sys
import time
#import numpy as np
import pandas as pd
import re
import spacy
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer #note: had to run>>> nltk.download('wordnet')
from nltk.corpus import stopwords #note: had to run >>> nltk.download('stopwords')
from unidecode import unidecode #use in cleanup() to convert unicode code chars to ascii e.g. Tánaiste to Tanaiste 
#initialize for stop words removal, stemming and lemmatization
# run >pip install gensim==3.8.3
from gensim.summarization.summarizer import summarize

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')
stopwords_file = open('stopwords_extra.txt', 'r')
stopwords_list = stopwords_file.readlines()
stopwords_list = [x.strip().lower() for x in stopwords_list]
stop_words.extend(stopwords_list)
stop_words = set(stop_words)
csvfolder = 'data\\'
# run: python -m spacy download en
nlpSpacy = spacy.load('en_core_web_sm')

def summary(text):
    text = removeHashMentWeb(text)
    #doc = nlpSpacy(text)
    #sentences=''
    #i=0
    #for sent in doc.sents:
        #sentences = sentences + re.sub(r'\W', ' ', sent.text) + '. ' #remove all special chars and add a period
        #if len(sent.text) > 3: #increment counter only if the sentence has at least 2 words.
            #i=i+1
    if len(text) > 1000:    #summarize if more than 1000 char.
        text = summarize(text, word_count=1000)
    else:
        text = "Summarization skipped (text is 1000 characters or less)."
    if (len(text.strip())  == 0):
        text = ' ' #replace None with a single space
    return(text)

def cleanup(text):
    text = text.lower()
    text = removeHashMentWeb(text)
    text = cleanTweet(text)
    text = applyLemmaStem(text)
    text = removeStopWords(text)
    if (len(text.strip())  == 0):
        text = ' ' #replace None with a single space
    return(text)

def removeHashMentWeb(text):
    #convert unicode chars to ascii
    text = unidecode(text)
    #remove @Mentions, #hashtags, URLs and html codes e.g. &amp, &qout etc
    words = text.split()
    words = filter(lambda x:x[0]!='@', words)
    words = filter(lambda x:x[0]!='#', words)
    words = filter(lambda x:x[0]!='&', words)
    words = filter(lambda x:x[0:4]!='http', words)
    text = " ".join(words)
    if (len(text.strip())  == 0):
        text = ' ' #replace None with a single space
    return text

def cleanTweet(text):
    #Replace common abbreviations and slangs
    text = text.replace(' luv ',' love ')
    text = text.replace(' true ',' truth ')
    text = text.replace(' ppl ',' people ')
    text = text.replace(' fb ',' facebook ')
    text = text.replace(' men ',' man ') #stemmer normalized women to woman, but not men to man
    text = re.sub(r'\d', ' ', text) #replace digits with space
    text = re.sub(r'\W', ' ', text) #replace ALL non-word characters, including emojis with space
    if (len(text.strip())  == 0):
        text = ' ' #replace None with a single space
    return(text)

def applyLemmaStem(text):
    #tokenize
    words = nltk.word_tokenize(text)
    #lemmatize words
    #words = text.split()
    text = " ".join([lemmatizer.lemmatize(word) for word in words])
    #stem words
    words = text.split()
    text = " ".join([stemmer.stem(word) for word in words])
    if (len(text.strip())  == 0):
        text = ' ' #replace None with a single space
    return(text)

def removeStopWords(text):
    #remove stop words 
    words = text.split()
    text = " ".join([word for word in words if word not in stop_words])
    #remove words with 1 character and longer 50 characters (e.g. extended hahahaha)
    words = text.split()
    text = " ".join([word for word in words if (len(word)>1 and len(word)<50)])
    if (len(text.strip())  == 0):
        text = ' ' #replace None with a single space
    return (text)

def uniqueValues(text):
    #returns only 1 word if the word exists more than once in the text... uses the set() function to achieve this.
    if text is None:
        return ' '
    else:
        text = unidecode(text).lower()
        # remove special characters
        for ch in [',','!','.','(',')','[',']','?','#','*','\'']:
            if ch in text:
                text = text.replace(ch,'')
        # re-assemble the text
        words = text.split()
        text = " ".join(set(words))
        text = text.lstrip()
        if (len(text.strip())  == 0):
            text = ' ' #replace None with a single space
    return text

def getNamedEntities(text):
    text = removeHashMentWeb(text)
    doc = nlpSpacy(text)
    entitiesText = ''
    for token in doc.ents:
        entitiesText = entitiesText + ' '+ token.text
    entitiesText = cleanTweet(entitiesText)
    entitiesText = applyLemmaStem(entitiesText)
    entitiesText = removeStopWords(entitiesText)
    if (len(entitiesText.strip())  == 0):
        entitiesText = ' ' #replace None with a single space    
    return entitiesText.lower()

def getNouns(text):
    text = removeHashMentWeb(text)
    doc = nlpSpacy(text)
    nounsText = ''
    for token in doc:
        if token.pos_ == 'NOUN' or token.pos_ == 'PROPN':
            nounsText = nounsText + ' '+ token.text
    nounsText = cleanTweet(nounsText)
    nounsText = applyLemmaStem(nounsText)
    nounsText = removeStopWords(nounsText)
    if (len(nounsText.strip())  == 0):
        nounsText = ' ' #replace None with a single space        
    return nounsText.lower()

def getVerbs(text):
    text = removeHashMentWeb(text)
    doc = nlpSpacy(text)
    verbsText = ''
    for token in doc:
        if token.pos_ == 'VERB':
            verbsText = verbsText + ' '+ token.text
    verbsText = cleanTweet(verbsText)
    verbsText = applyLemmaStem(verbsText)
    verbsText = removeStopWords(verbsText)
    if (len(verbsText.strip())  == 0):
        verbsText = ' ' #replace None with a single space        
    return verbsText.lower()

def getAdverbs(text):
    text = removeHashMentWeb(text)
    doc = nlpSpacy(text)
    adverbText = ''
    for token in doc:
        if token.pos_ == 'ADV':
            adverbText = adverbText + ' '+ token.text
    adverbText = cleanTweet(adverbText)
    adverbText = applyLemmaStem(adverbText)
    adverbText = removeStopWords(adverbText)
    if (len(adverbText.strip())  == 0):
        adverbText = ' ' #replace None with a single space
    return adverbText.lower()

def getAdjectives(text):
    text = removeHashMentWeb(text)
    doc = nlpSpacy(text)
    adjectiveText = ''
    for token in doc:
        if token.pos_ == 'ADJ':
            adjectiveText = adjectiveText + ' '+ token.text
    adjectiveText = cleanTweet(adjectiveText)
    adjectiveText = applyLemmaStem(adjectiveText)
    adjectiveText = removeStopWords(adjectiveText)
    if (len(adjectiveText.strip())  == 0):
        adjectiveText = ' ' #replace None with a single space
    return adjectiveText.lower()

def genTweets(NumSupTweets):
    import pyodbc
    import emoji
    #from emoji.unicode_codes import UNICODE_EMOJI
    EMOJI_RE = emoji.get_emoji_regexp()
    print ("Obtaining Record Tweets from SQL Server...")
    # connect to SQL Server and get record tweets to pandas data frame
    conn = pyodbc.connect('DRIVER={SQL Server};SERVER=KAZIM;DATABASE=SMRM;Trusted_Connection=yes')
    sqlRec = 'SELECT id, PubTitle, RecDoc, hashtags FROM vMasterData where lang='+"'en'"
    #sqlRec = 'SELECT id, parsed_created_at, RecDoc, hashtags, possibly_sensitive, user_verified FROM RecTweets where lang='+"'en'"
    #sqlRec = 'SELECT id, parsed_created_at, RecDoc, hashtags FROM RecTweets where lang='+"'en'" + ' and id in (1176288623550853120,1178606452795162624,1178745116758020097,1178962636278382592,1178984443366608908,1179061482966962177,1179081260943196161,1181299742980132870,1181468216612311041,1181620310946304003,1182345051172724736,1182661778498637829,1182755986622369792,1183021805570801665,1183074125994188801,1183707651252899841,1184110357171855362,1184672410836054017,1185091442969964546,1185187045150511110,1185206416409333761,1185507039008698368,1185726930764611589,1186901847690432512)'
    dfRec = pd.read_sql(sqlRec,conn)
    #Get Mentions from Rec tweets and add to Data Frame
    dfRec['mentions'] = dfRec['RecDoc'].apply(lambda x: " ".join(re.findall(r'@\S*',x)).replace('@','').lstrip()) 
    #Get emojis from Rec tweets. Add to Data Frame
    #dfRec['emojis'] = dfRec['RecDoc'].apply(lambda x: " ".join(EMOJI_RE.findall(x))) 
    #inplace apply cleanup() with RecDoc as parameter
    dfRec['cleantext'] = dfRec['RecDoc'].apply(cleanup) 
    #Get Support Tweets and write to file
    dfRec['supText'] = dfRec['RecDoc']
    dfRec['cleanSupText'] = dfRec['cleantext']
    dfRec['supHashtags'] = dfRec['hashtags'].apply(lambda x:'' if x is None else x)  #Avoid None value in the Series
    dfRec['supMentions'] = dfRec['mentions'].apply(lambda x:'' if x is None else x)
    #dfRec['smrSummary'] = dfRec['RecDoc'].apply(summary)
    #dfRec['supEmojis'] = dfRec['emojis'].apply(lambda x:'' if x is None else x)
    # rangeRec = 50
    rangeRec = len(dfRec.index)
    for x in range (rangeRec):
        print('Generating Supporting Tweets for RecTweet ' + str(x+1) + ' of ' + str(rangeRec))
        strRecID = str(dfRec['id'].loc[x])
        sqlSup = 'SELECT TOP ' + str(NumSupTweets) + ' id, in_reply_to_status_id, tweet_datetime, text2 as SupDoc, hashtags, possibly_sensitive, sel_priority FROM SupTweets WHERE in_reply_to_status_id=' + str(strRecID) + ' Order by sel_priority, tweet_size Desc, tweet_datetime'
        # SQL call for selected supporting tweets for the record tweet
        dfSup = pd.read_sql(sqlSup, conn)
        supText2 = dfSup['SupDoc']
        #Get Mentions from Rec tweets and add to Data Frame
        dfSup['mentions'] = dfSup['SupDoc'].apply(lambda x: " ".join(re.findall(r'@\S*',x)).replace('@','').lstrip()) 
        #Get emojis from Rec tweets. Add to Data Frame
        dfSup['emojis'] = dfSup['SupDoc'].apply(lambda x: " ".join(EMOJI_RE.findall(x))) 
        #inplace apply cleanup() with SupDoc as parameter
        dfSup['cleantext'] = dfSup['SupDoc'].apply(cleanup)
        clnSupText = dfSup['cleantext']
        supHashtags = dfSup['hashtags'].apply(lambda x:'' if x is None else x)
        supMentions = dfSup['mentions'].apply(lambda x:'' if x is None else x)
        #smrSummary = dfSup['SupDoc'].apply(summary)
        #supEmojis = dfSup['emojis'].apply(lambda x:'' if x is None else x)
        supText2 = ' '.join(supText2.tolist()) # Convert the Series to String (resolution to 'ValueError: Incompatible indexer with Series')
        clnSupText = ' '.join(clnSupText.tolist())
        supHashtags = ' '.join(supHashtags.tolist())
        supMentions = ' '.join(supMentions.tolist())
        #smrSummary = ' '.join(smrSummary.tolist())
        #supEmojis = ' '.join(supEmojis.tolist())
        #Add supTweet values to recTweet
        dfRec.loc[x, 'supText'] = dfRec.loc[x, 'supText'] +' ' + supText2
        dfRec.loc[x, 'cleanSupText'] = dfRec.loc[x, 'cleanSupText'] +' ' + clnSupText
        dfRec.loc[x, 'supHashtags'] = dfRec.loc[x, 'supHashtags'] +' ' + supHashtags
        dfRec.loc[x, 'supMentions'] = dfRec.loc[x, 'supMentions'] +' ' + supMentions
        #dfRec.loc[x, 'smrSummary'] = dfRec.loc[x, 'smrSummary'] +' ' + smrSummary
        #dfRec.loc[x, 'supEmojis'] = dfRec.loc[x, 'supEmojis'] +' ' + supEmojis
        #countSup = len(dfSup)
        dfRec.loc[x, 'countSupTweets'] = len(dfSup)
        #write the SupTweet data frame to CSV file.
        #dfSup.to_csv(csvfolder+strRecID+'.csv', encoding='utf-8') # write supporting tweets to csv
        time.sleep(1) #wait a sec in between for stability
    print("Finalizing Record Hashtags...")
    dfRec['hashtags'] = dfRec['hashtags'].apply(uniqueValues)  #get only the unique values
    print("Finalizing Record Mentions...")
    dfRec['mentions'] = dfRec['mentions'].apply(uniqueValues)
    print("Finalizing Record+Supporting Hashtags...")
    dfRec['smrHashtags'] = dfRec['supHashtags'].apply(uniqueValues)
    print("Finalizing Record+Supporting Mentions...")
    dfRec['smrMentions'] = dfRec['supMentions'].apply(uniqueValues)
    print("Finalizing Record+Supporting Named Entities...")
    dfRec['smrNER'] = dfRec['supText'].apply(getNamedEntities)
    print("Finalizing Record+Supporting Nouns...")
    dfRec['smrNouns'] = dfRec['supText'].apply(getNouns)
    print("Finalizing Record+Supporting Verbs...")
    dfRec['smrVerbs'] = dfRec['supText'].apply(getVerbs)
    print("Finalizing Record+Supporting Adverbs...")
    dfRec['smrAdverbs'] = dfRec['supText'].apply(getAdverbs)
    print("Finalizing Record+Supporting Adjectives...")
    dfRec['smrAdjectives'] = dfRec['supText'].apply(getAdjectives)
    print("Finalizing Record+Supporting Top 6000 Original Text Characters...")
    dfRec['smrTopText'] = dfRec['supText'].str.slice(0,6000,1)
    print("Finalizing Record+Supporting Summarization Text...")
    dfRec['smrSummary'] = dfRec['supText'].apply(summary)
    dfRec.index.names = ['rowid']
    dfRec.rename(columns={'id': 'RecID'}, inplace=True)
    #dfRec['smrToken3000'] = dfRec['cleanSupText'].str.slice(0,3000,1)
    #delete unnecessary colums
    print("Deleting unnecessary dataframe columns...")
    del dfRec['hashtags']
    del dfRec['mentions']
    del dfRec['cleantext']
    del dfRec['supText']
    del dfRec['cleanSupText']
    del dfRec['supHashtags']
    del dfRec['supMentions']
    print("Writing data to MasterTokens.csv...")
    dfRec.to_csv('MasterTokens.csv', encoding='utf-8')
    print ("Completed work on "+str(rangeRec)+" records.")
    return

# Run as:
# cd D:\KOPro\PhD\TechDelivery\SourceCode\py37
# e.g. python GenMasterTokens.py 260 (generate tweet files with 100 threaded Supporting Tweet replies)
if __name__ == "__main__":
        genTweets(sys.argv[1])


