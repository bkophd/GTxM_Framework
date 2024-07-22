# -*- coding: utf-8 -*-
"""
Created on Apr 25, 2020
@author: Babatunde Kazeem Oladejo
Description: Generate features (V2)
Last updated on Jan 6, 2022
Note: This version is for generating Master Data for BERT.
- As per BERT guidance, standard NLP pre-processing like lemmatization, stemming should not be done
(see: https://stackoverflow.com/questions/63633534/is-it-necessary-to-do-stopwords-removal-stemming-lemmatization-for-text-classif)
- Stopwords: we will keep the standard words e.g. cannot, else, etc (for context) but remove the jargons e.g. azz, xxxxx, yessss, etc.
"""
print('Preparing the GenMasterBERT program environment...')
import sys
import time
import pandas as pd
import re
import spacy
import nltk
from unidecode import unidecode #use in cleanup() to convert unicode code chars to ascii e.g. Tánaiste to Tanaiste
# from nltk.corpus import stopwords #note: had to run >>> nltk.download('stopwords')
# initialize for stop words removal
stop_words = [] # stopwords.words('english')
stopwords_file = open('stopwords_BERT.txt', 'r')
stopwords_list = stopwords_file.readlines()
stopwords_list = [x.strip().lower() for x in stopwords_list]
stop_words.extend(stopwords_list)
stop_words = set(stop_words)
csvfolder = 'data\\'

def cleanupHttpWeb(text):
    text = text.lower()
    text = removeHttpWeb(text)
    text = cleanTweet(text)
    text = removeStopWords(text)
    if (len(text.strip())  == 0):
        text = ' ' #replace None with a single space
    return(text)

def cleanupSpecChar(text):
    text = text.lower()
    text = removeHttpWeb(text)
    text = removeDigitSpecialChar(text)
    text = cleanTweet(text)
    text = removeStopWords(text)
    if (len(text.strip())  == 0):
        text = ' ' #replace None with a single space
    return(text)

def removeHttpWeb(text):
    #convert unicode chars to ascii
    text = unidecode(text)
    #remove @Mentions, #hashtags, URLs and html codes e.g. &amp, &qout etc
    words = text.split()
    #words = filter(lambda x:x[0]!='@', words)
    #words = filter(lambda x:x[0]!='#', words)
    words = list(filter(lambda x:x[0]!='&', words))
    words = list(filter(lambda x:x[0:4]!='http', words))
    text = " ".join(words)
    if (len(text.strip())  == 0):
        text = ' ' #replace None with a single space
    return text

def cleanTweet(text):
    #Replace common abbreviations and slangs
    text = text.replace(' i m ',' i am ')
    text = text.replace(' i ve ',' i have ')
    text = text.replace(' i ll ',' i will ')
    text = text.replace(' i d ',' i had ')
    text = text.replace(' that s ',' that is ')
    text = text.replace(' isn t ',' is not ')
    text = text.replace(' it s ',' it is ')
    text = text.replace(' she s ',' she is ')
    text = text.replace(' he s ',' he is ')
    text = text.replace(' u ',' you ')
    text = text.replace(' ur ',' your ')
    text = text.replace(' b4 ',' before ')
    text = text.replace(' wasnt ',' was not ')
    text = text.replace(' wasn t ',' was not ')
    text = text.replace(' cant ',' can not ')
    text = text.replace(' can t ',' can not ')
    text = text.replace(' couldnt ',' could not ')
    text = text.replace(' could t ',' could not ')
    text = text.replace(' wouldnt ',' would not ')
    text = text.replace(' would t ',' would not ')
    text = text.replace(' dont ',' do not ')
    text = text.replace(' don t ',' do not ')
    text = text.replace(' didnt ',' did not ')
    text = text.replace(' didn t ',' did not ')
    text = text.replace(' let s ',' let us ')
    text = text.replace(' luv ',' love ')
    text = text.replace(' true ',' truth ')
    text = text.replace(' ppl ',' people ')
    text = text.replace(' fb ',' facebook ')
    text = text.replace(' b day ',' birthday ')
    if (len(text.strip())  == 0):
        text = ' ' #replace None with a single space
    return(text)

def removeDigitSpecialChar(text):
    text = re.sub(r'\d', ' ', text) #replace digits with space
    text = re.sub(r'\W', ' ', text) #replace ALL non-word characters, including emojis with space
    return(text)

def removeStopWords(text):
    #remove stop words 
    words = text.split()
    text = " ".join([word for word in words if word not in stop_words])
    #remove words 50 characters (e.g. extended hahahaha)
    words = text.split()
    text = " ".join([word for word in words if (len(word)<50)])
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

def getHashtags(text):
    #text = '' if text is None else text #Avoid None value
    text = " ".join(re.findall(r'#\S*',text)).lstrip()
    text = uniqueValues(text)
    return

def getMentions(text):
    #text = '' if text is None else text
    text = " ".join(re.findall(r'@\S*',text)).lstrip()
    text = uniqueValues(text)
    return text

def getWordCount(text):
    words = text.split()
    return len(words)

# Generate tweets for Ground Truth Data (Expanded SMR)
def genGTrTweets(NumReplyTweets):
    import pyodbc

    dfGTr = pd.DataFrame(columns=['TID', 'OrigTweet', 'CleanTweetNoHttp', 'CleanTweetNoSpecChar', 'HashTags', 'Mentions', 'InReplyTo', 'ArticleTitle', 'CountReplyTweets', 'CountReplyWords', 'CountReplyChars', 'Target', 'Label'])

    print ("Obtaining Record Tweets from SQL Server...")

    # connect to SQL Server and get record tweets to pandas data frame
    conn = pyodbc.connect('DRIVER={SQL Server};SERVER=KAZIM;DATABASE=SMRM;Trusted_Connection=yes')
    sqlCited = 'SELECT CitedID, ArticleTitle, CitedTweet, hashtags, countReply, target, Label FROM vGroundTruth_CitedTweet where lang='+"'en'"
    dfCited = pd.read_sql(sqlCited,conn)

    # rangeCited = 3
    rangeCited = len(dfCited.index)
    for x in range (rangeCited):
        print('Generating CitedTweet and ReplyTweets ' + str(x+1) + ' of ' + str(rangeCited))
        #columns=['TID', 'OrigTweet', 'CleanTweetNoHttp', 'CleanTweetNoSpecChar', 'HashTags', 'Mentions', 'InReplyTo', 'ArticleTitle', 'CountReplyTweets', 'CountReplyWords', 'CountReplyChars', 'Target', 'Label']

        xCitedID = dfCited['CitedID'].loc[x]
        xCitedTweet = dfCited['CitedTweet'].loc[x]
        xHashTags = dfCited['hashtags'].loc[x]
        xArticleTitle = dfCited['ArticleTitle'].loc[x]
        xCountReplyTweets =  dfCited['countReply'].loc[x]
        xTarget = dfCited['target'].loc[x]
        xLabel = dfCited['Label'].loc[x]
        dfGTr.loc[len(dfGTr)] = [
            xCitedID,
            xCitedTweet,
            cleanupHttpWeb(xCitedTweet),
            cleanupSpecChar(xCitedTweet),
            xHashTags,
            getMentions(xCitedTweet),
            None, #No parent tweet id
            xArticleTitle,
            xCountReplyTweets,
            None, #CountReplyChars is not applicable to cited tweets
            None, #CountReplyWords is not applicable to cited tweets
            xTarget,
            xLabel
            ]

        sqlReply = 'SELECT TOP ' + str(NumReplyTweets) + ' ReplyID, ReplyTweet, ReplyHashtag, tweet_size FROM vGroundTruth_ReplyTweet WHERE CitedID=' + str(xCitedID) + ' Order by sel_priority, tweet_size Desc, tweet_datetime'
        # SQL call for selected supporting tweets for the record tweet
        dfReply = pd.read_sql(sqlReply, conn)

        # rangeReply = 5
        rangeReply = len(dfReply.index)
        for y in range (rangeReply):
            yReplyID = dfReply['ReplyID'].loc[y]
            yReplyTweet = dfReply['ReplyTweet'].loc[y]
            yHashTags = dfReply['ReplyHashtag'].loc[y]
            yCountReplyChars = dfReply['tweet_size'].loc[y]
            yCountReplyWords = getWordCount(yReplyTweet)

            dfGTr.loc[len(dfGTr)] = [
                yReplyID,
                yReplyTweet,
                cleanupHttpWeb(yReplyTweet),
                cleanupSpecChar(yReplyTweet),
                yHashTags,
                getMentions(yReplyTweet),
                xCitedID,
                None, #ArticleTitle is not applicable to replies
                None, #CountReplyTweets is not applicable to Replies
                yCountReplyWords,
                yCountReplyChars,
                xTarget,
                xLabel
                ]

        time.sleep(1) #wait a sec in between for stability

    dfGTr.to_csv('GrounTruthBERT.csv', encoding='utf-8')
    return

# Generate tweets for Computational Grounded Theory (CGT)
def genCGTTweets(NumReplyTweets):
    import pyodbc

    dfCGT = pd.DataFrame(columns=['TID', 'OrigTweet', 'CleanTweetNoHttp', 'CleanTweetNoSpecChar', 'HashTags', 'Mentions', 'InReplyTo', 'ArticleTitle', 'CountReplyTweets', 'CountReplyWords', 'CountReplyChars'])

    print ("Obtaining Record Tweets from SQL Server...")

    # connect to SQL Server and get record tweets to pandas data frame
    conn = pyodbc.connect('DRIVER={SQL Server};SERVER=KAZIM;DATABASE=SMRM;Trusted_Connection=yes')
    sqlCited = 'SELECT CitedID, ArticleTitle, CitedTweet, hashtags, countReply FROM vCGT_CitedTweet where lang='+"'en'"
    dfCited = pd.read_sql(sqlCited,conn)

    # rangeCited = 3
    rangeCited = len(dfCited.index)
    for x in range (rangeCited):
        print('Generating CitedTweet and ReplyTweets ' + str(x+1) + ' of ' + str(rangeCited))

        xCitedID = dfCited['CitedID'].loc[x]
        xCitedTweet = dfCited['CitedTweet'].loc[x]
        xHashTags = dfCited['hashtags'].loc[x]
        xArticleTitle = dfCited['ArticleTitle'].loc[x]
        xCountReplyTweets =  dfCited['countReply'].loc[x]
        dfCGT.loc[len(dfCGT)] = [
            xCitedID,
            xCitedTweet,
            cleanupHttpWeb(xCitedTweet),
            cleanupSpecChar(xCitedTweet),
            xHashTags,
            getMentions(xCitedTweet),
            None, #No parent tweet id
            xArticleTitle,
            xCountReplyTweets,
            None, #CountReplyChars is not applicable to cited tweets
            None, #CountReplyWords is not applicable to cited tweets
            ]

        sqlReply = 'SELECT TOP ' + str(NumReplyTweets) + ' ReplyID, ReplyTweet, ReplyHashtag, tweet_size FROM vCGT_ReplyTweet WHERE CitedID=' + str(xCitedID) + ' Order by sel_priority, tweet_size Desc, tweet_datetime'
        # SQL call for selected supporting tweets for the record tweet
        dfReply = pd.read_sql(sqlReply, conn)

        # rangeReply = 5
        rangeReply = len(dfReply.index)
        for y in range (rangeReply):
            yReplyID = dfReply['ReplyID'].loc[y]
            yReplyTweet = dfReply['ReplyTweet'].loc[y]
            yHashTags = dfReply['ReplyHashtag'].loc[y]
            yCountReplyChars = dfReply['tweet_size'].loc[y]
            yCountReplyWords = getWordCount(yReplyTweet)

            dfCGT.loc[len(dfCGT)] = [
                yReplyID,
                yReplyTweet,
                cleanupHttpWeb(yReplyTweet),
                cleanupSpecChar(yReplyTweet),
                yHashTags,
                getMentions(yReplyTweet),
                xCitedID,
                None, #ArticleTitle is not applicable to replies
                None, #CountReplyTweets is not applicable to Replies
                yCountReplyWords,
                yCountReplyChars,
                ]

        time.sleep(1) #wait a sec in between for stability

    dfCGT.to_csv('CGTexpandedSMR_Data.csv', encoding='utf-8')
    print("Done CGT data generation.")
    return


# Run as:
# cd D:\KOPro\PhD\TechDelivery\SourceCode\py37
# e.g. python GenMasterBERT.py 260 (generate tweet files with 100 threaded Supporting Tweet replies)
if __name__ == "__main__":
    #genGTrTweets(sys.argv[1])
    genCGTTweets(sys.argv[1])


