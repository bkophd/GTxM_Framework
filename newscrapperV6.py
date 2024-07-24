"""
#Created on 26 Nov, 2020
#@author: Babatunde Kazeem Oladejo
#@co-author: Tamir K. Oladejo
#Description: News Scrapper V2 for NYT, CNN, EuroNews and DW
#Last updated on 19 Jul, 2021
# Note for V6: Minor fixes

Scrap news websites for articles that include social media links
Use Beautiful Soup library to find article URLs on NYT, CNN, EuroNews, DW websites
and then navigate the article links to find twitter.com/status URLs.
For each twitter URL, write tweetID, tweetURL, sTime, sSource, sURL, sTitle and sCat to CSV file.
"s" stands for source article
"""

# INSTRUCTIONS

# 1. GENERATE HTML PAGES e.g. FOR 20210714.
# Note: Be careful to save the entire page, not just a particular link.
# For NYT, use this: https://www.nytimes.com/search?dropmab=true&query=twitter&sort=best&startDate=20210719&endDate=20210724 , click 'more', save offline NewYorkTimes
# For EuroNews, use this: https://www.euronews.com/search?query=twitter&p=1, save offline EuroNews
# # # For CNN, use this: https://edition.cnn.com/search?q=twitter&size=50&sort=newest , save offline CNN
# For DW, use this: https://www.dw.com/search/?languageCode=en&item=twitter&period=WEEK&sort=DATE&resultsCounter=50 , save offline DW

# 2. RUN THE PROGRAM
# Run e.g. >python newscrapperV6.py 20210830

# 3. ZIP THE DATA
# zip *.csv to newscited_20210714.zip
# upload to Google Docs
# delete *_20210714.*

import sys
import re
import time
import requests
import pandas as pd
import lxml
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import pyautogui

# Extract twitter ID from URLs
def getTweetID(tLink):
    try:
        x, y = tLink.split('status/')
        z = y.rsplit('?')
        t = re.split('/',z[0])
    except:
        t = ['error']
    return(t[0])

# Parse links for twitter.com/status links. Write to CSV file.
def findTweet(sURL, sTime, sTitle, sCat, sSubCat, sSource):
    return

# for NYT
def runNYT(siteurl, file_date):
    df = pd.read_csv("newscited.csv", index_col=False)
    result = requests.get(siteurl)
    source = result.content
    soup = BeautifulSoup(source, features='html.parser')
    mainhtml = soup.find("main")
    resultlines = mainhtml.find_all("li")
    i = 1
    for resultline in resultlines:
        if "css-1l4w6pd" in resultline.attrs['class']:
            print("working on article " + str(i) + "... ")
            sourceURL_A = resultline.find("a")
            sourceURL = sourceURL_A.attrs['href']
            sourceTime = sourceURL[24:34]
            sourceTitle = resultline.find("h4")  # first heading 4 sytle in the resultline contains article title
            sourceCat = resultline.find("p")
            print("working on article " + str(i) + " Source URL: " + sourceURL)
            # findTweet(sourceURL.attrs['href'], sourceTime.attrs['datetime'], sourceTitle, sourceCat, 'NYT')
            # findTweet(sourceURL.attrs['href'], sourceTime.string, sourceTitle.string, sourceCat.string, 'NYT')

            tweetResult = requests.get(sourceURL)
            tweetSource = tweetResult.content
            tweetSoup = BeautifulSoup(tweetSource, features='html.parser')
            linkTweets = tweetSoup.find_all('a')

            for linkTweet in linkTweets:

                try:
                    tweetURL = linkTweet.attrs['href']
                except:  # pass on KeyError or any other error
                    pass

                if "twitter.com" in tweetURL and "status" in tweetURL:
                    # Run getTweetID function
                    tweetID = getTweetID(tweetURL)
                    newdata = [tweetID, sourceTime, 'NYT', sourceURL, sourceTitle.text, sourceCat.text, ""]
                    # Write to dataframe
                    df.loc[len(df)] = newdata
                    print("working on tweetID: " + str(tweetID))

        # dedup and write out df
        #df.to_csv("newscited.csv", index=None)
        df.to_csv("newscited_nyt_" + file_date + ".csv", header=None, index=None)

        i = i + 1
    print("Finished NYT.")
    return

def runEuroNews(siteurl, file_date):
    df = pd.read_csv("newscited.csv", index_col=False)
    result = requests.get(siteurl)
    source = result.content
    soup = BeautifulSoup(source, "lxml")
    article_tags = soup.find_all("article", attrs={"data-partnered" : "Partner content"})
    # print(article_tags)
    i = 1
    for article_tag in article_tags:
        print("working on article " + str(i) + "... ")
        links = article_tag.find("a", attrs={"class" : "m-object__title__link"})["href"]
        # print(links)
        title_tag = article_tag.find("a", attrs={"class" : "m-object__title__link"}).text.strip()
        # print(title_tag)
        try:
            date_tag = article_tag.find("time", attrs={"class" : "m-object__date u-margin-top-2"}).text.strip()
        except:
            print("No date found")
            date_tag = "No date found"
            pass
        # print(date_tag)
        try:
            source_cats = article_tag.find("span", attrs={"class" : "program-name"}).text.strip()
        except:
            print("No category found")
            source_cats = "No category found"
            pass
        # print(source_cat)

        tweetResult = requests.get(links)
        tweetSource = tweetResult.content
        tweetSoup = BeautifulSoup(tweetSource, features='html.parser')
        linkTweets = tweetSoup.find_all('a')

        for linkTweet in linkTweets:

            try:
                tweetURL = linkTweet.attrs['href']
            except:  # pass on KeyError or any other error
                pass

            if "twitter.com" in tweetURL and "status" in tweetURL:
                # Run getTweetID function
                tweetID = getTweetID(tweetURL)
                newdata = [tweetID, date_tag, "Euronews", links, title_tag, source_cats,""]
                # Write to dataframe
                df.loc[len(df)] = newdata
                print("working on tweetID: " + str(tweetID))
        i = i + 1
    # dedup and write out df
    #df = df.drop_duplicates(subset='tweetID', keep='last')
    df.to_csv("newscited_euronews_" + file_date + ".csv", header=None, index=None)

    print("Finished Euronews.")
    return

def runCNN(siteurl, file_date):
    df = pd.read_csv("newscited.csv", index_col=False)
    result = requests.get(siteurl)
    source = result.content
    soup = BeautifulSoup(source, "html.parser")
    div_tags = soup.find_all("div", attrs={"class" : "cnn-search__result-contents"})
    # print(div_tags)
    i = 1
    for div_tag in div_tags:
        print("working on article " + str(i) + "... ")
        date_tag = div_tag.find_all("span")[1].text
        links = [a['href'] for a in div_tag.find_all('a', href=True)]
        links = links[0]
        title_tag = div_tag.find("a").text

        tweetResult = requests.get(links)
        tweetSource = tweetResult.content
        tweetSoup = BeautifulSoup(tweetSource, features='html.parser')
        linkTweets = tweetSoup.find_all('a')
        # print( soup.find("div", class_="col-l-4 mtop pagination-number")["aria-label"] )
        try:
            source_cats = tweetSoup.find("a", attrs={"data-test" : "section-link"})["aria-label"]
        except:
            source_cats = ""
        # print(source_cats)

        for linkTweet in linkTweets:

            try:
                tweetURL = linkTweet.attrs['href']
            except:  # pass on KeyError or any other error
                pass

            if "twitter.com" in tweetURL and "status" in tweetURL:
                # Run getTweetID function
                tweetID = getTweetID(tweetURL)
                newdata = [tweetID, date_tag, "CNN", links, title_tag, source_cats, ""]
                # Write to dataframe
                df.loc[len(df)] = newdata
                print("working on tweetID: " + str(tweetID))
        i = i + 1
    #df.to_csv("newscited.csv", index=None)
    df.to_csv("newscited_cnn_" + file_date + ".csv", header=None, index=None)

    print("Finished CNN.")
    return

def runDW(siteurl, file_date):
    df = pd.read_csv("newscited.csv", index_col=False)
    result = requests.get(siteurl)
    source = result.content
    soup = BeautifulSoup(source, "lxml")
    div_tags = soup.find_all("div", attrs={"class" : "searchResult"})
    i = 1
    for div_tag in div_tags:
        print("working on article " + str(i) + "... ")
        date_tag = div_tag.find("span", class_ = "date").text
        title_tag = div_tag.find("h2").text[:-11].strip()
        links = [a['href'] for a in div_tag.find_all('a', href=True)]
        links = links[0]
        # print(links)

        print("working on article " + str(i) + " Source URL: " + links)

        options = Options()
        options.headless = True
        options.add_argument("--log-level=3")
        options.add_argument('--ignore-certificate-errors')
        options.add_argument('--ignore-ssl-errors')
        driver = webdriver.Chrome(options=options)
        try:
            driver.get(links)
            print(" ")
            print("Waiting for Tweets to load...")
            print(" ")
        except:
            print("\nUnable to get the link\n")
            pass
        try:
            iframe = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.ID, "twitter-widget-0")))
            time.sleep(5)
            driver.switch_to.frame(iframe)
        except:
            print(" ")
            print("No Tweets found...")
            print(" ")
            pass
        tweetSource = driver.page_source
        tweetSoup = BeautifulSoup(tweetSource, features='html.parser')
        linkTweets = tweetSoup.find_all('a')

        for linkTweet in linkTweets:
            try:
                tweetURL = linkTweet.attrs['href']
            except:  # pass on KeyError or any other error
                pass
            if "twitter.com" in tweetURL and "status" in tweetURL:
                # Run getTweetID function
                tweetID = getTweetID(tweetURL)
                newdata = [tweetID, date_tag, "DW", links, title_tag, "News", ""]
                # Write to dataframe
                df.loc[len(df)] = newdata
                print("working on tweetID: " + str(tweetID))
        i = i + 1
    #df.to_csv("newscited.csv", index=None)
    df.to_csv("newscited_dw_" + file_date + ".csv", header=None, index=None)

    print("Finished DW.")
    print(" ")
    print("Shutting down ChromeDriver, please wait...")
    return

if __name__ == "__main__":
    # runNYT("http://localhost/newscited/NYT.html", sys.argv[1])
    # time.sleep(2)
    # runEuroNews("http://localhost/newscited/EuroNews.html", sys.argv[1])
    # time.sleep(2)
    # runCNN("http://localhost/newscited/CNN.html", sys.argv[1])
    # time.sleep(2)
    runDW("http://localhost/newscited/DW.html", sys.argv[1])
    # time.sleep(2)
