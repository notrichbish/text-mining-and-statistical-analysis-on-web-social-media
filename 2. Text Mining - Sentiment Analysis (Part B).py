import pandas as pd
import re
from bs4 import BeautifulSoup
from html import unescape
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk

#load data from Twython Streamer
dataset = pd.read_csv("Elon Musk Tweet.csv")
dataset.columns =  ['Hashtags','Text','Screen Name','Name','Location','Source','Verified','Created At','Followers Count','Retweet Count','Coordinates']

#Identify Duplicate
duplicate = dataset[dataset.duplicated()]

df = dataset['Text']

#Convert unescape characters into their original form
dfs = []
for i in df:
    soup = BeautifulSoup(unescape(i), 'lxml')
    dfs.append(soup.text)
dfs = pd.DataFrame(dfs, columns =['text'])

#Removing Emoji
def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

#Using RE to clean the tweet
def cleantext(string):
    string = re.sub(r'[^-9A-Za-z ]', '', string) #Remove 'punctuations' 
    string = re.sub(r'\n','', string)
    string = re.sub(r'http\S+', '', string)
    string = re.sub(r'@[A-Za-z0-9]+', '', string) #removes @mentions
    string = re.sub(r'#','', string) # removes #
    string = re.sub(r'RT[\s]+', '', string)#removes RT
    
    return string

#Apply the remove emoji and cleaning function to the tweet
dfs['text'] = dfs['text'].apply(remove_emoji)
dfs['text'] = dfs['text'].apply(cleantext)

#Define function to get the subjectivity and polarity using TextBlob library
def getSubject(text):
    return TextBlob(text).sentiment.subjectivity  

def getPolar(text):
    return TextBlob(text).sentiment.polarity

#Apply the function to the cleaned tweets
dfs['Subjectivity'] = dfs['text'].apply(getSubject)
dfs['Polarity'] = dfs['text'].apply(getPolar)

#Define a function to determine the sentiment analysis based on the polarity score
def getAnalysis(s):
    if s > 0:
        return "Positive"
    elif s == 0:
        return "Neutral"
    else:
        return "Negative"
dfs['Sentiment'] = dfs['Polarity'].apply(getAnalysis)

#Split into 3 different dataframe based on their sentiment results
Positive = dfs[dfs['Sentiment']== "Positive"]
Negative = dfs[dfs['Sentiment']== "Negative"]
Neutral = dfs[dfs['Sentiment']== "Neutral"]

#Define function for both wordcloud and word frequency
def visualiser(data):
    fdist = nltk.FreqDist()
    for i in data["text"]:
        i = nltk.word_tokenize(i)
        for j in i:
            fdist[j] +=1
    fdist.plot(30, cumulative=False)
 
    wordcloud = WordCloud(max_font_size=50, max_words = 100, background_color="white").generate_from_frequencies(fdist)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

#Apply visualisation function into the dataframes
visualiser(Positive)
visualiser(Negative)
visualiser(Neutral)
visualiser(dfs)

#Visualisation 
sns.scatterplot(data = dfs,  x='Polarity', y='Subjectivity', hue = 'Sentiment')
sns.jointplot(data = dfs,  x='Polarity', y='Subjectivity', hue = 'Sentiment')

#Pie Chart
sentiment_count = dfs['Sentiment'].value_counts()
colors = sns.color_palette('pastel')
plt.pie(sentiment_count, labels=['Neutral','Positive','Negative'], colors=colors, autopct='%.2f%%')
plt.show()

sentiment_count = pd.DataFrame(sentiment_count)
sentiment_count['Index'] = sentiment_count.index
sns.barplot(data = sentiment_count, x = 'Index', y = 'Sentiment').set(xlabel = 'Sentiment', ylabel = 'Count')

