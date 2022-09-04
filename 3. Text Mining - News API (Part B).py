import requests
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from gensim import corpora
from gensim.models import LsiModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from gensim.models.coherencemodel import CoherenceModel
import nltk
import re

#Define the parameters for the News API
secret = "insert secret key"
url_everything = "https://newsapi.org/v2/everything?"
parameters_everything ={
    'q': 'Elon Musk',
    'pagesize': 10,
    'apikey': secret
}

response_everything = requests.get(url_everything, params = parameters_everything)

response_json_everything = response_everything.json()

response_dict = {'author':[], 'source name':[], 'title':[], 'description':[], 'url':[], 'content':[], 'published':[]}

#Extract the required columns from the response variable
for status in response_json_everything['articles']:
    response_dict['author'].append(status['author'])
    response_dict['source name'].append(status['source']['name'])
    response_dict['title'].append(status['title'])
    response_dict['description'].append(status['description'])
    response_dict['url'].append(status['url'])
    response_dict['content'].append(status['content'])
    response_dict['published'].append(status['publishedAt'])
    
df = pd.DataFrame(response_dict)

df.to_csv("Elon Musk Articles.csv")

#Define functions for pre-processing, preparing the corpus, and LSA model.
def preprocess(doc_set):
    en_stop = set(stopwords.words('english'))
    lemma = WordNetLemmatizer()
    texts = []
    for i in doc_set:
        raw = i.lower()
        raw = re.sub(r"[^a-zA-Z0-9]+"," ", raw)
        tokens = nltk.word_tokenize(raw)
        stopped_tokens = [i for i in tokens if not i in en_stop]
        lemma_tokens = [lemma.lemmatize(i,pos=wordnet.VERB) for i in stopped_tokens]
        texts.append(lemma_tokens)
    return texts

def prepare_corpus(doc_clean):
    dictionary = corpora.Dictionary(doc_clean)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    
    return dictionary, doc_term_matrix

def lsa_model(doc_clean, numberoftopic, words):
    dictionary, doc_term_matrix = prepare_corpus(doc_clean)
    lsamodel = LsiModel(doc_term_matrix, num_topics = numberoftopic, id2word = dictionary)
    print(lsamodel.print_topics(num_topics=numberoftopic, num_words = words))

    cm = CoherenceModel(model=lsamodel, corpus=doc_term_matrix, coherence='u_mass')
    coherence = cm.get_coherence()  # get coherence value
    print(coherence)
    
    return lsamodel

def visualiser(data):
    fdist = nltk.FreqDist()
    for i in data:
        fdist[i] +=1
        # for j in i:
        #     fdist[j] +=1
    fdist.plot(30, cumulative=False)
    
    wordcloud = WordCloud(max_font_size=50, max_words = 100, background_color="white").generate_from_frequencies(fdist)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

df = pd.read_csv("Elon Musk Articles.csv")

number_of_topics = 10
words = 10

#Extract the content from each article
content_clean = []
for i in df['url']:
    response = requests.get(url=i)
    page_content = response.text
    soup = BeautifulSoup(page_content, "html.parser")
    text = ''
    for j in soup.find_all("p"):
        text += j.get_text()
    content_clean.append(text)

#manual deleting an article that is redunant
del content_clean[7]

#Perform data pre-proccessing and LSA modelling
text_pre = preprocess(content_clean)
model = lsa_model(text_pre, number_of_topics, words)
    
#Visualization of Word Frequency and Word Cloud
for i in text_pre:
    visualiser(i)

#To find the number of sentence in each article
sent =[]
for i in content_clean:
    sent.append(nltk.sent_tokenize(i))

#Summary article
en_stop = set(stopwords.words('english'))
text = content_clean[2]
word_text =  nltk.word_tokenize(text)
freqTable = dict()
for i in word_text:
    i = i.lower()
    if i in en_stop:
        continue
    if i in freqTable:
        freqTable[i] += 1
    else:
        freqTable[i] = 1
sent_text = nltk.sent_tokenize(text)
sent_value = dict()

for i in sent_text:
    for word, freq in freqTable.items():
        if word in i.lower():
            if i in sent_value:
                sent_value[i] += freq
            else:
                sent_value[i] = freq
sumVal = 0
for i in sent_value:
    sumVal += sent_value[i]

avg = int(sumVal/len(sent_value))

summary = ''
for i in sent_text:
    if (i in sent_value) and (sent_value[i] > (1.2 * avg)):
        summary += " " + i
print(summary)