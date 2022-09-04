#import library
import pandas as pd
import seaborn as sns
import re
from twython import Twython
import json
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import numpy as np
import gmplot
import webbrowser

#Load API Keys
with open('twitter_credentials.json','r') as file:
    creds = json.load(file)

python_tweets = Twython(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])

#Get the Tweets from the API on different dates
dates =["2022-03-26","2022-03-25","2022-03-24","2022-03-23","2022-03-22","2022-03-21","2022-03-20","2022-03-19"]

dict_test = {'user':[], 'date':[], 'text':[], 'favorite_count': [], 'location': [], 'verified':[], 
         'protected':[], 'retweet':[], 'source':[], 'coordinates':[], 'timezone':[],
         'geo':[], 'place':[], 'Followers':[], 'Account Created': []}

for i in dates:
    query = {'q' : 'SpringStatement', 'result_type' : 'mixed', 'count' : 100, 'lang' : 'en', "until":i}
    sample_return = python_tweets.search(**query)    
    for status in python_tweets.search(**query)['statuses']:
        dict_test['user'].append(status['user']['screen_name'])
        dict_test['date'].append(status['created_at'])
        dict_test['text'].append(status['text'])
        dict_test['favorite_count'].append(status['favorite_count'])
        dict_test['location'].append(status['user']['location'])
        dict_test['verified'].append(status['user']['verified'])
        dict_test['protected'].append(status['user']['protected'])
        dict_test['retweet'].append(status['retweet_count'])
        dict_test['source'].append(status['source'])
        dict_test['coordinates'].append(status['coordinates'])
        dict_test['timezone'].append(status['user']['time_zone'])
        dict_test['geo'].append(status['geo'])
        dict_test['place'].append(status['place'])
        dict_test['Followers'].append(status['user']['followers_count'])
        dict_test['Account Created'].append(status['user']['created_at'])
    print(len(sample_return['statuses']))

df_test2 = pd.DataFrame(dict_test)

# df_test2 = pd.read_csv("C:/Users/richa/Desktop/22102875_assessment2/tweet.csv")
# df_test2 = df_test2.drop(['Unnamed: 0'],axis =1)

#Data Manipulation
df_test2['date']= pd.to_datetime(df_test2['date'])
df_test2['Tweet Created'] = df_test2['date'].dt.date

df_test2['Account Created']= pd.to_datetime(df_test2['Account Created'])
df_test2['Date Account Created'] = df_test2['Account Created'].dt.date

df_test2['Account Age'] = df_test2['Tweet Created'] - df_test2['Date Account Created']
df_test2['Account Age'] = df_test2['Account Age'].astype(str)
df_test2.sort_values(by='Account Age', inplace = True, ascending = True)


#When does the trend starts?
a = df_test2.groupby(['Tweet Created']).sum()
a['date'] = a.index
a.plot(x="date", y=["retweet", "favorite_count"], kind="bar")
plt.xticks(rotation=45)


a.plot(x="date", y=["retweet", "favorite_count"], kind="line")
plt.xticks(rotation=45)

#What devices are used?
source_count = df_test2['source'].value_counts()

source_plot = pd.DataFrame(source_count)
source_plot = source_plot.rename(columns={"source": "count"})
source_plot['Source'] = source_plot.index

#Cleaning the source using Regex
temp = []
for i in source_plot['Source']:
    a = re.findall(">.*\w.<", i)
    temp.append(a)
temp = pd.DataFrame(temp)

#Removing the first and last character of the source
source_clean = []
for i in temp[0]:
    source_clean.append(i[1:-1])

source_plot = list(zip(source_count, source_clean))
source_plot = pd.DataFrame(source_plot, columns = ['Count', 'Source Clean'])
sns.barplot(x="Count", y="Source Clean", data = source_plot).set(ylabel = 'Devices')

#Which source to trust
verified = df_test2['verified'].value_counts()
colors = sns.color_palette('pastel')

# create pie chart using matplotlib
plt.pie(verified, labels=['Unverified', 'Verified'], colors=colors, autopct='%.0f%%')
plt.title("Percentage of Verified Accounts")
plt.show()

verified_sum = df_test2.groupby(['verified']).sum()
verified_sum['verified'] = verified_sum.index

verified_sum.plot(x="verified", y=["favorite_count","retweet"], kind="bar")
plt.xticks(rotation=0)

sns.barplot(x = "Followers", y = "verified", data = verified_sum)
verified_sum.plot(x = "verified", y="Followers", kind="bar")

#Cleaning Account Age
df_test2['Account Age Clean'] = df_test2['Account Age'].str.extract('(\d+)')
df_test2['Account Age Clean'] = df_test2['Account Age Clean'].astype(int)

sns.lineplot(x = "Account Age Clean", y = "Followers", data = df_test2).set(xlabel = "Account Age")
sns.histplot(data=df_test2, x="Account Age Clean").set(xlabel = "Account Age")

#Location of the trend
Location = df_test2['location']
Location = Location.dropna()

#Clean the Emoji within the location column
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
                               u"\ufe0f"
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

Location_clean = []
for i in Location:
    Location_clean.append(remove_emoji(i))
    
#Extract the coordinates using the geocode function
geolocator = Nominatim(user_agent="geo")
def geolocate(country):
    try:
        # Geolocate the center of the country
        loc = geolocator.geocode(country)
        # And return latitude and longitude
        return (loc.latitude, loc.longitude)
    except:
        # Return missing value
        return np.nan

coordinates = []
for i in Location_clean:
    coordinates.append(geolocate(i))

#Clean Coordinates that has NA values and plot into Heatmap using gmplot
coordinates = [x for x in coordinates if str(x) != 'nan']
coordinates = pd.DataFrame(coordinates, columns=["Latitude", 'Logitude'])
coordinates = coordinates.dropna()
Lat = coordinates['Latitude']
Long = coordinates['Logitude']
map_plot = gmplot.GoogleMapPlotter(53.81604806664296, -3.0548307614209813, 3)
map_plot.heatmap(Lat, Long)
map_plot.draw("C:/Users/richa/Desktop/Web Social Media Code/heatmap.html")
webbrowser.open_new_tab("heatmap.html")
