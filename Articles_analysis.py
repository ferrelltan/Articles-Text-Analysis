# -*- coding: utf-8 -*-
"""
Dataset taken from Kaggle
url = https://www.kaggle.com/szymonjanowski/internet-news-eda

Goal: To classify by article titles whether an article was a top article or not

@author: FerrellFT
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes  import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


pd.set_option('display.max_rows',     20)
pd.set_option('display.max_columns',  20)
pd.set_option('display.width',       800)
pd.set_option('display.max_colwidth', 20)

np.random.seed(1)
os.chdir('C:/Users/FerrellFT/Downloads')

#%%

articles_df = pd.read_csv('articles_data.csv')
articles_df.head()
articles_df.shape
articles_df.columns

#%% Preprocessing

articles_df.engagement_share_count
articles_df['description'] #We can utilize countvectorizer to obtain keywords
articles_df['description'].isnull().values #Identifying NA values for description to drop
articles_df['author'].nunique() #Overlap in authors; maybe popularity can be a factor
articles_df['engagement_comment_plugin_count'].unique()
articles_df['url_to_image'].isnull().values.any()

#Rename url_to_image to have_image and transform url strings into 0,1 values 
#for articles that have/don't have images
articles_df.rename(columns = {'url_to_image': 'have_image'}, inplace = True)
articles_df['have_image'] = pd.notnull(articles_df['have_image']).astype(int)
articles_df['have_image'].unique()

#Replace NA values in description with 'None'
articles_df['description'].fillna('None', inplace = True)
articles_df['description'].isnull().any()

#Drop unwanted features
articles_df.drop(columns=['Unnamed: 0', 'url'], inplace = True)

#Now we can drop rows with NA in the other features
articles_df.isnull().any()
articles_df = articles_df.dropna(axis=0)

articles_df.reset_index()

#%% Visualizations

#Wordcloud
comment_words = ''
stopwords = set(STOPWORDS)
for words in articles_df.title:
    words = str(words)
    tokens = words.split()
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    comment_words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
plt.figure(figsize = (10, 10), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()
#Trump shows up real big lol



#%% Dataset splits
X_train, X_test, y_train, y_test = train_test_split(articles_df['title'], 
                                                    articles_df['top_article'], 
                                                    test_size = .25,
                                                    random_state=0)


#%% Vectorizing

vect = CountVectorizer().fit(X_train)
X_train_vect          = vect.transform(X_train)

print(vect.get_feature_names())

#%% Model Accuracies

X_test_vect = vect.transform(X_test)

#Now with Logistic Regression
log = LogisticRegression()
log.fit(X_train_vect, y_train)
log.predict(X_test_vect)

print(confusion_matrix(log.predict(X_test_vect), y_test))
log_cfmat = plot_confusion_matrix(log, X_test_vect, y_test,
                                  cmap=plt.cm.Blues,
                                  normalize='true')
plt.title('Confusion matrix for Logistic Regression')
plt.show(log_cfmat)
plt.show()
print(log.score(X_train_vect, y_train))
print(log.score(X_test_vect, y_test)) 



#Not bad, but a lot of mistakes still

#Using an NB classifier
NB = GaussianNB()
NB.fit(X_train_vect.toarray(), y_train)
NB_pred = NB.predict(X_test_vect.toarray())

print(confusion_matrix(NB_pred, y_test))
NB_cfmat = plot_confusion_matrix(NB, X_test_vect.toarray(), y_test,
                                  cmap=plt.cm.Blues,
                                  normalize='true')
plt.title('Confusion matrix for Naive Bayes Regression')
plt.show(NB_cfmat)
plt.ylabel("True Top Article")
plt.xlabel("Predicted Top Article")
plt.show()


print(NB.score(X_train_vect.toarray(), y_train))
print(NB.score(X_test_vect.toarray(), y_test)) 
#False positives too high

#Try KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn_trainS = []
knn_testS = []

neighbors = range(1,20)
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_vect, y_train)
    knn_trainS.append(knn.score(X_train_vect, y_train))
    knn_testS.append(knn.score(X_test_vect, y_test))
    
    print("Confusion Matrix with K = ", k)
    print(confusion_matrix(knn.predict(X_test_vect), y_test))

plt.plot(neighbors, knn_trainS, label="Training Accuracy")
plt.plot(neighbors, knn_testS, label="Test Accuracy")
plt.ylabel("Accuracy (R^2)")
plt.xlabel("Number of Neighbors")
plt.grid()
plt.legend()


#Tree Classifier
