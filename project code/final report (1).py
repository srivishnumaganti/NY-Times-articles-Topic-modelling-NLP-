#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 17:23:48 2020

@author: vishnumaganti
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
import csv
import re
with open('/Users/vishnumaganti/Documents/text _analysis_project/project_texts_articles..txt', newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
        
data = [str(x) for x in data]

#pre processing 
#1.Expand contractions 


CONTRACTION_MAP = {
"'cause": "because",
"'til": "until",
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

def expand_contractions(text):
    for word in text.split():
        if word.lower() in CONTRACTION_MAP:
            text = text.replace(word, CONTRACTION_MAP[word.lower()])
    return(text)


def remove_special_characters(string):
    # remove any leading or trailing spaces using string.strip method
    string = string.strip()
    
    # create a pattern for anything but alpha-numeric characters  ^ = not
    PATTERN = r'[^a-zA-Z0-9 ]'
    filtered_string = re.sub(PATTERN, r'', string)
    return filtered_string

from nltk.stem import PorterStemmer  
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stemmer=PorterStemmer()
    return [stemmer.stem(token) for token in tokens]
    


new_documents=[]
new_documents2=[]
new_documents3=[]

for i in data:
    new_documents.append(expand_contractions(i))

for i in new_documents:
    new_documents2.append(remove_special_characters(i))     
for i in new_documents2:
    new_documents3.append(tokenize(i))
    
preprocessed = [str(x) for x in new_documents3]  

num_features = 4500
num_topics = 5
num_top_words=30

tf_vectorizer = CountVectorizer(lowercase='boolean',max_features=num_features ,stop_words='english',min_df=1,max_df=1.0)
tf=tf_vectorizer.fit_transform(preprocessed)
tf_feature_names = tf_vectorizer.get_feature_names()


def display_topics(model, feature_names, num_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic %d:" % (topic_idx))
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-num_top_words - 1:-1]]))
            
            
            
lda = LatentDirichletAllocation(n_components=num_topics, max_iter=20, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
display_topics(lda, tf_feature_names, num_top_words)


