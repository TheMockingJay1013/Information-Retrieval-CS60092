# load the sampled_data.csv file and build the index for the data

import pandas as pd
import numpy as np
import pickle
import os
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# load the data
data = pd.read_csv('sampled_data.csv')

# create the index
# index is of the form {word : {cord_uid : frequency}}
inverted_index = {}

stop_words = set(stopwords.words('english'))

for index, row in data.iterrows():
    cord_uid = row['cord_uid']
    abstract = row['abstract']
    if type(abstract) == str:

        abstract = re.sub(r'[^a-zA-Z0-9]', ' ', abstract)

        tokens = nltk.word_tokenize(abstract)
        filtered_words = [word for word in tokens if word.lower() not in stop_words]
        lemmatizer = PorterStemmer()
        lemmatized_words = [lemmatizer.stem(word) for word in filtered_words]
        for word in lemmatized_words:
            if word not in inverted_index:
                inverted_index[word] = {cord_uid : 1}
            else:
                if cord_uid in inverted_index[word]:
                    inverted_index[word][cord_uid] += 1
                else:
                    inverted_index[word][cord_uid] = 1



# save the index as a binary file
with open('model_queries_21CS30032.bin', 'wb') as f:
    pickle.dump(inverted_index, f)
