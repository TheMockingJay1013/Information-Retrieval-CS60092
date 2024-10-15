import pickle
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,PorterStemmer
import re
import sys

# take cli arguments 1. path to xml file 2. path to bin file
xml_file = sys.argv[1]
bin_file = sys.argv[2]



stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# load the inverted index
inverted_index = pickle.load(open(bin_file, 'rb'))


# number of documents N in the collection
data = pd.read_csv('sampled_data.csv')
N = len(data)


document_freq = {}
for word in inverted_index:
    document_freq[word] = len(inverted_index[word])


# load the xml file contents to xml_content
xml_content = open(xml_file, 'r').read()

# Parse the XML content
root = ET.fromstring(xml_content)

# Create a dictionary to store topic_id: query pairs
topics_dict = {}

# Iterate over each topic in the XML
for topic in root.findall('topic'):
    topic_id = topic.get('number')
    query = topic.find('query').text
    topics_dict[topic_id] = query

# perforn tokenization , stopword removal, stemming on the queries
for topic_id in topics_dict:
    query = topics_dict[topic_id]

    # replace non alphanumeric characters with space
    query = re.sub(r'[^a-zA-Z0-9]', ' ', query)

    # tokenize the query
    query = nltk.word_tokenize(query)
    # remove stopwords
    stopwords = set(nltk.corpus.stopwords.words('english'))
    query = [word for word in query if word.lower() not in stopwords]
    # stemming
    query = [stemmer.stem(word) for word in query]

    topics_dict[topic_id] = query





# vectorinsing the documents and the queries
doc_vectors = {}
for index, row in data.iterrows():
    cord_uid = row['cord_uid']
    doc_vectors[cord_uid] = []


# Scheme A : lnc.ltc

A_doc_vectors = {}
# calculate the document vector for each document
for cord_uid in doc_vectors:
    A_doc_vectors[cord_uid] = []
    for word in inverted_index:
        if cord_uid in inverted_index[word]:
            tf = inverted_index[word][cord_uid]
            idf = 1
            A_doc_vectors[cord_uid].append((1 + np.log(tf)) * idf)
        else:
            A_doc_vectors[cord_uid].append(0)
    # normalise the document vector
    if np.linalg.norm(A_doc_vectors[cord_uid]) != 0:
        A_doc_vectors[cord_uid] = A_doc_vectors[cord_uid] / np.linalg.norm(A_doc_vectors[cord_uid])



# calculate the query vector for each query
A_query_vectors = {}
for topic_id in topics_dict:
    A_query_vectors[topic_id] = []
    for word in inverted_index:
        if word in topics_dict[topic_id]:
            tf = topics_dict[topic_id].count(word)
            idf = np.log(N / document_freq[word])
            A_query_vectors[topic_id].append((1 + np.log(tf)) * idf)
        else:
            A_query_vectors[topic_id].append(0)

    # normalise the query vector
    if np.linalg.norm(A_query_vectors[topic_id]) != 0:
        A_query_vectors[topic_id] = A_query_vectors[topic_id] / np.linalg.norm(A_query_vectors[topic_id])



# calculate the cosine similarity between the query and the documents and print to the output file
output_file_A = open('Assignment2_21CS30032_ranked_list_A.txt', 'w')
for topic_id in A_query_vectors:
    query_vector = A_query_vectors[topic_id]
    similarity = {}
    for cord_uid in A_doc_vectors:
        doc_vector = A_doc_vectors[cord_uid]
        similarity[cord_uid] = np.dot(query_vector, doc_vector)
    # sort the documents based on the similarity
    sorted_similarity = sorted(similarity.items(), key=lambda x: x[1], reverse=True)


    # write the results to the output file as <topic_id> : <space separated list of cord_uids>
    output_file_A.write(topic_id + ' : ')
    for cord_uid, sim in sorted_similarity[:50]:
        output_file_A.write(cord_uid + ' ')

    output_file_A.write('\n')

output_file_A.close()


# Scheme B : lnc.Ltc

B_doc_vectors = {}
# calculate the document vector for each document
for cord_uid in doc_vectors:
    B_doc_vectors[cord_uid] = []
    for word in inverted_index:
        if cord_uid in inverted_index[word]:
            tf = inverted_index[word][cord_uid]
            idf = 1
            B_doc_vectors[cord_uid].append((1 + np.log(tf)) * idf)
        else:
            B_doc_vectors[cord_uid].append(0)

    # normalise the document vector
    if np.linalg.norm(B_doc_vectors[cord_uid]) != 0:
        B_doc_vectors[cord_uid] = B_doc_vectors[cord_uid] / np.linalg.norm(B_doc_vectors[cord_uid])

# calculate the query vector for each query
B_query_vectors = {}

for topic_id in topics_dict:
    B_query_vectors[topic_id] = []
    for word in inverted_index:
        if word in topics_dict[topic_id]:
            tf = topics_dict[topic_id].count(word)/(1+np.log(np.average([topics_dict[topic_id].count(word) for word in topics_dict[topic_id]])))
            idf = np.log(N / document_freq[word])
            B_query_vectors[topic_id].append((1 + np.log(tf)) * idf)
        else:
            B_query_vectors[topic_id].append(0)

    # normalise the query vector
    if np.linalg.norm(B_query_vectors[topic_id]) != 0:
        B_query_vectors[topic_id] = B_query_vectors[topic_id] / np.linalg.norm(B_query_vectors[topic_id])


# calculate the cosine similarity between the query and the documents and print to the output file
output_file_B = open('Assignment2_21CS30032_ranked_list_B.txt', 'w')

for topic_id in B_query_vectors:
    query_vector = B_query_vectors[topic_id]
    similarity = {}
    for cord_uid in B_doc_vectors:
        doc_vector = B_doc_vectors[cord_uid]
        similarity[cord_uid] = np.dot(query_vector, doc_vector)
    # sort the documents based on the similarity
    sorted_similarity = sorted(similarity.items(), key=lambda x: x[1], reverse=True)

    # write the results to the output file as <topic_id> : <space separated list of cord_uids>
    output_file_B.write(topic_id + ' : ')
    for cord_uid, sim in sorted_similarity[:50]:
        output_file_B.write(cord_uid + ' ')

    output_file_B.write('\n')

output_file_B.close()

# Scheme C : anc.apc

C_doc_vectors = {}
# calculate the document vector for each document
for cord_uid in doc_vectors:
    C_doc_vectors[cord_uid] = []
    list = [inverted_index[word][cord_uid] for word in inverted_index if cord_uid in inverted_index[word]]
    max_tf = max(list) if list else 0
    for word in inverted_index:
        if cord_uid in inverted_index[word]:
            tf = 0.5 + 0.5 * inverted_index[word][cord_uid] / max_tf
            idf = 1
            C_doc_vectors[cord_uid].append(tf*idf)
        else:
            C_doc_vectors[cord_uid].append(0)

    # normalise the document vector
    if np.linalg.norm(C_doc_vectors[cord_uid]) != 0:
        C_doc_vectors[cord_uid] = C_doc_vectors[cord_uid] / np.linalg.norm(C_doc_vectors[cord_uid])

# calculate the query vector for each query
C_query_vectors = {}

for topic_id in topics_dict:
    C_query_vectors[topic_id] = []
    for word in inverted_index:
        if word in topics_dict[topic_id]:
            tf = 0.5 + 0.5 * topics_dict[topic_id].count(word) / max([topics_dict[topic_id].count(word) for word in topics_dict[topic_id]])
            idf = max(0,(N-document_freq[word])/(document_freq[word]))
            C_query_vectors[topic_id].append(tf*idf)
        else:
            C_query_vectors[topic_id].append(0)

    # normalise the query vector
    if np.linalg.norm(C_query_vectors[topic_id]) != 0:
        C_query_vectors[topic_id] = C_query_vectors[topic_id] / np.linalg.norm(C_query_vectors[topic_id])


# calculate the cosine similarity between the query and the documents and print to the output file
output_file_C = open('Assignment2_21CS30032_ranked_list_C.txt', 'w')

for topic_id in C_query_vectors:
    query_vector = C_query_vectors[topic_id]
    similarity = {}
    for cord_uid in C_doc_vectors:
        doc_vector = C_doc_vectors[cord_uid]
        similarity[cord_uid] = np.dot(query_vector, doc_vector)
    # sort the documents based on the similarity
    sorted_similarity = sorted(similarity.items(), key=lambda x: x[1], reverse=True)
    # write the results to the output file as <topic_id> : <space separated list of cord_uids>
    output_file_C.write(topic_id + ' : ')
    for cord_uid, sim in sorted_similarity[:50]:
        output_file_C.write(cord_uid + ' ')

    output_file_C.write('\n')

output_file_C.close()
