import numpy as np
import pandas as pd
import sys
import re
import time

dataset_path = sys.argv[1]

# load the dataset from the csv file
dataset = pd.read_csv(dataset_path)

# remove the rows from this dataset which has summary length >200 words
dataset = dataset[dataset['highlights'].apply(lambda x: len(x.split()) <= 200)]

print("The size of the dataset is ", dataset.shape[0])


# vocab stores the frequency of each word in the dataset
vocab = {}
# doc_freq stores the number of documents in which a word is present
doc_freq = {}

# import nltk word tokenizer
from nltk.tokenize import word_tokenize

for i in range(dataset.shape[0]):
    text = dataset.iloc[i]['article']

    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)


    words = word_tokenize(text)

    for word in words:
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1

    for word in set(words):
        if word not in doc_freq:
            doc_freq[word] = 1
        else:
            doc_freq[word] += 1



# # save the vocabulary and document frequency in binary files
# import pickle

# with open('vocab.pkl', 'wb') as f:
#     pickle.dump(vocab, f)

# with open('doc_freq.pkl', 'wb') as f:
#     pickle.dump(doc_freq, f)
# # exit()

# # load the vocabulary and document frequency from the binary files
# import pickle

# with open('vocab.pkl', 'rb') as f:
#     vocab = pickle.load(f)

# with open('doc_freq.pkl', 'rb') as f:
#     doc_freq = pickle.load(f)



# find the tf-idf vector for each document
# term frequency matrix (TF)
tf_matrix = np.zeros((dataset.shape[0], len(vocab)))

# Fill the TF matrix with term frequencies
for i, tex in enumerate(dataset['article']):
    # copy tex to text
    text = tex
    # remove punctuation from text
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)

    words = word_tokenize(text)

    word_count = {}
    for word in words :
        if word not in word_count :
            word_count[word] = 1
        else :
            word_count[word] += 1

    total_words = len(words)
    for j, word in enumerate(vocab):
        if word in word_count:
            tf_matrix[i][j] = 1 + np.log(word_count[word] / total_words)

# Calculate inverse document frequency (IDF) vector
idf_vector = np.log(dataset.shape[0] / np.array([doc_freq[word] for word in vocab]))

# Calculate TF-IDF matrix by element-wise multiplication of TF and IDF vectors
tf_idf = tf_matrix * idf_vector

# Step 5: Normalize each TF-IDF vector to unit length
norms = np.linalg.norm(tf_idf, axis=1, keepdims=True)
tf_idf = np.divide(tf_idf, norms, where=(norms != 0))


print("The shape of the tf-idf matrix is ", tf_idf.shape)

# D_vec stores the mean of the tf-idf vectors of all the documents
D_vec = np.mean(tf_idf, axis=0)

def Rel(ti,pos,D_vec):
    return 1/pos + np.dot(ti,D_vec)

def Red(ti,tj):
    return np.dot(ti,tj)

# set limit on the number of words to be in a summary
K = 200

# we consider sentences as the textual unit
# so we need a function , which takes a document as input, splits it into sentences, computes tf idf of the sentences and we use the McDonald theorem by using ILP to ge the summary

# import sentence tokeniser
from nltk.tokenize import sent_tokenize

def get_summary(doc,D_vec):

    # sent is a list of dictionary , where each dictionary has the following
    # keys: 'sentence', 'tf_idf', 'position', 'number of words'
    sent = []

    # split the document into sentences
    # the document is split into sentences and stored in the list 'sentences'
    sentences = sent_tokenize(doc)

    for i in range(len(sentences)):
        text = sentences[i]
        text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
        words = word_tokenize(text)
        if len(words) == 0:
            continue
        tf_idf = np.zeros(len(vocab))

        for j, word in enumerate(vocab):
            tf = 0
            if word in words:
                tf = 1 + np.log(words.count(word) / len(words))
            idf = np.log(dataset.shape[0] / doc_freq[word])
            tf_idf[j] = tf * idf

        if np.linalg.norm(tf_idf) != 0:
            tf_idf = tf_idf / np.linalg.norm(tf_idf)

        sent.append({'sentence':text, 'tf_idf':tf_idf, 'position':i+1, 'number of words':len(words)})

    # ILP
    # Import the required libraries
    from pulp import LpMaximize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD

    # Define the ILP problem
    model = LpProblem("Integer_Linear_Programming_Example", LpMaximize)

    # Define the decision variables
    x = LpVariable.dicts("x", range(len(sent)), cat="Binary")  # x_i = 1 if the ith sentence is selected in the summary, 0 otherwise

    # another variable y which is also an LP variable and is a 2d matrix of size len(sent) x len(sent)
    # y[i][j] = 1 if ith and jth sentence are selected in the summary, 0 otherwise
    y = LpVariable.dicts("y", (range(len(sent)),range(len(sent))), cat="Binary")

    # Define the objective function
    # maximize the relevance of the summary
    model += lpSum([Rel(sent[i]['tf_idf'],sent[i]['position'],D_vec) * x[i] for i in range(len(sent))])-lpSum([Red(sent[i]['tf_idf'],sent[j]['tf_idf']) * y[i][j] for j in range(i+1,len(sent)) for i in range(len(sent))]), "Objective"

    # Define the constraints
    # the summary should have at most K words
    model += lpSum([sent[i]['number of words'] * x[i] for i in range(len(sent))]) <= K

    for i in range(len(sent)):
        for j in range(i+1,len(sent)):
            model += y[i][j] <= x[i]
            model += y[i][j] <= x[j]
            model += x[i] + x[j] - y[i][j] <= 1



    # Solve the problem
    model.solve(PULP_CBC_CMD(msg=False))

    # return the summary
    summary = ''
    for i in range(len(sent)):
        if x[i].varValue == 1:
            summary += sent[i]['sentence'] + '. '

    return summary


output_file = "summary.txt"
f = open(output_file, "w")

start_time = time.time()

# get summary for all the documents
predicted_summary = []
for i in range(dataset.shape[0]):
    summary = get_summary(dataset.iloc[i]['article'],D_vec)
    f.write("Document " + str(i+1) + "\n")
    f.write(summary + "\n")
    f.write("<EOS>\n")
    predicted_summary.append(summary)
    print("Document ", i+1)

f.close()
print(f"Time taken: {time.time() - start_time:.2f} seconds")
