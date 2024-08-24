import nltk
import sys
import pickle

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer,PorterStemmer



filepath = sys.argv[1]
doc = {}
full = ""


file = open(filepath, "r")
punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''

line = file.readline()


while(True):
    l = line.split()
    doc_id = int(l[1])
    content = ""
    while(True):
        lline = file.readline()
        if(lline.startswith(".W")):
            break
    while(True):
        lline = file.readline()
        if(lline.startswith(".X")):
            break
        content += lline

    for ele in content:
        if ele in punc:
            content = content.replace(ele, " ")

    content = content.lower()
    # doc[doc_id] = content.split()
    tokens = nltk.word_tokenize(content)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in tokens if word.lower() not in stop_words]
    lemmatizer = PorterStemmer()
    lemmatized_words = [lemmatizer.stem(word) for word in filtered_words]
    lemmatized_words = list(set(lemmatized_words))
    doc[doc_id] = lemmatized_words
    full+=content

    while(True):
        line = file.readline()
        if(line == "" or line.startswith(".I")):
            break
    if(line == ""):
        break

# tokenize the full text
tokens = nltk.word_tokenize(full)

# remove stop words
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in tokens if word.lower() not in stop_words]

# lemmatize the words
lemmatizer = PorterStemmer()
lemmatized_words = [lemmatizer.stem(word) for word in filtered_words]

# remove duplicates
lemmatized_words = list(set(lemmatized_words))

# building the inverted index
inverted_index = {}

no_of_docs = len(doc)

for(doc_id, content) in doc.items():
    for word in content :
        if word in lemmatized_words:
            if word not in inverted_index:
                inverted_index[word] = []

            if word in inverted_index:
                if doc_id not in inverted_index[word]:
                    inverted_index[word].append(doc_id)

# print(inverted_index)

with open("model_queries_21CS30032.bin", "wb") as f :
    pickle.dump(inverted_index, f)

file.close()
