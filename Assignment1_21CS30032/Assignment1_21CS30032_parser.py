import nltk
import sys
import pickle

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer,PorterStemmer

filepath = sys.argv[1]

f = "queries_21CS30032.txt"

file = open(filepath, "r")
punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
f = open(f,"w")

line = file.readline()

while(True) :
    l = line.split()
    query_id = int(l[1])
    content = ""
    line = file.readline()
    if not line.startswith(".W") :
        break
    while(True) :
        line = file.readline()
        if(line.startswith(".I") or line == ""):
            break
        content += line



    for ele in content:
        if ele in punc:
            content = content.replace(ele, " ")

    content = content.lower()
    tokens = nltk.word_tokenize(content)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in tokens if word.lower() not in stop_words]
    lemmatizer = PorterStemmer()
    lemmatized_words = [lemmatizer.stem(word) for word in filtered_words]
    f.write(str(query_id) + "\t" + " ".join(lemmatized_words) + "\n")
    if(line == ""):
        break

f.close()
file.close()
