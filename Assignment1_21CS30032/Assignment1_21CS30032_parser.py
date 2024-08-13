import nltk 
import sys
import pickle
nltk.download('punkt_tab')
nltk.download('wordnet')

from nltk.corpus import stopwords
nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer 

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
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    # print(lemmatized_words)
    # exit()
    f.write(str(query_id) + "\t" + " ".join(lemmatized_words) + "\n")
    if(line == ""):
        break

f.close()
file.close()