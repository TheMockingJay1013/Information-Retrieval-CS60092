Roll No : 21CS30032
Name : Navaneeth Shaji

### Python Version
Python 3.9.19

### Libraries Used:
click==8.1.7
joblib==1.4.2
nltk==3.8.2
regex==2024.7.24
tqdm==4.66.5

### Details of the design

Task A :
1. This is done in the indexer.py file. The objective was to make an inverted index.
2. The inverted index is a dictionary where the key is the word and the value is a list of document ids where the word is present.
3. The words are however stemmed using the Porter Stemmer. There are about **5956** unique words in the dataset after stemming.
4. The inverted index is then stored in a file using the pickle library.

Task B :
1. Similar to the previous task, the extraction of the queries and its stemming is done in the parser.py file
2. A regex is used to extract queries that are only having the .I and .W tags.
3. Each query is pushed to a queries.txt file.

Task C :
