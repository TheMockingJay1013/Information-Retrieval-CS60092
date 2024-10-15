21CS30032

Name: Navaneeth Shaji

Libraries Requirements :
click==8.1.7
joblib==1.4.2
nltk==3.9.1
numpy==2.1.2
pandas==2.2.3
python-dateutil==2.9.0.post0
pytz==2024.2
regex==2024.9.11
six==1.16.0
tqdm==4.66.5
tzdata==2024.2

Python Version : 3.12.5

Design Details :

Vocaulary length : 10215
Preprocessing : The document files were first passed through a regex and all non-alphanumeric characters were replaced with space. Then using the nltk library, the words were tokenized and stopwords were removed. The words were then stemmed using the PorterStemmer. The words were then converted to lowercase and the vocabulary was created. The vocabulary was then converted to a dictionary with the words as keys and the cord_uids as the values
