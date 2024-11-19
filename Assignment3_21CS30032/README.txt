Roll no : 21CS30032

library requirements:
click==8.1.7
joblib==1.4.2
nltk==3.9.1
numpy==2.1.3
pandas==2.2.3
PuLP==2.9.0
python-dateutil==2.9.0.post0
pytz==2024.2
regex==2024.11.6
rouge==1.0.1
six==1.16.0
tqdm==4.67.0
tzdata==2024.2

Python version: 3.12.5


Design Details:
1. The summarizer.py file :
    The initial part of the code in this file is mostly preprocessing, first I removed those samples from the dataset which have summary length greater than 200. Then voabulary is created from the text data and stored in a dictionary. Additionaly, another dictionary was also used to calculate the documnet frequency of each word in the vocabulary. The next step was to create the tf-idf matrix for the text data. These were made for each document and then mean was taken to find the vector for the Document_Collection (doc_vec variable). Next define the function for the Red and Rel calculations.

    After this , sample through the dataset articles , make them into sentences vectors, and stores them in a dictionary along with their position and word_length. Next model the ILP problem using the McDonald formulation and solve the problem using the PuLP library. The output variable is used to get the summary and is stored in a summary.txt file.

2. The evaluator.py file :
    The initial part of the code loads the dataset and the summaries from the csv and txt files respectively.
    The next part of the code is the evaluation of the summaries using the Rouge metric
    The Counter library is used to calculate the number of unigram and bigrams and these are then used to find the  Rouge scores for the summaries and the results are the outputed to the console.
    lasty the average Rouge scores are calculated and outputed to the console.
