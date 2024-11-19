import pandas as pd
import numpy as np
import sys
from nltk.tokenize import word_tokenize

dataset_path = sys.argv[1]
pred_path = sys.argv[2]

# load the dataset from the csv file
dataset = pd.read_csv(dataset_path)

# remove the rows from this dataset which has summary length >200 words
dataset = dataset[dataset['highlights'].apply(lambda x: len(x.split()) <= 200)]

# extract the summaries from the pred_path, it is stored in the format:
    # Document i
    # summary
    # <EOS>

with open(pred_path, 'r') as f:
    lines = f.readlines()

summaries = []
summary = []
for line in lines:
    # if the first word is Document, then it is the start of a new summary so continue
    if line.split()[0] == 'Document':
        continue

    if line == '<EOS>\n':
        summaries.append(' '.join(summary))
        summary = []
    else:
        summary.append(line.strip())


# calculate the rouge scores manually

from collections import Counter
rouge1_p = []
rouge1_r = []
rouge1_f = []

rouge2_p = []
rouge2_r = []
rouge2_f = []

from nltk.stem import PorterStemmer
ps = PorterStemmer()




for i in range(len(summaries)) :
    pred_summary = summaries[i]
    true_summary = dataset.iloc[i]['highlights']


    pred_uni = word_tokenize(pred_summary)
    true_uni = word_tokenize(true_summary)

    # stem the words
    pred_uni = [ps.stem(word.lower()) for word in pred_uni]
    true_uni = [ps.stem(word.lower()) for word in true_uni]

    pred_counter = Counter(pred_uni)
    true_counter = Counter(true_uni)

    C = len(pred_uni)
    R = len(true_uni)

    overlap = sum((pred_counter & true_counter).values())

    precision = overlap/C
    recall = overlap/R
    f1 = 0
    if overlap != 0:
        f1 = 2*precision*recall/(precision+recall)

    rouge1_p.append(precision)
    rouge1_r.append(recall)
    rouge1_f.append(f1)


    pred_bi = [' '.join(pred_uni[i:i+2]) for i in range(len(pred_uni)-1)]
    true_bi = [' '.join(true_uni[i:i+2]) for i in range(len(true_uni)-1)]

    pred_bi_counter = Counter(pred_bi)
    true_bi_counter = Counter(true_bi)

    C2 = len(pred_bi)
    R2 = len(true_bi)

    overlap2 = sum((pred_bi_counter & true_bi_counter).values())

    precision2 = overlap2/C2
    recall2 = overlap2/R2
    f2 = 0
    if overlap2 != 0:
        f2 = 2*precision2*recall2/(precision2+recall2)

    rouge2_p.append(precision2)
    rouge2_r.append(recall2)
    rouge2_f.append(f2)


    print("Document ", i+1, " : ")
    print("Rouge-1 Precision : ", rouge1_p[-1], " Recall : ", rouge1_r[-1], " F1 : ", rouge1_f[-1])
    print("Rouge-2 Precision : ", rouge2_p[-1], " Recall : ", rouge2_r[-1], " F1 : ", rouge2_f[-1])
    print()

print("Average Rouge-1 Precision : ", np.mean(rouge1_p), " Recall : ", np.mean(rouge1_r), " F1 : ", np.mean(rouge1_f))
print("Average Rouge-2 Precision : ", np.mean(rouge2_p), " Recall : ", np.mean(rouge2_r), " F1 : ", np.mean(rouge2_f))
print()
