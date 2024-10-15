
from os import write
import pandas as pd
import numpy as np
import sys

gold_standard_file = sys.argv[1]

# store the relevance in a dictionary as topic_id : {cord_uid : relevance}
relevance_list = {}

with open(gold_standard_file, 'r') as file:
    while True:
        line = file.readline()
        if not line:
            break

        topic_id, iteration, cord_uid, relevance = line.strip().split()
        if topic_id not in relevance_list:
            relevance_list[topic_id] = {cord_uid : int(relevance)}
        else:
            relevance_list[topic_id][cord_uid] = int(relevance)


# function to evaluate the ranked list of documents
def evaluate(input_file,output_file):
    f = open(output_file, 'w')
    avg10 = 0
    avg20 = 0
    ndcg10 = 0
    ndcg20 = 0

    with open(input_file, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break
            # first element is the topic_id
            # rest is the cord_uids
            relevance = []


            topic_id,colon, *cord_uids = line.strip().split()

            # print the topic_id to the output file as Query ID : \n
            f.write(f"Query ID : {topic_id}\n")

            for cord_uid in cord_uids[:20]:
                if(cord_uid in relevance_list[topic_id]):
                    relevance.append(relevance_list[topic_id][cord_uid])
                else:
                    relevance.append(0)


            # calculate average precision @10
            avg = 0
            count = 0
            for i in range(10):
                if relevance[i] != 0:
                    count += 1
                    avg += count/(i+1)
            if count != 0:
                avg/=count
            avg10 += avg

            f.write(f"Average Precision @10 : {avg}\n")

            # calculate average precision @20
            avg = 0
            count = 0
            for i in range(20):
                if relevance[i] != 0:
                    count += 1
                    avg += count/(i+1)

            if count != 0:
                avg/=count
            avg20 += avg

            f.write(f"Average Precision @20 : {avg}\n")


            # calculate NDCG @10
            dcg = 0
            for i in range(10):
                if relevance[i] == 2:
                    dcg = dcg + (np.power(2,2)-1)/np.log(i+2)

                elif relevance[i] == 1:
                    dcg = dcg + (np.power(2,1)-1)/np.log(i+2)

            # ideal relevance from the relevance list
            # sort the relevance list and take the top 10
            ideal_rel = sorted(relevance_list[topic_id].items(), key = lambda x: x[1], reverse = True)
            ideal_rel = [x[1] for x in ideal_rel[:20]]

            idcg = 0
            for i in range(10):
                if ideal_rel[i] == 2:
                    idcg = idcg + (np.power(2,2)-1)/np.log(i+2)

                elif ideal_rel[i] == 1:
                    idcg = idcg + (np.power(2,1)-1)/np.log(i+2)

            ndcg = dcg
            if idcg != 0:
                ndcg = dcg/idcg
            ndcg10 += ndcg

            f.write(f"NDCG @10 : {ndcg}\n")

            # calculate NDCG @20
            dcg = 0
            for i in range(20):
                if relevance[i] == 2:
                    dcg = dcg + (np.power(2,2)-1)/np.log(i+2)

                elif relevance[i] == 1:
                    dcg = dcg + (pow(2,1)-1)/np.log(i+2)

            idcg = 0
            for i in range(20):
                if ideal_rel[i] == 2:
                    idcg = idcg + (pow(2,2)-1)/np.log(i+2)

                elif ideal_rel[i] == 1:
                    idcg = idcg + (pow(2,1)-1)/np.log(i+2)

            ndcg = dcg
            if idcg != 0:
                ndcg = dcg/idcg
            ndcg20 += ndcg

            f.write(f"NDCG @20 : {ndcg}\n")

        # print the average values
        f.write(f"Mean Average Precision @10 : {avg10/50}\n")
        f.write(f"Mean Average Precision @20 : {avg20/50}\n")
        f.write(f"Mean NDCG @10 : {ndcg10/50}\n")
        f.write(f"Mean NDCG @20 : {ndcg20/50}\n")

    f.close()


# input file and evaluate
input_file = sys.argv[2]
letter = input_file[-5]
output_file = f"Assignment2_21CS30032_metrics_{letter}.txt"
evaluate(input_file,output_file)
