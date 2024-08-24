import pickle

#load the inverted index in model_queries_21CS30032.bin
inverted_index = pickle.load(open("model_queries_21CS30032.bin", "rb"))

filepath = "queries_21CS30032.txt"

f1 = open(filepath, "r")
f2 = open("Assignment1_21CS30032_result.txt", "w")

curr = []

while(True) :
    line = f1.readline()
    if(line == ""):
        break
    l = line.split()

    # storing the query id
    q_id = int(l[0])
    l.remove(l[0])
    if l[0] in inverted_index :
        curr = inverted_index[l[0]]
    else :
        curr = []
        continue

    # performing the merge operation
    for word in l :
        new = []
        if word not in inverted_index :
            curr = []
            continue
        i = j =0
        while(i < len(curr) and j < len(inverted_index[word])) :
            if curr[i] == inverted_index[word][j] :
                new.append(curr[i])
                i += 1
                j += 1
            elif curr[i] < inverted_index[word][j] :
                i += 1
            else :
                j += 1
        curr = new

    f2.write(str(q_id) + ": " + " ".join([str(x) for x in curr]) + "\n")

# closing the files
f1.close()
f2.close()
