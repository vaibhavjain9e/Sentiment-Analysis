# use this file to classify using naive-bayes classifier 
# Expected: generate nboutput.txt


import sys
import glob
import os
import re
import numpy as np
import string


def tokenization(textfile):
    textfile = re.sub(r'\.-', " ", textfile)
    words = re.sub('[^a-zA-Z ]', "", textfile)
    words = words.split()
    words = [word.lower() for word in words]
    stop_words = ['friends', 'knickerbocker', 'things', 'swissotel', 'ive', 'sheraton',  'im', 'see', 'hyatt', 'morning', 'going', 'suite', 'michigan', 'door', 'say', 'etc', 'am', 'pm', ' ', '', 'si', 'st', 'sw', 'sp', 'sq', 'rm', 're', 'rd', 'saw', 'nights', 'may', 'done', 'also',
                  'given', 'went', 'could', 'th', 'affina', 'already',
                  'hotels', 'got', 'go', 'two', 'would', 'staff', 'service', 'one', 'stay', 'staying', 'stayed',
                  'rooms', 'room', 'bed', 'beds', 'affinia', 'us', 'chicago', 'hotel', 'room', 'ie', 'i', 'me', 'my',
                  'myself', 'we', 'our', 'ours', 'ourselves', 'you', "youre", "youve", "youll", "youd", 'your', 'yours',
                  'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "shes", 'her', 'hers', 'herself',
                  'it', "its", 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who',
                  'whom', 'this', 'that', "thatll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
                  'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
                  'or', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
                  'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
                  'down', 'in', 'out', 'on', 'off', 'over', 'under', 'further', 'then', 'once', 'here',
                  'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                  'some', 'such', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will',
                  'just', 'now', 'd', 'll', 'i', 'e']

    words = [w for w in words if w not in stop_words]
    words = [i for i in words if len(i) > 1]
    return words


all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))

X_test = []
y_test = []

for f in all_files:
    data = tokenization(open(f, "r").read())
    X_test.append([f, data])

nb_model_file = open("nbmodel.txt", "r").read().split("\n")
class_vocab_dictionary = {}

log_prior_nd = float(nb_model_file[0].split(" ")[0])
log_prior_nt = float(nb_model_file[0].split(" ")[1])
log_prior_pd = float(nb_model_file[0].split(" ")[2])
log_prior_pt = float(nb_model_file[0].split(" ")[3])


for item in nb_model_file[1:len(nb_model_file) - 1]:
    item = item.split(" ")
    class_vocab_dictionary[item[0]] = [float(item[1]), float(item[2]), float(item[3]), float(item[4])]


for words in X_test:
    sum_nd = log_prior_nd
    sum_nt = log_prior_nt
    sum_pd = log_prior_pd
    sum_pt = log_prior_pt
    for word in words[1]:
        if word in class_vocab_dictionary:
            sum_nd += class_vocab_dictionary[word][0]
            sum_nt += class_vocab_dictionary[word][1]
            sum_pd += class_vocab_dictionary[word][2]
            sum_pt += class_vocab_dictionary[word][3]

    index = np.argmax([sum_nd, sum_nt, sum_pd, sum_pt])
    if index == 0:
        y_test.append("deceptive negative ")
    elif index == 1:
        y_test.append("truthful negative ")
    elif index == 2:
        y_test.append("deceptive positive ")
    elif index == 3:
        y_test.append("truthful positive ")

nb_output_file = open("nboutput.txt", "w")

for i in range(0, len(y_test)):
    nb_output_file.write(y_test[i] + X_test[i][0] + "\n")


if __name__ == "main":
    model_file = "nbmodel.txt"
    output_file = "nboutput.txt"
    input_path = str(sys.argv[0])