# use this file to learn naive-bayes classifier 
# Expected: generate nbmodel.txt

import sys
import glob
import os
import re
import math
import collections
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
                  'down', 'in', 'out', 'on', 'off', 'over', 'under', 'then', 'once', 'here',
                  'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                  'some', 'such', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will',
                  'just', 'now', 'll']

    words = [w for w in words if w not in stop_words]
    words = [i for i in words if len(i) > 1]
    return collections.Counter(words)


all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))

X_train = {'nd': [], 'nt': [], 'pd': [], 'pt': []}

total_number_of_train_documents = 0
total_number_of_test_documents = 0
number_tokens_nd = 0
number_tokens_nt = 0
number_tokens_pd = 0
number_tokens_pt = 0
total_token_in_vocabulary = 0
log_prior_nd = 0
log_prior_nt = 0
log_prior_pd = 0
log_prior_pt = 0


for f in all_files:
    class1, class2, fold, fname = f.split('/')[-4:]
    if "negative" in str(class1) and "deceptive" in str(class2):
        data = tokenization(open(f, "r").read())
        X_train["nd"].append(data)
    elif "negative" in str(class1) and "truthful" in str(class2):
        data = tokenization(open(f, "r").read())
        X_train["nt"].append(data)
    elif "positive" in str(class1) and "deceptive" in str(class2):
        data = tokenization(open(f, "r").read())
        X_train["pd"].append(data)
    elif "positive" in str(class1) and "truthful" in str(class2):
        data = tokenization(open(f, "r").read())
        X_train["pt"].append(data)


documents_class_nd = collections.Counter([])
documents_class_nt = collections.Counter([])
documents_class_pd = collections.Counter([])
documents_class_pt = collections.Counter([])
for item in X_train.items():
    if item[0] == "nd":
        for doc in item[1]:
            documents_class_nd = documents_class_nd + doc
    elif item[0] == "nt":
        for doc in item[1]:
            documents_class_nt = documents_class_nt + doc
    elif item[0] == "pd":
        for doc in item[1]:
            documents_class_pd = documents_class_pd + doc
    elif item[0] == "pt":
        for doc in item[1]:
            documents_class_pt = documents_class_pt + doc

vocabulary = documents_class_nd + documents_class_nt + documents_class_pd + documents_class_pt
vocabulary = collections.Counter(dict(filter(lambda x: x[1] > 1, vocabulary.items())))

tmpDict = documents_class_nd.copy()
for key, value in tmpDict.items():
    if key not in vocabulary:
        del documents_class_nd[key]

tmpDict = documents_class_nt.copy()
for key, value in tmpDict.items():
    if key not in vocabulary:
        del documents_class_nt[key]

tmpDict = documents_class_pd.copy()
for key, value in tmpDict.items():
    if key not in vocabulary:
        del documents_class_pd[key]

tmpDict = documents_class_pt.copy()
for key, value in tmpDict.items():
    if key not in vocabulary:
        del documents_class_pt[key]


number_tokens_nd = sum(documents_class_nd.values())
number_tokens_nt = sum(documents_class_nt.values())
number_tokens_pd = sum(documents_class_pd.values())
number_tokens_pt = sum(documents_class_pt.values())
            
total_token_in_vocabulary = number_tokens_nd + number_tokens_nt + number_tokens_pd + number_tokens_pt

log_prior_nd = math.log(float(number_tokens_nd)/float(total_token_in_vocabulary))
log_prior_nt = math.log(float(number_tokens_nt)/float(total_token_in_vocabulary))
log_prior_pd = math.log(float(number_tokens_pd)/float(total_token_in_vocabulary))
log_prior_pt = math.log(float(number_tokens_pt)/float(total_token_in_vocabulary))
log_prior = [log_prior_nd, log_prior_nt, log_prior_pd, log_prior_pt]


log_likelihood = []

for word in vocabulary:
    nd = math.log(float((documents_class_nd[word] + 1)) / float(number_tokens_nd))
    nt = math.log(float((documents_class_nt[word] + 1)) / float(number_tokens_nt))
    pd = math.log(float((documents_class_pd[word] + 1)) / float(number_tokens_pd))
    pt = math.log(float((documents_class_pt[word] + 1)) / float(number_tokens_pt))
    log_likelihood.append([word, nd, nt, pd, pt])


nb_model_file = open("nbmodel.txt", "w")
for log_prior_c in log_prior:
    nb_model_file.write("%s " % log_prior_c)
nb_model_file.write("\n")
for log_likelihood_w_c in log_likelihood:
    for w_c in log_likelihood_w_c:
        nb_model_file.write("%s " % w_c)
    nb_model_file.write("\n")


if __name__ == "main":
    model_file = "nbmodel.txt"
    input_path = str(sys.argv[0])