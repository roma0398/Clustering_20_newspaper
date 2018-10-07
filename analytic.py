from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import k_means_
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
from random import shuffle

categories = [
    "alt.atheism",
    "sci.crypt",
    "talk.politics.guns",
    "misc.forsale",
    "rec.motorcycles",
]
stop_words = stopwords.words("english")


def spread(arg):
    ret = []
    for i in arg:
        if isinstance(i, list):
            ret.extend(i)
        else:
            ret.append(i)
    return ret


def deep_flatten(arr):
    result = []
    result.extend(
        spread(list(map(lambda x: deep_flatten(x) if type(x) == list else x, arr)))
    )
    return result


def tokenize(text):
    min_length = 3
    res = re.sub(r"[/=*^|\\_+1234567890.~,:\'\s]", " ", text)
    words = map(lambda word: word.lower(), word_tokenize(res))
    words = [word for word in words if word not in stop_words]
    tokens = list(map(lambda token: PorterStemmer().stem(token), words))
    p = re.compile("[a-zA-Z]+")
    filtered_tokens = list(
        filter(lambda token: p.match(token) and len(token) >= min_length, tokens)
    )
    filtered_tokens = deep_flatten(filtered_tokens)
    return filtered_tokens


def folder_make(folder, k):
    fol = []
    for i in folder:
        q = "data/" + categories[k - 1] + "/" + i
        fol.append(q)
    return fol


folder1 = os.listdir(path="./data/alt.atheism/")
folder2 = os.listdir(path="./data/sci.crypt/")
folder3 = os.listdir(path="./data/talk.politics.guns/")
folder4 = os.listdir(path="./data/misc.forsale/")
folder5 = os.listdir(path="./data/rec.motorcycles/")
folders_way = [folder1, folder2, folder3, folder4, folder5]


# составление словаря
dictionary = {}
k = 0
v = CountVectorizer(tokenizer=tokenize)
for j in folders_way:
    for i in j:
        q = "data/" + categories[k] + "/" + i
        f = open(q, "r")
        r = f.read().split()
        f.close()
        dictionary.update(v.fit(r).vocabulary_)
    k += 1
dic = []
f = open("res.txt", "w")
for i in dictionary.keys():
    f.write(str(i) + " ")
    dic.append(str(i))
f.close()


def KMeans_cosine_fit(sparse_data, n):
    def euc_dist(X, Y=None, Y_norm_squared=None, squared=False):
        return cosine_similarity(X, Y)

    k_means_.euclidean_distances = euc_dist
    kmeans = k_means_.KMeans(n_clusters=n)
    _ = kmeans.fit(sparse_data)
    return kmeans


def find_prec_recall_F(way, clases, clusters, k):
    s = ""
    kol = [0, 0, 0, 0, 0]
    for i in way:
        q = "data/" + clases[k] + "/" + i
        f = open(q, "r")
        r = f.read()
        r = tokenize(r)
        f.close()
        for z in r:
            s = s + " " + z
        if s in clusters[0]:
            kol[0] += 1
        elif s in clusters[1]:
            kol[1] += 1
        elif s in clusters[2]:
            kol[2] += 1
        elif s in clusters[3]:
            kol[3] += 1
        else:
            kol[4] += 1
        s = ""
    max1 = 0
    num1 = 0
    for i in range(5):
        if kol[i] > max1:
            max1 = kol[i]
            num1 = i
    fp = len(clusters[num1]) - max1
    rc = max1 / len(way)
    pr = max1 / (max1 + fp)
    fc = 2 * (pr * rc) / (pr + rc)
    cl = len(clusters[num1])
    return cl, rc, pr, fc


def finaly_measure(way, clases, clusters):
    k = 0
    D = 0
    p = 0
    for i in way:
        cl, rc, pr, fc = find_prec_recall_F(i, clases, clusters, k)
        k += 1
        D += len(i)
        p += cl * fc
    return p / D


data = []
k = 0
s = ""
for j in folders_way:
    for i in j:
        q = "data/" + categories[k] + "/" + i
        f = open(q, "r")
        r = f.read()
        r = tokenize(r)
        f.close()
        for z in r:
            s = s + " " + z
        data.append(s)
        s = ""
    k += 1
shuffle(data)
tfidf_v = TfidfVectorizer(min_df=0.1, max_df=0.9)
X = tfidf_v.fit_transform(data)
n_clusters = 5
k_means = KMeans_cosine_fit(X, n_clusters)
cluster = {}
for key, label in zip(data, k_means.labels_):
    cluster[label] = cluster.get(label, []) + [key]

with open("clusters.txt", "w") as out:
    for key, val in cluster.items():
        out.write("{}:{}\n".format(key, val))

FM = finaly_measure(folders_way, categories, cluster)
print(FM)
