import spacy
from tqdm import tqdm

import numpy as np
import seaborn as sns
import scipy
import nltk
import pandas as pd
import json

from tqdm import tqdm as tqdm
from tqdm import trange
import tqdm.notebook as tq


from sklearn.cluster import KMeans
from scipy.stats import poisson, gamma, dirichlet
from scipy.special import digamma
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
from itertools import combinations

def check_name(nlp, name):
    doc = nlp(name)[0]
    return doc.ent_type_


# if __name__=="__main__":
#     # print(check_name("walesa")=="PERSON")
#     with open("datasets/ap"+"/"+"vocab.txt",'r') as f:
#         lines = f.read().splitlines()

#     nlp = spacy.load("en_core_web_sm")
#     with open("fucking_person.txt","w") as f:
#         for word in tqdm(lines):
#             if check_name(nlp, word)=="PERSON":
#                 f.writelines(word+"\n")



if __name__=="__main__":
    aa = np.random.randint(10, size=(20,100))
    kmeans = KMeans(n_clusters=3, random_state=0).fit(aa)
    print(kmeans.labels_.astype(int))