import os
import pickle
import json
import numpy as np

# import nltk
# from nltk.corpus import stopwords
import spacy

from tqdm import tqdm

from scipy.stats import gamma, poisson
from scipy.special import logsumexp
from sklearn.cluster import KMeans

# set parameter
def make_config():
    config = {}
    ##### for option #####
    config["START_IDX"] = 50
    config["N_WORDS"] = 2000
    config["DEFAULT_DATA_PATH"] = "datasets/ap"
    config["FORCE_LOAD_FROM_RAW_DATA"] = False
    config["ALPHA"] = 5e+1
    config["BETA"] = 1e-2
    config["N_CLS"] = 5
    config["N_ITER"] = 100
    config["N_TOPK"] = 10
    #############################
    return config


class ApData:
    def __init__(self, start_idx, n_words, data_path):

        # # for stop word download
        # nltk.download('stopwords')
        # english_stopwords = stopwords.words('english')

        # load custom stopwords (nltk + person name + etc..)
        english_stopwords = []
        with open("stopwords/stopwords.txt", "rt", encoding='UTF8') as f:
            lines = f.readlines()
            for line in lines:
                english_stopwords.append(line.replace("\n",""))

        # read vocabulary
        with open(data_path+"/"+"vocab.txt",'r') as f:
            raw_vocabs = f.read().splitlines()

        # check frequency
        freq_raw_dict = {}
        with open(data_path+"/"+"ap.dat",'r') as f:
            lines = f.readlines()
            for line in lines:
                freqs = line.split(' ')[1:]
                for i in range(len(freqs)):
                    key, val = int(freqs[i].split(':')[0]),int(freqs[i].split(':')[1])
                    if raw_vocabs[key] not in english_stopwords:
                        if key not in freq_raw_dict.keys():
                            freq_raw_dict[key] = val
                        else:
                            freq_raw_dict[key] += val
        freq_raw_dict = sorted(freq_raw_dict.items(), key=lambda x:x[1], reverse=True)
        
        # check for validity
        if freq_raw_dict[start_idx + n_words][1] == 0:
            raise Exception("Invalid start idx and n words!!!")
        
        valid_idx = [i for i,_ in freq_raw_dict[start_idx:start_idx+n_words]]
        valid_idx_dict = {idx:i for i,idx in enumerate(valid_idx)}

        # make matrix of data : row idx -> article num, col idx -> valid word
        preprocessed_data = []
        with open(data_path+"/"+"ap.dat",'r') as f:
            lines = f.readlines()
            for line in lines:
                freq_dicts = line.split(' ')[1:]
                row_data = np.zeros(shape=n_words)
                for _, freq_dict in enumerate(freq_dicts):
                    key, val = int(freq_dict.split(':')[0]),int(freq_dict.split(':')[1])
                    if key in valid_idx:
                        row_data[valid_idx_dict[key]] = val
                preprocessed_data.append(row_data)
        preprocessed_data = np.vstack(preprocessed_data)

        # for save
        self.raw_vocabs = raw_vocabs
        self.preprocessed_data = preprocessed_data
        self.valid_idx = valid_idx
        self.n_words = n_words
        self.n_docs = preprocessed_data.shape[0]



class GibbsSampler:
    def __init__(self, data, X, n_cls, n_topK=10):
        self.data = data
        self.X = X
        self.n_cls = n_cls
        self.n_doc = X.shape[0]
        self.n_word = X.shape[1]
        self.K = n_topK

    def init_label(self, alpha=1, beta=1, opt="uniform_random"):
        # init alpha and beta using hyperparameter
        self.Alpha = np.ones(shape=(self.n_cls, self.n_word)) * alpha
        self.Beta = np.ones(shape=(self.n_cls, self.n_word)) * beta

        # method for initialize label
        if opt=="uniform_random":
            self.Z = np.random.randint(self.n_cls, size=(self.n_doc, 1))

        elif opt=="kmeans":
            kmeans = KMeans(n_clusters=self.n_cls, random_state=0).fit(self.X)
            self.Z = kmeans.labels_.astype(int)

        else:
            raise Exception(str("There is no initialize option named \"") + opt + "\"!")
        
        # pre-calculating increment of alpha and beta for alpha hat and beta hat
        self.alpha_hat = np.zeros_like(self.Alpha)
        self.beta_hat = np.zeros_like(self.Beta)
        for doc, label in zip(self.X, self.Z):
            self.alpha_hat[int(label)] += doc
            self.beta_hat[int(label)] += np.where(doc != 0, 1, 0)

        
    def sampling(self, n_iter):
        for n in tqdm(range(n_iter)):
            for doc_idx in tqdm(range(self.n_doc)):
                doc = self.X[doc_idx]
                label_prev = self.Z[doc_idx]
                alpha_hat = self.alpha_hat + self.Alpha
                beta_hat = self.beta_hat + self.Beta

                lambdas = np.random.gamma(shape=alpha_hat, scale=np.divide(1, beta_hat))
                self.lambdas = lambdas
                prior_log = gamma.logpdf(lambdas, self.Alpha, self.Beta)
                likelihood_log = poisson.logpmf(doc, lambdas)
                posterior_log = likelihood_log + prior_log   # shape = (n_cls, n_word)
                posterior_log = np.sum(posterior_log, axis=1)   # shape = (n_word,)

                ## original case --> nan
                # posterior_log -= np.min(posterior_log)
                # posterior = np.exp(posterior_log)
                # w = posterior / np.sum(posterior)

                # using log scale calculation because of precision
                aa = logsumexp(posterior_log)
                posterior_log -= aa
                w = np.exp(posterior_log)
                w = w / np.sum(w)

                z_new = np.random.multinomial(1, w)
                label_new = np.where(z_new==1)[0]
                self.Z[doc_idx] = label_new

                self.alpha_hat[label_prev] -= doc
                self.alpha_hat[label_new] += doc
                self.beta_hat[label_prev] -= np.where(doc != 0, 1, 0)
                self.beta_hat[label_new] += np.where(doc != 0, 1, 0)

            topK_idx = self.get_topK_words()
            res = {}
            for i in range(self.n_cls):
                idx = topK_idx[i]
                res[i] = []
                for j in idx:
                    res[i].append(self.data.raw_vocabs[self.data.valid_idx[j]])

            write_json(res, "top_word_iter"+str(n)+".json")

    
    def get_topK_words(self):
        topK_idx = []
        for i in range(self.n_cls):
            sorted_idx = np.argsort(self.lambdas[i])[::-1]
            topK_idx.append(sorted_idx[:self.K])

        return topK_idx



def check_save_and_load(config):
    start_idx = config["START_IDX"]
    n_words = config["N_WORDS"]
    force_load_raw_data = config["FORCE_LOAD_FROM_RAW_DATA"]
    data_path = config["DEFAULT_DATA_PATH"]

    file_name = "preprocessed_data/data_" + str(start_idx) + "_" + str(n_words) + ".pickle"
    if os.path.isfile(file_name) and not force_load_raw_data:
        print("##### load from save file( "+ file_name +" )!")
        with open(file_name,'rb') as fr:
            data = pickle.load(fr)
    else:
        print("##### load from raw data!")
        data = ApData(start_idx, n_words, data_path)
        with open(file_name, 'wb') as fw:
            pickle.dump(data, fw)
    print("##### load finish!")
    return data


def write_json(dictionary, json_file="result.json"):
    file_name = "result/" + json_file
    with open(file_name, 'w') as f:
        json.dump(dictionary, f)




if __name__=="__main__":
    config = make_config()
    data = check_save_and_load(config)

    res = {}
    res["valid_word"] = []
    for idx in data.valid_idx:
        res["valid_word"].append(data.raw_vocabs[idx])
    write_json(res, "valid_word.json")
    
    sampler = GibbsSampler(data, data.preprocessed_data, config["N_CLS"], config["N_TOPK"])
    sampler.init_label(config["ALPHA"], config["BETA"], "kmeans")
    sampler.sampling(config["N_ITER"])