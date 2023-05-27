import os
import pickle
import json
import numpy as np

# import nltk
# from nltk.corpus import stopwords
import spacy

from tqdm import tqdm

from scipy.stats import gamma, poisson
from scipy.special import logsumexp, digamma
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

from itertools import combinations

# set parameter
def make_config():
    config = {}
    ##### for option #####
    config["START_IDX"] = 0
    config["N_WORDS"] = 0
    config["DEFAULT_DATA_PATH"] = "datasets/ap"
    config["FORCE_LOAD_FROM_RAW_DATA"] = False

    # # for gibbs sampling
    # config["ALPHA"] = 5e+1
    # config["BETA"] = 1e-2
    # config["GAMMA"] = 1e+0

    # for variational inference
    config["ALPHA"] = 1
    config["BETA"] = 1
    config["GAMMA"] = 1

    config["N_CLS"] = 10
    config["N_ITER"] = 100
    config["N_TOPK"] = 25
    #############################
    return config


class ApData:
    def __init__(self, start_idx, n_words, data_path):

        # # for stop word download
        # nltk.download('stopwords')
        # english_stopwords = stopwords.words('english')

        # load custom stopwords (nltk + person name + etc..)
        english_stopwords = []

        # # not using stopwords in final project
        # with open("stopwords/stopwords.txt", "rt", encoding='UTF8') as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         english_stopwords.append(line.replace("\n",""))

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
        
        # # get word based on frequency ranking
        # valid_idx = [i for i,_ in freq_raw_dict[start_idx:start_idx+n_words]]

        valid_idx = []
        for idx,n in freq_raw_dict:
            if n <= 1000 and n >= 10:
                valid_idx.append(idx)
        # print(len(valid_idx))
        n_words = len(valid_idx)
        

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
        self.K = int(n_topK)

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
    


class Vi:
    def __init__(self, data, X, n_cls, n_topK=10):
        self.data = data
        self.X = X
        self.n_cls = n_cls
        self.n_doc = X.shape[0]
        self.n_word = X.shape[1]
        self.K = n_topK
        self.elbos = []
        self.coherence = []

    def init_label(self, alpha=1, beta=1, gamma=1, opt="uniform_random"):
        # init alpha and beta using hyperparameter
        self.Alpha = np.ones(shape=(self.n_cls, self.n_word)) * alpha
        self.Beta = np.ones(shape=(self.n_cls, self.n_word)) * beta
        self.Gamma = np.ones(shape=(self.n_cls, 1)) * gamma
        self.Pi = np.ones(shape=(self.n_cls, 1)) / self.n_cls
        self.lambdas = np.zeros(shape=(self.n_cls, self.n_word))
        self.Alpha_hat = self.Alpha
        self.Beta_hat = self.Beta
        self.Gamma_hat = self.Gamma
        self.rho_nk = np.zeros(shape=(self.n_doc, self.n_cls))
        self.r_nk = None

        # method for initialize label
        if opt=="uniform_random":
            self.Z = np.random.randint(self.n_cls, size=(self.n_doc, 1))

        elif opt=="kmeans":
            kmeans = KMeans(n_clusters=self.n_cls, random_state=0).fit(self.X)
            self.Z = kmeans.labels_.astype(int)

        else:
            raise Exception(str("There is no initialize option named \"") + opt + "\"!")
        
    def update_z(self):
        mean_lambda_lambda = self.Alpha_hat / self.Beta_hat
        mean_lambda_log_lambda = digamma(self.Alpha_hat) - np.log(self.Beta_hat)
        mean_pi_pi = digamma(self.Gamma_hat) - digamma(self.Gamma_hat.sum())

        for idx in range(self.n_doc):
            self.rho_nk[idx] = (self.X[idx] * mean_lambda_log_lambda - mean_lambda_lambda).sum(axis=1).T + mean_pi_pi.T.squeeze()

        self.r_nk = self.rho_nk / self.rho_nk.sum(axis=1).reshape(self.rho_nk.shape[0],1)

        for idx in range(self.n_doc):
            z_new = np.random.multinomial(1, self.r_nk[idx])
            label_new = np.where(z_new == 1)[0]
            self.Z[idx] = label_new

    def update_pi(self):
        self.Gamma_hat = self.Gamma + self.r_nk.sum(axis=0).reshape(self.n_cls, 1)

    def update_lambdas(self):
        self.Alpha_hat = self.Alpha
        self.Beta_hat = self.Beta

        for idx, (doc, label) in enumerate(zip(self.X, self.Z)):
            self.Alpha_hat[int(label)] += doc * self.r_nk[idx][int(label)]
            self.Beta_hat[int(label)] += np.where(doc != 0, 1, 0) * self.r_nk[idx][int(label)]

        self.lambdas = np.random.gamma(shape=self.Alpha_hat, scale=np.divide(1, self.Beta_hat))

    def calc_elbo(self):
        mean_lambda_log_lambda = digamma(self.Alpha_hat) - np.log(self.Beta_hat)
        mean_pi_pi = digamma(self.Gamma_hat) - digamma(self.Gamma_hat.sum())

        elbo = (self.r_nk*((np.dot(self.X,(mean_lambda_log_lambda).T) + mean_pi_pi.T.squeeze()) - np.log(self.r_nk))).sum(axis=1)
        elbo += (mean_pi_pi.sum() * (self.Gamma_hat.sum() - 1) - np.multiply(mean_pi_pi, digamma(self.Gamma_hat) - 1).sum())

        ret_val = elbo.mean()
        self.elbos.append(ret_val)

        return ret_val
    
    def get_topK_words(self):
        topK_idx = []
        for i in range(self.n_cls):
            sorted_idx = np.argsort(self.lambdas[i])[::-1]
            # sorted_idx = list(sorted_idx)
            topK_idx.append(sorted_idx[:self.K])

        return topK_idx
    
    def get_coherence(self, topK_idx):
        coherence = []
        for c in range(self.n_cls):
            combs = list(combinations(topK_idx[c][:self.K], 2))
            for comb in combs:
                D_i_j = (self.X[:,comb].prod(axis=-1)!=0).sum()
                D_i = (self.X[:,comb[0]]!=0).sum()
                coherence.append(np.log(np.divide(D_i_j+1,D_i)))
        
        coherence = np.array(coherence)
        ret_val = coherence.mean()
        self.coherence.append(ret_val)

        return ret_val
    
    def fit(self, n_iter):
        for n in tqdm(range(n_iter)):
            res = {}

            self.update_z()
            self.update_pi()
            self.update_lambdas()
            elbo = self.calc_elbo()
            res["elbo"] = elbo

            topK_idx = self.get_topK_words()
            # coh = self.get_coherence(topK_idx)
            # res["coh"] = coh
            
            for i in range(self.n_cls):
                idx = topK_idx[i]
                res[i] = []
                for j in idx:
                    res[i].append(self.data.raw_vocabs[self.data.valid_idx[j]])

            write_json(res, "top_word_iter"+str(n)+".json")






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


def do_gibbs_sampling():
    config = make_config()
    data = check_save_and_load(config)

    # res = {}
    # res["valid_word"] = []
    # for idx in data.valid_idx:
    #     res["valid_word"].append(data.raw_vocabs[idx])
    # write_json(res, "valid_word.json")
    
    sampler = GibbsSampler(data, data.preprocessed_data, config["N_CLS"], config["N_TOPK"])
    sampler.init_label(config["ALPHA"], config["BETA"], "kmeans")
    sampler.sampling(config["N_ITER"])


def do_varational_inference():
    config = make_config()
    data = check_save_and_load(config)

    vi = Vi(data, data.preprocessed_data, config["N_CLS"], config["N_TOPK"])
    # vi.init_label(config["ALPHA"], config["BETA"], config["GAMMA"])
    vi.init_label(config["ALPHA"], config["BETA"], config["GAMMA"], "kmeans")
    vi.fit(config["N_ITER"])

    plt.figure()
    plt.plot(vi.elbos)

    plt.figure()
    plt.plot(vi.coherence)

    plt.show()



if __name__=="__main__":
    # do_gibbs_sampling()
    do_varational_inference()