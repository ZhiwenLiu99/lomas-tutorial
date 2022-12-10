import re
import time
import numpy as np
import pandas as pd

# 设置这个能够看到模型的训练进度
import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

from gensim import corpora
from gensim.models import LdaModel

__all__ = ["LomasModel"]

class LomasModel:
    def __init__(self, ip_id_dict, ordered_ippair, cdf_iat, cdf_size):
        self.ip_id_dict = self.transpose_dict(ip_id_dict)
        self.ordered_ippair = ordered_ippair
        self.cdf_iat = pd.DataFrame({'percentile': list(cdf_iat.keys()), 
                                     'cdf': list(cdf_iat.values())})
        self.cdf_size = pd.DataFrame({'percentile': list(cdf_size.keys()), 
                                      'cdf': list(cdf_size.values())})
        self.arr_flow_type = []
        self.dictionary = None
        self.corpus = None
        self.doc_topics = None
        self.topic_terms = None
        self.trace_syn = None

    def initialize(self, trace_input):
        start_t = time.time()
        for ippair in self.ordered_ippair:
            tmp = trace_input[trace_input['pairid']==ippair]['flow_type'].values
            self.arr_flow_type.append(list(tmp))
        self.dictionary = corpora.Dictionary(self.arr_flow_type)
        self.corpus = [self.dictionary.doc2bow(s) for s in self.arr_flow_type]
        print(f"[Info >> model init] Finished! Using time: {time.time()-start_t : .2f}(s).")
        print(f"[Info >> model init] Num. of unique flow-type: {len(self.dictionary)}.")
        print(f"[Info >> model init] Num. of ip-pair: {len(self.corpus)}.")

    def transpose_dict(self, dic):
        return dict((v, k) for k, v in dic.items())

    def train(self, 
              num_topics=25, 
              chunksize=2000, 
              passes=20, 
              iterations=400, 
              eval_every = None):
        """
        Lomas 模型训练

        :param int num_topics: 隐空间的维度。维度越高，建模精确度越高，但隐空间的可解释性越差
        :param int chunksize:  num of documents will be processed at a time
        :param int passes:     epochs
        :param int iterations: how often we repeat a particular loop over each document
        :param int eval_every: Don't evaluate model perplexity, takes too much time
        """
        # Fix this random seed, or you will get different results
        # And do not change the dictionary-file nor the corpus-file
        start_t = time.time()
        print(f"[Info >> model training] Params: num_topics={num_topics}, chunksize={chunksize}, passes={passes}, iterations={iterations}, eval_every={eval_every}")
        lda = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            chunksize=chunksize,
            alpha='auto',
            eta='auto',
            iterations=iterations,
            num_topics=num_topics,
            passes=passes,
            eval_every=eval_every,
            random_state=np.random.RandomState(2022)
        )
        top_topics = lda.top_topics(self.corpus)
        avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
        print("[Info >> model training] Finished! Using time: %.2f(s)." % (time.time()-start_t))
        print("[Info >> model training] Average topic coherence: %.2f (closer to zero is better)." % avg_topic_coherence)

        # document-topic distribution
        self.doc_topics = np.zeros((len(self.corpus), num_topics))
        for i in range(len(self.corpus)):
            this_topics = lda.get_document_topics(self.corpus[i])
            for j in this_topics:
                self.doc_topics[i,j[0]] = j[1]
        # topic-word distribution
        self.topic_terms = lda.get_topics()

    def generate(self, time_limit, time_unit):
        np.random.seed(2022)
        start_t = time.time()
        syn_trace = []
        for i in range(len(self.ordered_ippair)):
            ts = 0
            pair = self.ordered_ippair[i]
            src_id = int(pair.split('_')[0])
            dst_id = int(pair.split('_')[1])
            src_ip = self.ip_id_dict[src_id]
            dst_ip = self.ip_id_dict[dst_id]
            while ts<time_limit:
                iat, size = self.sampling_value(doc_idx=i)
                ts += iat
                flow = [src_ip, dst_ip, ts, iat, size]
                syn_trace.append(flow)
        self.trace_syn = pd.DataFrame(syn_trace, columns=['srcip', 'dstip', 'ts', 'iat', 'size'])
        print("[Info >> model generating] Finished! Using time: %.2f(s)." % (time.time()-start_t))
        print(f"[Info >> model generating] Time limit is: {time_limit}({time_unit}). Generate {self.trace_syn.shape[0]} flows in total.")

    def sampling_value(self, doc_idx):
        theta = self.doc_topics[doc_idx].copy()
        z = np.argmax(np.random.multinomial(1, theta))    # sample topic index , e.g. select topic
        beta = self.topic_terms[z].copy()                 # sample word from topic
        beta /= (1+1e-7)                                  # to avoid sum(pval)>1 because of decimal round
        maxidx = np.argmax(np.random.multinomial(1, beta))
        new_word = self.dictionary[maxidx]
        percentile = re.split(',|\(|\)', new_word)        # e.g. ['', ' 65', '25', '']
        byt = self.sampling_helper(self.cdf_size, int(percentile[1]))
        iat = self.sampling_helper(self.cdf_iat, int(percentile[2]))
        iat = 10.0**iat - 1.0
        byt = 10.0**byt - 1.0
        return np.ceil(iat), np.ceil(byt)
    
    def sampling_helper(self, cdf, tag):
        percent = list(cdf['percentile'].values)
        bounds = cdf['cdf'].values
        idx = percent.index(tag)
        if idx==0:
            return bounds[idx]
        x0 = percent[idx-1]
        y0 = bounds[idx-1]
        x1 = tag
        y1 = bounds[idx]
        x = np.random.uniform(low=x0, high=x1, size=None)
        return y0 + (y1-y0)/(x1-x0)*(x-x0)
            