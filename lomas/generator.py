import re
import sys
import time
import random
import heapq
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from lomas import utils
import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

__all__ = ["GeneratorLomas", "GeneratorCommon"]

class GeneratorLomas(object):
    def __init__(self, ip_id_dict, ordered_ippair, cdf_iat, cdf_size):
        """
        基于历史流量数据进行模型训练、基于训练好的模型产生新的合成流量数据

        :param dict ip_id_dict: key=index of IP, value=(anonymized)IP addr
        :param list ordered_ippair: ordered IP pair (IP is represented by its index)
        :param dict cdf_iat: key=percentile, value=values of interarrival time CDF at some percentile
        :param dict cdf_size: key=percentile, value=values of flow size CDF at some percentile
        """
        self.ip_id_dict = utils.transpose_dict(ip_id_dict)
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
        """
        获取每个源目的对之间的流数据，并以二维数组的数据类型储存

        :param pandas.DataFrame trace_input: can be accessed using lomas.Preprocessor.trace_input
        :return: self.arr_flow_type, self.dictionary, self.corpus
        :rtype: 2D list, 1D list, 2D list
        """
        start_t = time.time()
        for ippair in self.ordered_ippair:
            tmp = trace_input[trace_input['pairid']==ippair]['flow_type'].values
            self.arr_flow_type.append(list(tmp))
        self.dictionary = corpora.Dictionary(self.arr_flow_type)
        self.corpus = [self.dictionary.doc2bow(s) for s in self.arr_flow_type]
        print(f"[Info >> model-lomas init] Finished! Using time: {time.time()-start_t : .2f}(s).")
        print(f"[Info >> model-lomas init] Num. of unique flow-type: {len(self.dictionary)}.")
        print(f"[Info >> model-lomas init] Num. of ip-pair: {len(self.corpus)}.")

    def train(self, 
              num_topics=25, 
              chunksize=2000, 
              passes=20, 
              iterations=400, 
              eval_every = None):
        """
        模型训练

        :param int num_topics: dimension of latent space
        :param int chunksize:  num of documents will be processed at a time
        :param int passes:     num of epochs
        :param int iterations: how often we repeat a particular loop over each document
        :param int eval_every: don't evaluate model perplexity, takes too much time
        :return: self.doc_topics (document-topic distribution), self.topic_terms (topic-word distribution)
        :rtype: 2D np.array, 2D np.array
        """
        # Fix this random seed, or you will get different results
        # And do not change the dictionary-file nor the corpus-file
        start_t = time.time()
        print(f"[Info >> model-lomas training] Params: num_topics={num_topics}, chunksize={chunksize}, passes={passes}, iterations={iterations}, eval_every={eval_every}")
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
        print("[Info >> model-lomas training] Finished! Using time: %.2f(s)." % (time.time()-start_t))
        print("[Info >> model-lomas training] Average topic coherence: %.2f (closer to zero is better)." % avg_topic_coherence)

        # document-topic distribution
        self.doc_topics = np.zeros((len(self.corpus), num_topics))
        for i in range(len(self.corpus)):
            this_topics = lda.get_document_topics(self.corpus[i])
            for j in this_topics:
                self.doc_topics[i,j[0]] = j[1]
        # topic-word distribution
        self.topic_terms = lda.get_topics()

    def generate(self, time_limit, time_unit):
        """
        生成新的合成流量数据

        :param int time_limit: control how many flows will be generated (s.t. [num. of flow]*[avg. iat] <= [time_limit])
        :param int time_unit: time uint of time_limit
        :return: self.trace_syn (synthetic trace)
        :rtype: pandas.DataFrame
        """
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
        print("[Info >> model-lomas generating] Finished! Using time: %.2f(s)." % (time.time()-start_t))
        print(f"[Info >> model-lomas generating] Time limit is: {time_limit}({time_unit}). Generate {self.trace_syn.shape[0]} flows in total.")

    def sampling_value(self, doc_idx):
        """
        从隐空间概率分布矩阵中采样，以概率分布产生流大小和流间隔的联合取值

        :param int doc_idx: index according to ordered IP pair
        :return: interarrival time, flow size
        :rtype: int, int
        """
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
        """
        辅助函数，将离散化的流大小、流间隔标签映射回实数值

        :param pandas.DataFrame cdf: pandas.DataFrame(['percentile', 'cdf'])
        :param int tag: discretized size or iat tag
        :return: continuous value of size or iat
        :rtype: float
        """
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


class GeneratorCommon(object):
    def __init__(self, ip_id_dict, cdf_size):
        self.avg_iat = None
        self.size_sampler = CustomRand()
        self.nhost = len(ip_id_dict)
        self.ip_id_dict = utils.transpose_dict(ip_id_dict)
        self.cdf_size = pd.DataFrame({'percentile': list(cdf_size.keys()), 
                                      'cdf': list(cdf_size.values())})
        self.trace_syn = None
    
    def initialize(self, trace_input):
        self.avg_iat = trace_input['iat'].mean()
        if not self.size_sampler.set_cdf(self.cdf_size[['cdf','percentile']].values):
            print("Error: Not valid cdf")
            sys.exit(0)
    
    def poisson(self, lam):
        return int(np.random.poisson(lam, 1)[0])
    
    def generate(self, time_limit, time_unit):
        np.random.seed(2022)
        start_t = time.time()
        host_list = [(self.poisson(self.avg_iat), i) for i in range(self.nhost)]
        heapq.heapify(host_list)
        syn_trace = []
        while len(host_list) > 0:
            ts, src = host_list[0]
            iat = self.poisson(self.avg_iat)
            dst = random.randint(0, self.nhost-1)
            while (dst == src):
                dst = random.randint(0, self.nhost-1)
            if (ts + iat > time_limit):
                heapq.heappop(host_list)
            else:
                size = int(self.size_sampler.rand())
                if size <= 0:
                    size = 1
                heapq.heapreplace(host_list, (ts+iat, src))
                flow = [self.ip_id_dict[src], self.ip_id_dict[dst], ts+iat, iat, size]
                syn_trace.append(flow)
        self.trace_syn = pd.DataFrame(syn_trace, columns=['srcip', 'dstip', 'ts', 'iat', 'size'])
        print("[Info >> model-common generating] Finished! Using time: %.2f(s)." % (time.time()-start_t))
        print(f"[Info >> model-common generating] Time limit is: {time_limit}({time_unit}). Generate {self.trace_syn.shape[0]} flows in total.")


class CustomRand(object):
    def __init__(self):
        self.cdf = None
    
    def test_cdf(self, cdf):
        if int(cdf[0][1]) != 0:
            return False
        if int(cdf[-1][1]) != 10000:
            return False
        for i in range(1, len(cdf)):
            if cdf[i][1]<=cdf[i-1][1] or cdf[i][0]<cdf[i-1][0]:
                return False
        return True
    
    def set_cdf(self, cdf):
        if not self.test_cdf(cdf):
            return False
        self.cdf = cdf
        return True
    
    def get_avg(self):
        s = 0
        last_x, last_y = self.cdf[0]
        for c in self.cdf[1:]:
            x, y = c
            s += (x + last_x)/2.0 * (y - last_y)
            last_x = x
            last_y = y
        return s/100
    
    def rand(self):
        r = random.random() * 10000
        val = 10.0**self.get_value_from_percentile(r)-1.0
        return int(val)
    
    def get_value_from_percentile(self, y):
        for i in range(1, len(self.cdf)):
            if y <= self.cdf[i][1]:
                x0,y0 = self.cdf[i-1]
                x1,y1 = self.cdf[i]
                return x0 + (x1-x0)/(y1-y0)*(y-y0)
    
