import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

__all__ = ["load_json_setting"]

def load_json_setting(f_name: str) -> dict:
    """
    读取json格式配置文件

    :param str f_name: 配置文件的绝对路径
    :return: 相关配置参数
    :rtype: dict
    :raises FileNotFoundError: 如果配置文件的路径不存在
    """
    try:
        with open(f_name, mode='r') as f:
            setting_dict = json.load(f)
        return setting_dict
    except FileNotFoundError:
        print('File does not exist')


def transpose_dict(dic):
    """
    将字典的键和值进行转置

    :param dict dic: dict to be transposed by key-value
    :return: transposed dict
    :rtype: dict
    """
    return dict((v, k) for k, v in dic.items())


def metric_ksd_avg(dfs=[], models=[], col="") -> dict:
    """
    评价指标：KS距离
    作图方式：柱状对比图。横坐标是不同的数据集，纵坐标为评价指标取值，每个数据集对比 [CommonPractice，NetShare，Lomas]

    :param list[pandas.DataFrame] dfs: 包括原始数据集和一组模型产生的合成数据集
    :param list[str] models: 对应参数dfs中每个数据集的标签
    :param str cols: 需要评估的流量属性维度
    """
    assert len(dfs)==len(models), "length of param dfs should be the same as param models!"
    assert models[0]=="real", "the first element in dfs should be the input trace!"
    df_raw = dfs[0][['srcip', 'dstip', col]].copy(deep=True)
    df_raw['pair'] = df_raw['srcip'].astype('str') + '_' + df_raw['dstip'].astype('str')
    pair_raw = np.unique(df_raw['pair'].values)
    metric_dict = {}
    for i in range(1, len(models)):
        df_comp = dfs[i][['srcip', 'dstip', col]].copy(deep=True)
        df_comp['pair'] = df_comp['srcip'].astype('str') + '_' + df_comp['dstip'].astype('str')
        pair_comp = np.unique(df_comp['pair'].values)
        pair_inter = list(set(pair_raw) & set(pair_comp))
        pair_raw_only = list(set(pair_raw) - set(pair_comp))
        pair_comp_only = list(set(pair_comp) - set(pair_raw))
        metric_tmp = []
        for pair in pair_inter:
            ks = ks_2samp(np.log2(1.0+df_raw[df_raw['pair']==pair][col].values), 
                          np.log2(1.0+df_comp[df_comp['pair']==pair][col].values))
            metric_tmp.append(ks[0])
        for _ in pair_raw_only:
            metric_tmp.append(1.0)
        for _ in pair_comp_only:
            metric_tmp.append(1.0)
        metric_dict[models[i]] = np.nanmean(metric_tmp)
    return metric_dict

def metric_ksd(dfs=[], models=[], col="") -> dict:
    """
    评价指标：KS距离
    作图方式：CDF分布对比图。每个数据集一个子图，每个子图对比 [CommonPractice，NetShare，Lomas]

    :param list[pandas.DataFrame] dfs: 包括原始数据集和一组模型产生的合成数据集
    :param list[str] models: 对应参数dfs中每个数据集的标签
    :param str cols: 需要评估的流量属性维度
    """
    assert len(dfs)==len(models), "length of param dfs should be the same as param models!"
    assert models[0]=="real", "the first element in dfs should be the input trace!"
    df_raw = dfs[0][['srcip', 'dstip', col]].copy(deep=True)
    df_raw['pair'] = df_raw['srcip'].astype('str') + '_' + df_raw['dstip'].astype('str')
    pair_raw = np.unique(df_raw['pair'].values)
    metric_dict = {}
    for i in range(1, len(models)):
        df_comp = dfs[i][['srcip', 'dstip', col]].copy(deep=True)
        df_comp['pair'] = df_comp['srcip'].astype('str') + '_' + df_comp['dstip'].astype('str')
        pair_comp = np.unique(df_comp['pair'].values)
        pair_inter = list(set(pair_raw) & set(pair_comp))
        pair_raw_only = list(set(pair_raw) - set(pair_comp))
        pair_comp_only = list(set(pair_comp) - set(pair_raw))
        metric_tmp = []
        for pair in pair_inter:
            ks = ks_2samp(np.log2(1.0+df_raw[df_raw['pair']==pair][col].values), 
                          np.log2(1.0+df_comp[df_comp['pair']==pair][col].values))
            metric_tmp.append(ks[0])
        for _ in pair_raw_only:
            metric_tmp.append(1.0)
        for _ in pair_comp_only:
            metric_tmp.append(1.0)
        metric_dict[models[i]] = get_cdf(metric_tmp)
    return metric_dict

def get_cdf(arr):
    arr = np.sort(arr)
    # percentile = np.arange(0, 102, 2)
    percentile = np.arange(0, 101, 1)
    length = len(arr)-1
    cdf_val = []
    for j in percentile:
        if j==0.0:
            cdf_val.append(0.)
        else:
            idx = int(1.0*j*length/100.0)
            cdf_val.append(arr[idx])
    # return dict(zip(percentile, cdf_val))
    return cdf_val

def metric_topN_pair(dfs=[], models=[], N=5) -> dict:
    """
    评价指标：IP对之间的流数量占比
    作图方式：柱状对比图。每个数据集一个子图，每个子图对比 [CommonPractice，NetShare，Lomas]

    :param list[pandas.DataFrame] dfs: 包括原始数据集和一组模型产生的合成数据集
    :param list[str] models: 对应参数dfs中每个数据集的标签
    :param str cols: 需要评估的流量属性维度
    """
    assert len(dfs)==len(models), "length of param dfs should be the same as param models!"
    assert models[0]=="real", "the first element in dfs should be the input trace!"
    df_raw = dfs[0][['srcip', 'dstip']].copy(deep=True)
    df_raw['pair'] = df_raw['srcip'].astype('str') + '_' + df_raw['dstip'].astype('str')
    df_raw_stats = df_raw.groupby('pair', as_index=False)['srcip'].agg('count').rename(columns={'srcip': 'count'})
    df_raw_stats.sort_values(by='count', ascending=False, inplace=True)
    pair_raw = df_raw_stats['pair'].values[:N]
    metric_dict = {}
    for i in range(0, len(models)):
        df_comp = dfs[i][['srcip', 'dstip']].copy(deep=True)
        df_comp['pair'] = df_comp['srcip'].astype('str') + '_' + df_comp['dstip'].astype('str')
        df_comp_stats = df_comp.groupby('pair', as_index=False)['srcip'].agg('count').rename(columns={'srcip': 'count'})
        metric_tmp = []
        for pair in pair_raw:
            freq = df_comp_stats.loc[df_comp_stats['pair']==pair, 'count'].values
            if len(freq)==0:
                metric_tmp.append(0.)
            else:
                metric_tmp.append(1.0*freq[0]/df_comp_stats['count'].sum())
        metric_dict[models[i]] = metric_tmp
    return metric_dict

def params_comp_3d(f_path="/root/zliu/Lomas/simulation_res_thesis/", 
                   params=[], 
                   trace="EDU1"):
    X, Y, Z = [], [], []
    for i in params[0]:
        for j in params[1]:
            f_name = f"incast_{trace}_Ti_{i}_Td_{j}/fct_star_incastTraffic_{trace}_dcqcn_paper_vwin.txt"
            df = pd.read_csv(f_path+f_name, sep=" ", header=None)
            df.columns = ['srcip', 'dstip', 'srcport', 'dstport', 'size', 'ts', 'fct', 'sdfct']
            df['slowdown'] = 1.0*df['fct'] / df['sdfct']
            df.loc[df['slowdown']<1.0, 'slowdown'] = 1.0
            X.append(i)
            Y.append(j)
            # Z.append(df['slowdown'].quantile(0.95))
            Z.append(df['slowdown'].mean())
    df_res = pd.DataFrame({'X':X, 'Y':Y, 'Z':Z})
    return df_res

def pfc_delay(f_path="/root/zliu/Lomas/simulation_res_thesis/", 
              params=[], 
              trace="EDU1"):
    X, Y, Z = [], [], []
    for i in params[0]:
        for j in params[1]:
            f_name = f"incast_{trace}_Ti_{i}_Td_{j}/fct_star_incastTraffic_{trace}_dcqcn_paper_vwin.txt"
            df_fct = pd.read_csv(f_path+f_name, sep=" ", header=None)
            df_fct.columns = ['srcip', 'dstip', 'srcport', 'dstport', 'size', 'ts', 'fct', 'sdfct']
            f_name = f"incast_{trace}_Ti_{i}_Td_{j}/pfc_star_incastTraffic_{trace}_dcqcn_paper_vwin.txt"
            df_pfc = pd.read_csv(f_path+f_name, sep=" ", header=None)
            df_pfc.columns = ['ts', 'node_id', 'node_type', 'is_index', 'state']
            pfc_delay = 0
            pfc_dict = {}
            for idx, row in df_pfc.iterrows():
                if row.node_id in pfc_dict.keys():
                    if row.state==0:
                        pfc_delay += (row.ts-pfc_dict[row.node_id])
                        pfc_dict.pop(row.node_id)
                    else:
                        pfc_dict[row.node_id] = row.ts
                elif row.state==1:
                    pfc_dict[row.node_id] = row.ts
            X.append(i)
            Y.append(j)
            Z.append(100*pfc_delay/df_fct['fct'].sum())
    df_res = pd.DataFrame({'X':X, 'Y':Y, 'Z':Z})
    return df_res