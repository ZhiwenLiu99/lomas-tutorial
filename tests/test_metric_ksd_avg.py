import os
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent
from lomas import utils
from lomas.preprocessor import Preprocessor
from lomas.generator import GeneratorLomas, GeneratorCommon

def generate_and_eval(param):
    trace_name, target_col = param[0], param[1]
    ########## Preprocessor ##########
    data = Preprocessor(f_path=config[trace_name]['path'], 
                        f_name=config[trace_name]['filenames'], 
                        f_type=config[trace_name]['type'],
                        column_names=config[trace_name]['cols'])
    
    ########## Generator-Lomas ##########
    model_lomas = GeneratorLomas(ip_id_dict=data.ip_id_dict, 
                                ordered_ippair=data.ordered_ippair, 
                                cdf_iat=data.cdf_iat, 
                                cdf_size=data.cdf_size)
    model_lomas.initialize(data.trace_input)
    model_lomas.train(num_topics=25, 
                    chunksize=2000, 
                    passes=20, 
                    iterations=400)
    model_lomas.generate(time_limit=data.trace_input['ts'].max(), 
                        time_unit=config[trace_name]['ts_unit'])
    
    ########## Generator-Common ##########
    model_common = GeneratorCommon(ip_id_dict=data.ip_id_dict, 
                                cdf_size=data.cdf_size)
    model_common.initialize(data.trace_input)
    model_common.generate(time_limit=data.trace_input['ts'].max(), 
                        time_unit=config[trace_name]['ts_unit'])
    
    ########## evaluation-iat ##########
    metric_dict = utils.metric_ksd_avg(dfs=[data.trace_input[['srcip', 'dstip', 'iat', 'size']].copy(deep=True), \
                                            model_lomas.trace_syn[['srcip', 'dstip', 'iat', 'size']].copy(deep=True), \
                                            model_common.trace_syn[['srcip', 'dstip', 'iat', 'size']].copy(deep=True)],
                                        models=['real', 'lomas', 'common'],
                                        col=target_col)
    df_tmp = pd.DataFrame({"model": list(metric_dict.keys()),
                           "metric": list(metric_dict.values())})
    df_tmp['trace'] = trace_name
    return df_tmp


if __name__ == "__main__":
    ########## configuration ##########
    setting_path = os.path.join(os.getcwd(), "./config/data_path_v3.json")
    config = utils.load_json_setting(setting_path)
    trace_names = ['UN1_flow', 'UN2_flow', 'FB_Cluster_A', 'FB_Cluster_B', 'FB_Cluster_C']
    target_col = 'size'

    params = []
    for trace_name in trace_names:
        params.append([trace_name, target_col])
    
    df_eval_li = []
    MAX_WORKER = int(0.8 * os.cpu_count())
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKER) as executor:
        results = executor.map(generate_and_eval, params)
        for res in results:
            df_eval_li.append(res)
        
    df_eval = pd.concat(df_eval_li)
    ax = sns.barplot(x="trace", 
                     y="metric", 
                     hue="model", 
                     data=df_eval)
    pdb.set_trace()