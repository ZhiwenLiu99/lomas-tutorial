import os
import time
import datetime as dt
import numpy as np
import pandas as pd
import concurrent.futures
from scapy.utils import RawPcapReader
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, TCP, UDP

__all__ = ["Preprocessor", "PCAPProcessor"]

class Preprocessor:
    """
    用于对流级别数据进行预处理，包括特征离散化、数据集分割、特征频率统计等。
    """
    def __init__(self, f_path, f_name, f_type, column_names=None, protocol=None, MAIG=None):
        """
        preprocessor submodel

        :param str f_path: 输入的trace数据所在文件夹路径
        :param str f_name: 输入的trace数据文件名（pcap文件类型输入一组 f_name，List[f_name]）
        :param str f_type: 输入的trace数据类型
        :param List[str] f_type: 当输入的trace数据类型为csv或excel格式时，需要指定需要研究的列名
        :param int MAIG: 从pcap数据包中恢复流级别数据所设定的最大超时间隔（minimum allowed interflow gap），单位毫秒
        :raise ValueError: 如果f_type既不等于'flow'也不等于'pcap'
        """
        start_t = time.time()
        print("[Info >> preprocessing] Loading input trace and proprocessing......")
        self.cln_names = column_names
        if f_type=="flow":
            self.trace_input = self.get_flow_level_trace(f_path, f_name)
        elif f_type=="pcap":
            # self.trace_input = self.get_pcap_level_trace(f_path, f_name, protocol, MAIG)
            self.trace_input = self.get_pcap_level_trace_mp(f_path, f_name, protocol, MAIG)
        else:
            raise ValueError("Type of input trace should be one of ['flow', 'pcap']")
        self.ts_to_interval()
        self.select_active_ip(lower_bound=100)
        self.ip_id_dict = self.get_id_of_ip()
        self.ordered_ippair = self.get_ordered_ippair()
        self.cdf_iat = self.get_cdf('iat')
        self.cdf_size = self.get_cdf('size')
        self.discretization_to_flow_type()
        print("[Info >> preprocessing] Finished! Using time: %.2f(s)." % (time.time()-start_t))
        print(f"[Info >> preprocessing] Input trace has {self.trace_input.shape[0]} flows in total.")
    
    def get_flow_level_trace(self, f_path, f_name):
        """
        :param str f_path: 输入的trace数据所在文件夹路径
        :param str f_name: 输入的trace数据文件名
        :raise ValueError: 如果f_name的文件类型不属于 [csv, xlsx, pkl, parquet] 中的任意一种
        """
        _path = os.path.join(f_path, f_name)
        _type = f_name.split('.')[-1]
        if _type=="csv":
            return pd.read_csv(_path,
                               usecols=list(self.cln_names.keys())
                               ).rename(columns=self.cln_names)
        elif _type=="xlsx":
            return pd.read_csv(_path,
                               usecols=list(self.cln_names.values())
                               ).rename(columns=self.cln_names)
        elif _type=="pkl":
            return pd.read_pickle(_path)
        elif _type=="parquet":
            return pd.read_parquet(_path)
        else:
            raise ValueError("File type not defined!")

    def get_pcap_level_trace(self, f_path, f_name, protocol, MAIG):
        """
        :param str f_path: 输入的trace数据所在文件夹路径
        :param str f_name: 输入的trace数据文件名
        :param str protocol: 指定PCAP数据包的协议类型，如TCP、UDP等
        :param int MAIG: 最大超时间隔（minimum allowed interflow gap）
        :return: 从单个pcap文件中解析得到的流级别数据
        :rtype: pd.DataFrame
        """
        dfs = []
        for f in f_name[:2]:
            pcap_processor = PCAPProcessor(os.path.join(f_path, f), protocol, MAIG)
            dfs.append(pcap_processor.process_pcap())
            del pcap_processor
        return pd.concat(dfs)

    def pcap_process_helper(self, params):
        """
        解析单个pcap文件

        :param str params[0]: absolute path for pcap file
        :param str params[1]: protocol
        :param int params[2]: MAIG
        :return: 从单个pcap文件中解析得到的流级别数据
        :rtype: pd.DataFrame
        """
        pcap_processor = PCAPProcessor(params[0], params[1], params[2])
        return pcap_processor.process_pcap()

    def get_pcap_level_trace_mp(self, f_path, f_name, protocol, MAIG):
        """
        功能同 get_pcap_level_trace 函数，多进程处理
        """
        dfs = []
        params = []
        MAX_WORKER = int(0.8 * os.cpu_count())
        for f in f_name:
            params.append([os.path.join(f_path, f), protocol, MAIG])
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKER) as executor:
            results = executor.map(self.pcap_process_helper, params)
            for res in results:
                dfs.append(res)
        return pd.concat(dfs)

    def ts_to_interval(self):
        """
        根据时间戳计算“流到达间隔”
        """
        # def group_diff(x):
        #     return pd.Series(x).diff()
        self.trace_input.sort_values(by=['srcip', 'dstip', 'ts'], inplace=True)
        if 'iat' not in list(self.trace_input.columns):
            self.trace_input['ts'] -= self.trace_input['ts'].min()
            self.trace_input['iat'] = self.trace_input.groupby(['srcip', 'dstip'])['ts'].diff()
            self.trace_input.dropna(how='any', inplace=True)
            # self.trace_input.drop('ts', axis=1, inplace=True)
        else:
            print("[Info >> preprocessing] column ['iat'] already exists!")
    
    def select_active_ip(self, lower_bound):
        """
        只保留样本量大于lower_bound的IP对
        :param int lower_bound: 样本数量的筛选阈值
        """
        self.trace_input['num_of_flow'] = self.trace_input.groupby(['srcip', 'dstip'])['iat'].transform('count')
        self.trace_input = self.trace_input[self.trace_input['num_of_flow']>lower_bound]
        self.trace_input.drop('num_of_flow', axis=1, inplace=True)
        
    def get_id_of_ip(self):
        """
        将IP按字典序排序，并存为dict数据类型。其中dict-key为字典序编号，dict-value为IP
        :return: 排序后的IP及其下标
        :rtype: dict
        """
        ip_list = self.trace_input['srcip'].values
        ip_list = np.append(ip_list, self.trace_input['dstip'].values)
        ip_list = list(np.sort(np.unique(ip_list)))     # 所有 ip 按字符顺序排序
        return dict(zip(ip_list, np.arange(len(ip_list))))
    
    def get_ordered_ippair(self) -> list:
        """
        统计数据集中所有的IP-Pair，并按字典序进行排序。
        :return: 排序后的IP-Pair
        :rtype: list
        """
        self.trace_input['srcid'] = self.trace_input['srcip'].map(self.ip_id_dict)
        self.trace_input['dstid'] = self.trace_input['dstip'].map(self.ip_id_dict)
        self.trace_input['pairid'] = self.trace_input['srcid'].astype('str') + '_' + \
                                     self.trace_input['dstid'].astype('str')
        self.trace_input.drop(['srcid', 'dstid'], axis=1, inplace=True)
        pair_list = list(np.sort(np.unique(self.trace_input['pairid'].values)))
        return pair_list
    
    def get_cdf(self, col_name):
        """
        特征离散化，计算特征的百分位值
        :param string col_name: 特征对应的列名
        :return: 百分位点，及对应的百分位取值
        :rtype: dict
        """
        arr = np.log10(1.0 + np.sort(self.trace_input[col_name].values))
        percentile = self.auto_percentile(arr, num_of_step=40)
        length = len(arr)-1
        cdf_val = []
        for j in percentile:
            if j==0.0:
                cdf_val.append(np.min(arr)*0.9)
            else:
                idx = int(1.0*j*length/10000.0)
                cdf_val.append(arr[idx])
        return dict(zip(percentile, cdf_val))
    
    def auto_percentile(self, arr, num_of_step=40):
        """
        将原始特征进行等频分箱，返回每个分箱区间的下标值
        :param np.array arr: 数组格式存储的原始特征值
        :param int num_of_step: 区间数量
        :return: 每个分箱区间的下标值
        :rtype: np.array
        """
        step = (np.max(arr)-np.min(arr)) / num_of_step
        percentile = [0]
        curr = np.min(arr)
        while curr<np.max(arr):
            tmp = 100*np.mean(arr<=curr)  # 元素curr在数组arr中所处的分位点
            tmp = int(100*round(tmp, 4))
            if tmp not in percentile:
                percentile.append(tmp)
            curr += step
        percentile.append(100*100)
        return np.array(percentile).astype('int')
    
    def discretization_to_flow_type(self):
        """
        将流大小和流间隔两个维度的特征，转换为离散化处理的二元组（flow type），并以字符串形式存储
        """
        arr_size = np.log10(1.0 + self.trace_input['size'].values)
        arr_iat = np.log10(1.0 + self.trace_input['iat'].values)
        df_cdf_size = pd.DataFrame({'percentile': list(self.cdf_size.keys()), 
                                    'cdf': list(self.cdf_size.values())})
        df_cdf_iat = pd.DataFrame({'percentile': list(self.cdf_iat.keys()), 
                                   'cdf': list(self.cdf_iat.values())})
        self.trace_input['size_d'] = self.discretization(df_cdf_size, arr_size)
        self.trace_input['iat_d'] = self.discretization(df_cdf_iat, arr_iat)
        self.trace_input['flow_type'] = '(' + self.trace_input['size_d'].astype('str') + \
                                        ', ' + self.trace_input['iat_d'].astype('str') + ')'
        del arr_size, arr_iat, df_cdf_size, df_cdf_iat
        self.trace_input.drop(['size_d', 'iat_d'], axis=1, inplace=True)

    def discretization(self, cdf, arr):
        """
        根据百分位值对特征进行离散化处理，原始特征值变为对应特征值区间的区间编号
        :param dict cdf: 特征值对应的百分位值
        :param np.array arr: 原始特征值
        :return: 散化处特征值
        :rtype: list
        """
        percent = cdf['percentile'].values
        bounds = cdf['cdf'].values
        length = len(percent)
        arr_res = []
        for idx, x in enumerate(arr):
            # if idx%100000==0:
                # print(dt.datetime.now(), idx, x)
            if x<=np.min(arr):
                arr_res.append(0)
            elif x>bounds[-1]:
                # print(f"warning: input x exceed the max value in CDF.\n-> x:{x}, max of CDF:{bounds[-1]}")
                arr_res.append(10000)
            else:
                for i in range(1,length):
                    if (x>bounds[i-1]) and (x<=bounds[i]):
                        arr_res.append(int(percent[i]))
                        break
        # print(len(arr_res))
        return arr_res


class PCAPProcessor:
    """
    用于将 PCAP 包级别数据聚合成为流级别数据。
    """
    def __init__(self, f_name, protocol, MAIG):
        """
        pcap-file processor submodel
        :param str f_name: 输入的trace数据文件名（pcap文件类型输入一组 f_name，List[f_name]）
        :param str protocol: 指定PCAP数据包的协议类型，如TCP、UDP等
        :param int MAIG: 从pcap数据包中恢复流级别数据所设定的最大超时间隔（minimum allowed interflow gap），单位毫秒
        """
        self.f_name = f_name
        self.protocol = protocol
        self.MAIG = MAIG
    
    def process_pcap(self):
        """
        根据PCAP数据包的协议类型调用不同的预处理函数
        :raise ValueError: 如果协议类型既不是TCP也不是UDP
        """
        if self.protocol=="tcp":
            return self.process_pcap_tcp()
        elif self.protocol=="udp":
            return self.process_pcap_udp()
        else:
            raise ValueError("protocol type undefined!")

    def process_pcap_tcp(self):
        """
        对协议类型为TCP的PCAP数据包文件进行预处理
        :return: 流级别数据
        :rtype: pd.DataFrame
        """
        print(f"process_pcap_tcp: {self.f_name}")
        flow_metadata_dict = {}
        for (pkt_data, pkt_metadata) in RawPcapReader(self.f_name):
            ether_pkt = Ether(pkt_data)
            if ('type' not in ether_pkt.fields) or (ether_pkt.type!=0x8100):
                continue
            try:
                ip_pkt = ether_pkt[IP]  # to obtain the IPv4 header
            except:
                continue
            if ip_pkt.proto != 6:
                continue
            tcp_pkt = ip_pkt[TCP]
            # SYN:表示建立连接；FIN:表示关闭连接；ACK:表示响应；PSH:表示有 DATA数据传输；RST:表示连接重置
            if (str(tcp_pkt.flags)!='PA') and (len(tcp_pkt.payload)<=2):
                continue
            src = str(ip_pkt.src)
            dst = str(ip_pkt.dst)
            idx = src+'-'+dst
            tt = int(1e6*int(pkt_metadata.sec) + 1*int(pkt_metadata.usec))
            pkt_size = int(pkt_metadata.wirelen)

            if idx not in flow_metadata_dict.keys():
                # 0:源地址、1:目的地址、2:开始时间、3:结束时间、4:流大小、5:包数量
                flow_metadata_dict[idx] = [[src, dst, tt, tt, pkt_size, 1]]
            else:
                if (tt-flow_metadata_dict[idx][-1][3]) < self.MAIG:   # 属于同一条流
                    flow_metadata_dict[idx][-1][3] =  tt         # 更新流完成时间
                    flow_metadata_dict[idx][-1][4] += pkt_size   # 累加流size
                    flow_metadata_dict[idx][-1][5] += 1          # 累加流的packet数量
                else:                                            # 新增一条流
                    flow_metadata_dict[idx].append([src, dst, tt, tt, pkt_size, 1])  
        
        arr_2d = []
        for _, val in flow_metadata_dict.items():
            if len(arr_2d)==0:
                arr_2d = val
            else:
                arr_2d.extend(val)
        df = pd.DataFrame(arr_2d, 
                          columns=['srcip', 'dstip', 'ts', 'end_ts', 'size', 'pkt_num'])
        del flow_metadata_dict
        return df[['srcip', 'dstip', 'ts', 'size']]
    
    def process_pcap_udp(self):
        """
        对协议类型为UDP的PCAP数据包文件进行预处理
        :return: 流级别数据
        :rtype: pd.DataFrame
        """
        print(f"process_pcap_tcp: {self.f_name}")
        flow_metadata_dict = {}
        for (pkt_data, pkt_metadata) in RawPcapReader(self.f_name):
            ether_pkt = Ether(pkt_data)
            if ('type' not in ether_pkt.fields) or (ether_pkt.type!=0x0800):
                continue
            try:
                ip_pkt = ether_pkt[IP]  # to obtain the IPv4 header
            except:
                continue
            if ip_pkt.proto != 17:
                continue
            tcp_pkt = ip_pkt[UDP]
            src = str(ip_pkt.src)
            dst = str(ip_pkt.dst)
            idx = src+'-'+dst
            tt = int(1e6*int(pkt_metadata.sec) + 1*int(pkt_metadata.usec))
            pkt_size = int(pkt_metadata.wirelen)

            if idx not in flow_metadata_dict.keys():
                # 0:源地址、1:目的地址、2:开始时间、3:结束时间、4:流大小、5:包数量
                flow_metadata_dict[idx] = [[src, dst, tt, tt, pkt_size, 1]]
            else:
                if (tt-flow_metadata_dict[idx][-1][3]) < self.MAIG:   # 属于同一条流
                    flow_metadata_dict[idx][-1][3] =  tt         # 更新流完成时间
                    flow_metadata_dict[idx][-1][4] += pkt_size   # 累加流size
                    flow_metadata_dict[idx][-1][5] += 1          # 累加流的packet数量
                else:                                            # 新增一条流
                    flow_metadata_dict[idx].append([src, dst, tt, tt, pkt_size, 1])  
        
        arr_2d = []
        for _, val in flow_metadata_dict.items():
            if len(arr_2d)==0:
                arr_2d = val
            else:
                arr_2d.extend(val)
        df = pd.DataFrame(arr_2d, 
                          columns=['srcip', 'dstip', 'ts', 'end_ts', 'size', 'pkt_num'])
        del flow_metadata_dict
        return df[['srcip', 'dstip', 'ts', 'size']]