import os
import time
import numpy as np
import pandas as pd
import concurrent.futures
from scapy.utils import RawPcapReader
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, TCP, UDP

__all__ = ["Preprocessor"]

class Preprocessor:
    def __init__(self, f_path, f_name, f_type, column_names=None, protocol=None, MAIG=None):
        """
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
        :raise ValueError: 如果f_name的文件类型不属于 [csv, xlsx, pkl, parquet]
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
        def group_diff(x):
            return pd.Series(x).diff()
        self.trace_input['ts'] -= self.trace_input['ts'].min()
        self.trace_input.sort_values(by=['srcip', 'dstip', 'ts'], inplace=True)
        self.trace_input['iat'] = self.trace_input.groupby(['srcip', 'dstip'])['ts'].apply(group_diff)
        self.trace_input.dropna(how='any', inplace=True)
        # self.trace_input.drop('ts', axis=1, inplace=True)
    
    def select_active_ip(self, lower_bound):
        """
        只保留样本量大于lower_bound的IP对
        """
        self.trace_input['num_of_flow'] = self.trace_input.groupby(['srcip', 'dstip'])['iat'].transform('count')
        self.trace_input = self.trace_input[self.trace_input['num_of_flow']>lower_bound]
        self.trace_input.drop('num_of_flow', axis=1, inplace=True)
        
    def get_id_of_ip(self):
        ip_list = self.trace_input['srcip'].values
        ip_list = np.append(ip_list, self.trace_input['dstip'].values)
        ip_list = list(np.sort(np.unique(ip_list)))     # 所有 ip 按字符顺序排序
        return dict(zip(ip_list, np.arange(len(ip_list))))
    
    def get_ordered_ippair(self) -> list:
        self.trace_input['srcid'] = self.trace_input['srcip'].map(self.ip_id_dict)
        self.trace_input['dstid'] = self.trace_input['dstip'].map(self.ip_id_dict)
        self.trace_input['pairid'] = self.trace_input['srcid'].astype('str') + '_' + \
                                     self.trace_input['dstid'].astype('str')
        self.trace_input.drop(['srcid', 'dstid'], axis=1, inplace=True)
        pair_list = list(np.sort(np.unique(self.trace_input['pairid'].values)))
        return pair_list
    
    def get_cdf(self, col_name):
        arr = np.log10(1.0 + np.sort(self.trace_input[col_name].values))
        percentile = self.auto_percentile(arr, num_of_step=40)
        length = len(arr)-1
        cdf_val = []
        for j in percentile:
            if j==0.0:
                cdf_val.append(0)
            else:
                idx = int(1.0*j*length/10000.0)
                cdf_val.append(arr[idx])
        return dict(zip(percentile, cdf_val))
    
    def auto_percentile(self, arr, num_of_step=40):
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
        percent = cdf['percentile'].values
        bounds = cdf['cdf'].values
        length = len(percent)
        arr_res = []
        for x in arr:
            if x==0.0:
                arr_res.append(0)
            elif x>bounds[-1]:
                # print(f"warning: input x exceed the max value in CDF.\n-> x:{x}, max of CDF:{bounds[-1]}")
                arr_res.append(10000)
            else:
                for i in range(1,length):
                    if (x>bounds[i-1]) and (x<=bounds[i]):
                        arr_res.append(int(percent[i]))
        return arr_res


class PCAPProcessor:
    def __init__(self, f_name, protocol, MAIG):
        self.f_name = f_name
        self.protocol = protocol
        self.MAIG = MAIG
    
    def process_pcap(self):
        if self.protocol=="tcp":
            return self.process_pcap_tcp()
        elif self.protocol=="udp":
            return self.process_pcap_udp()
        else:
            raise ValueError("protocol type undefined!")

    def process_pcap_tcp(self):
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