import os
import pdb
from lomas import utils
from lomas.preprocessor import Preprocessor

if __name__ == "__main__":
    setting_path = os.path.join(os.getcwd(), "./config/data_path_v2.json")
    config = utils.load_json_setting(setting_path)

    # trace_name = 'FB_Cluster_A'
    # data = Preprocessor(f_path=config[trace_name]['path'], 
    #                     f_name=config[trace_name]['filenames'], 
    #                     f_type=config[trace_name]['type'],
    #                     column_names={'srcip':'srcip', 
    #                                   'dstip':'dstip', 
    #                                   'timestamp':'ts',
    #                                   'packetlength':'size'})

    trace_name = 'UN1'
    data = Preprocessor(f_path=config[trace_name]['path'], 
                        f_name=config[trace_name]['filenames'], 
                        f_type=config[trace_name]['type'],
                        protocol=config[trace_name]['protocol'],
                        MAIG=config[trace_name]['MAIG'])

    pdb.set_trace()