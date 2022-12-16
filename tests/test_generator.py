import os
import pdb
from lomas import utils
from lomas.preprocessor import Preprocessor
from lomas.generator import Generator

if __name__ == "__main__":
    setting_path = os.path.join(os.getcwd(), "./config/data_path_v3.json")
    config = utils.load_json_setting(setting_path)
    trace_name = 'FB_Cluster_C'
    
    data = Preprocessor(f_path=config[trace_name]['path'], 
                        f_name=config[trace_name]['filenames'], 
                        f_type=config[trace_name]['type'],
                        column_names=config[trace_name]['cols'])

    model = Generator(ip_id_dict=data.ip_id_dict, 
                       ordered_ippair=data.ordered_ippair, 
                       cdf_iat=data.cdf_iat, 
                       cdf_size=data.cdf_size)
    model.initialize(data.trace_input)
    model.train(num_topics=25, 
                chunksize=2000, 
                passes=20, 
                iterations=400)
    
    # pdb.set_trace()
    model.generate(time_limit=data.trace_input['ts'].max(), 
                   time_unit=config[trace_name]['ts_unit'])
    
    pdb.set_trace()