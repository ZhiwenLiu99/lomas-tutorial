import json

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