import os
from typing import Any, Dict, Optional
from dotenv import load_dotenv, find_dotenv

def parse_llm_api_key(model:str, env_file:Dict[str, Any]=None):
    """
    通过 model 和 env_file 的来解析平台参数
    model: 模型平台的名字
    env_file: 存放所有平台参数的文件
    """   
    if env_file == None:
        _ = load_dotenv(find_dotenv())
        env_file = os.environ
    if model == "openai":
        return get_from_dict_or_env(env_file, "open_api_key", "OPENAI_API_KEY")
    elif model == "wenxin":
        return get_from_dict_or_env(env_file, "wenxin_api_key", "WENXIN_API_KEY"), get_from_dict_or_env(env_file, "wenxin_secret_key", "WENXIN_SECRET_KEY")
    elif model == "spark":
        return get_from_dict_or_env(env_file, "spark_api_key", "SPARK_API_KEY"), get_from_dict_or_env(env_file, "spark_appid", "SPARK_APPID"), get_from_dict_or_env(env_file, "spark_api_secret", "SPARK_API_SECRET")
    elif model == "zhipuai":
        return get_from_dict_or_env(env_file, "zhipuai_api_key", "ZHIPUAI_API_KEY")
        # return env_file["ZHIPUAI_API_KEY"]
    else:
        raise ValueError(f"model {model} not support!!!")


def get_from_dict_or_env(
    data: Dict[str, Any], key: str, env_key: str, default: Optional[str] = None, uniform: bool = True
) -> str:
    """
    从字典或环境变量中获取值。（from langchain.utils）
    data: 存放平台参数的字典
    key: 要在 data 中查找的平台参数
    env_key: 要在环境变量中查找的平台参数（通常大写）
    """
    if key in data and data[key]:
        return data[key]
    if uniform:
        if key.upper() in data and data[key.upper()]:
            return data[key.upper() ]
        elif key.lower() in data and data[key.lower()]:
            return data[key.lower()]
    else:
        return get_from_env(key, env_key, default=default)


def get_from_env(key: str, env_key: str, default: Optional[str] = None, uniform: bool = True) -> str:
    """从字典或环境变量中获取值。（from langchain.utils）"""
    if env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    if uniform:
        if env_key.upper() in os.environ and os.environ[env_key.upper()]:
            return os.environ[env_key.upper()]
        if env_key.lower() in os.environ and os.environ[env_key.lower()]:
            return os.environ[env_key.lower()]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or add"
            f"  `{key}` in the passed data file."
        )
        
