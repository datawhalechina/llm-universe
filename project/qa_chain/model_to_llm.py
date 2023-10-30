from wenxin_llm import Wenxin_LLM
from spark_llm import Spark_LLM
from zhipuai_llm.py import ZhipuAILLM
from langchain.chat_models import ChatOpenAI


def model_to_llm(model:str=None, temperature:float=0.0, appid:str=None, api_key:str=None, api_secret:str=None):
        """
        星火：model,temperature,appid,api_key,api_secret
        百度问心：model,temperature,api_key,api_secret
        智谱：model,temperature,api_key
        OpenAI：model,temperature,api_key
        """
        if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"]:
            llm = ChatOpenAI(model_name = model, temperature = temperature , openai_api_key = api_key)
        elif model in ["ERNIE-Bot", "ERNIE-Bot-4", "ERNIE-Bot-turbo"]:
            llm = Wenxin_LLM(model=model, temperature = temperature, api_key=api_key, secret_key=api_secret)
        elif model in ["Spark-1.5", "Spark-2.0"]:
            llm = Spark_LLM(model=model, temperature = temperature, appid=appid, api_secret=api_secret, api_key=api_key)
        elif model in ["chatglm_pro", "chatglm_std", "chatglm_lite"]:
            llm = ZhipuAILLM(model=model, api_key=api_key, temperature = temperature)
        else:
            return "不正确的模型"
        return llm