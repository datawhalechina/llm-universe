import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from embedding.zhipuai_embedding import ZhipuAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.dashscope import DashScopeEmbeddings
from langchain.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint
from llm.call_llm import parse_llm_api_key

def get_embedding(embedding: str, embedding_key: str=None, env_file: str=None):
    if embedding == "openai":
        return OpenAIEmbeddings()
    elif embedding == "zhipuai":
        return ZhipuAIEmbeddings()
    elif embedding in ["text-embedding-v1(dashscope)", "text-embedding-v2(dashscope)"]:
        embedding = embedding.split("(")[0]
        return DashScopeEmbeddings(model=embedding)
    elif embedding in ["Embedding-V1(qianfan)", "bge-large-en", "bge-large-zh"]:
        embedding = embedding.split("(")[0]
        return QianfanEmbeddingsEndpoint(model=embedding)
    elif embedding == 'm3e':
        return HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
    else:
        raise ValueError(f"embedding {embedding} not support ")
