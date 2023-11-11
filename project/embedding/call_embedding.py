import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from embedding.zhipuai_embedding import ZhipuAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings

def get_embedding(embedding: str):
    if embedding == "openai":
        return OpenAIEmbeddings()
    elif embedding == "zhipuai":
        return ZhipuAIEmbeddings()
    elif embedding == 'm3e':
        return HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
    else:
        raise ValueError(f"embedding {embedding} not support ")
