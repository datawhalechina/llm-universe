import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from embedding.zhipuai_embedding import ZhipuAIEmbeddings

def get_embedding(embedding : str):
    if embedding == "openai":
        embedding = OpenAIEmbeddings() 
    elif embedding == "zhipuai":
        embedding = ZhipuAIEmbeddings()
    elif name == 'm3e':
        embedding = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
    return embedding
