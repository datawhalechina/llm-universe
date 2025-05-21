from typing import List
from langchain_core.embeddings import Embeddings
import os
import dashscope
from dashscope import TextEmbedding

class DashscopeEmbeddings(Embeddings):
    """`OpenAI Embeddings` embedding models."""
    def __init__(self):
        """
        实例化OpenAI为values["client"]

        Args:

            values (Dict): 包含配置信息的字典，必须包含 client 的字段.
        Returns:

            values (Dict): 包含配置信息的字典。如果环境中有openai库，则将返回实例化的OpenAI类；否则将报错 'ModuleNotFoundError: No module named 'openai''.
        """
        
        dashscope.api_key = os.getenv('OPENAI_API_KEY')
        self.model_name = "text-embedding-v1"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本列表的 embedding.
        Args:
            texts (List[str]): 要生成 embedding 的文本列表.

        Returns:
            List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """

        embeddings = []
        batch_size = 25  # DashScope 单次请求最多 25 条文本
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = TextEmbedding.call(
                model=self.model_name,
                input=batch
            )
            if response.status_code == 200:
                embeddings.extend([item['embedding'] for item in response.output['embeddings']])
            else:
                raise ValueError(
                    f"DashScope Embedding 请求失败，错误码: {response.code}, 错误信息: {response.message}"
                )
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.

        Args:
            texts (str): 要生成 embedding 的文本.

        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """

        return self.embed_documents([text])[0]