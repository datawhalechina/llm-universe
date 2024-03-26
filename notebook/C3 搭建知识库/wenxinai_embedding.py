from __future__ import annotations
import os

import logging
from typing import Any, Dict, List, Optional

import json
import requests
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator

logger = logging.getLogger(__name__)

class WenxinAIEmbeddings(BaseModel, Embeddings):

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """
        通过环境变量获取values["wenxin_access_token"]

        Args:
            values (Dict): 包含配置信息的字典，必须包含 wenxin_access_token 的字段
        Returns:

            values (Dict): 包含配置信息的字典。如果环境变量或配置文件中未提供 wenxin_access_token，则将返回原始值；否则将返回包含 zhipuai_api_key 的值。
        """
        _ = load_dotenv(find_dotenv())
        values["WENXIN_ACCESS_TOKEN"] = os.environ['WENXIN_ACCESS_TOKEN']
        
        return values
    
    def _embed(self, texts: str) -> List[float]:
        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1?access_token=" + self.wenxin_access_token
        input = []
        input.append(texts)
        payload = json.dumps({
            "input": input
        })
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        return json.loads(response.text)['data'][0]['embedding']

    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding。

        Args:
            texts (str): 要生成 embedding 的文本。

        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表。
        """
        resp = self.embed_documents([text])
        return resp[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本列表的 embedding。
        Args:
            texts (List[str]): 要生成 embedding 的文本列表.

        Returns:
            List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """
        return [self._embed(text) for text in texts]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        raise NotImplementedError("Please use `embed_documents`. Official does not support asynchronous requests")

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        raise NotImplementedError("Please use `aembed_query`. Official does not support asynchronous requests")