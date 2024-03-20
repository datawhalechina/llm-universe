from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional, Dict, Union, Tuple
from pydantic import Field
from zhipuai import ZhipuAI

from self_llm import Self_LLM
import json
import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun


class ZhipuLLM(Self_LLM):
    # 智谱清言GLM大模型的自定义 LLM
    # api_key 继承自Self_LLM
    model: str = "glm-4"

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        client = ZhipuAI(api_key=self.api_key)  # 请填写您自己的APIKey

        # 构造消息
        messages = [
            {"role": "system", "content": "你是一位经验丰富的数据分析师。"},  # system message
            {"role": "user", "content": prompt},
        ]

        # 调用 ChatCompletion 接口
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )

        try:
            return response.choices[0].message.content
        except Exception as e:
            print(e)
            print("请求失败")
            return "请求失败"

    @property
    def _llm_type(self) -> str:
        return "Zhipu"
