#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   wenxin_llm.py
@Time    :   2023/12/12 17:45:50
@Author  :   Xu Hu 
@Version :   2.0
@Desc    :   基于 LangChain 定义文心模型调用方式
'''

import json
import time
from typing import Any, List, Mapping, Optional, Dict, Union, Tuple
import requests
from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env
from pydantic import Field, root_validator
from langchain.callbacks.manager import CallbackManagerForLLMRun
import qianfan



# 继承自 langchain.llms.base.LLM
class Wenxin_LLM(LLM):
    # 默认选用 ERNIE-Bot-turbo 模型，即目前一般所说的百度文心大模型
    model_name: str = "ERNIE-Bot-turbo"
    # 访问时延上限
    request_timeout: float = None
    # 温度系数
    temperature: float = 0.1
    # API_Key
    api_key: str = None
    # Secret_Key
    secret_key : str = None
    # 必备的可选参数
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)


    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        if self.api_key != None and self.secret_key != None:
            # 两个 Key 均非空才可以获取 access_token
            try:
                chat_comp = qianfan.ChatCompletion()
                # 调用默认模型，即 ERNIE-Bot-turbo
                resp = chat_comp.do(messages=[{
                    "role": "user",# user prompt
                    "content": "{}".format(prompt)# 输入的 prompt
                }],
                temperature = self.temperature)
            except Exception as e:
                print(e)
        else:
            print("API_Key 或 Secret_Key 为空，请检查 Key")
        
        return resp["result"]
        
        
    # 首先定义一个返回默认参数的方法
    @property
    def _default_params(self) -> Dict[str, Any]:
        """获取调用Ennie API的默认参数。"""
        normal_params = {
            "temperature": self.temperature,
            "request_timeout": self.request_timeout,
            }
        # print(type(self.model_kwargs))
        return {**normal_params}

    @property
    def _llm_type(self) -> str:
        return "Wenxin"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}
