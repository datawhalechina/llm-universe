#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   wenxin_llm.py
@Time    :   2023/12/12 18:24:26
@Author  :   Xu Hu
@Version :   2.0
@Desc    :   基于百度文心大模型自定义 LLM 类
'''

from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional, Dict, Union, Tuple
from pydantic import Field
from self_llm import Self_LLM
import json
import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
import qianfan

class Wenxin_LLM(Self_LLM):
    # 文心大模型的自定义 LLM

    # 默认选用 ERNIE-Bot-turbo 模型，即目前一般所说的百度文心大模型
    model: str = "ERNIE-Bot-turbo"
    # Secret_Key
    secret_key : str = None
    # API_Key
    api_key: str = None
    # Secret_Key
    secret_key : str = None

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
                temperature = self.temperature,
                model = self.model)
            except Exception as e:
                print(e)
        else:
            print("API_Key 或 Secret_Key 为空，请检查 Key")
        
        return resp["result"]
        
    @property
    def _llm_type(self) -> str:
        return "Wenxin"
