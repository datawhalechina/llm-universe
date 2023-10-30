#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   self_llm.py
@Time    :   2023/10/16 18:48:08
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   loganzou0421@163.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   在 LangChain LLM 基础上封装的项目类，统一了 GPT、文心、讯飞、智谱多种 API 调用
'''

from langchain.llms.base import LLM
from typing import Dict, Any, Mapping
from pydantic import Field

class Self_LLM(LLM):
    # 自定义 LLM
    # 继承自 langchain.llms.base.LLM
    # 原生接口地址
    url : str =  None
    # 默认选用 GPT-3.5 模型，即目前一般所说的百度文心大模型
    model_name: str = "gpt-3.5-turbo"
    # 访问时延上限
    request_timeout: float = None
    # 温度系数
    temperature: float = 0.1
    # API_Key
    api_key: str = None
    # 必备的可选参数
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    # 定义一个返回默认参数的方法
    @property
    def _default_params(self) -> Dict[str, Any]:
        """获取调用默认参数。"""
        normal_params = {
            "temperature": self.temperature,
            "request_timeout": self.request_timeout,
            }
        # print(type(self.model_kwargs))
        return {**normal_params}
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}