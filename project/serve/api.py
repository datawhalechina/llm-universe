#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   spark_api.py
@Time    :   2023/09/24 11:00:46
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   loganzou0421@163.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   启动服务为本地 API
'''

from fastapi import FastAPI
from pydantic import BaseModel
import os
import sys
# 导入功能模块目录
sys.path.append("../")
from qa_chain.QA_chain_self import QA_chain_self

app = FastAPI() # 创建 api 对象

template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
{context}
问题: {question}
有用的回答:"""

# 定义一个数据模型，用于接收POST请求中的数据
class Item(BaseModel):
    prompt : str # 用户 prompt
    model : str # 使用的模型
    temperature : float = 0.1# 温度系数
    if_history : bool = False # 是否使用历史对话功能
    # API_Key
    api_key: str = None
    # Secret_Key
    secret_key : str = None
    # access_token
    access_token: str = None
    # APPID
    appid : str = None
    # APISecret
    api_secret : str = None
    # 数据库路径
    db_path : str = "../database"
    # prompt template
    prompt_template : str = template
    # Template 变量
    input_variables : list = ["context","question"]
    # Embdding
    embedding : str = "openai"
    # Top K
    top_k : int = 5

@app.post("/answer/")
async def get_response(item: Item):

    # 首先确定需要调用的链
    if not item.if_history:
        # 调用 Chat 链
        chain = QA_chain_self(item.db_path, item.model, item.prompt_template, 
                                   item.input_variables, item.temperature, item.api_key,
                                   item.secret_key, item.access_token, item.appid, item.api_secret, item.embedding)
        
        response = chain.answer(question = item.prompt, top_k = item.top_k, temperature = item.temperature)
    
        return response
    
    # 由于 API 存在即时性问题，不能支持历史链
    else:
        return "API 不支持历史链"