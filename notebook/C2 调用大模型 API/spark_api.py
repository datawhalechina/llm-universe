#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   spark_api.py
@Time    :   2023/09/24 11:00:46
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   loganzou0421@163.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   将讯飞星火 API 封装成本地 API
'''

from fastapi import FastAPI
from pydantic import BaseModel
import os
import SparkApiSelf

app = FastAPI() # 创建 api 对象

# 定义一个数据模型，用于接收POST请求中的数据
class Item(BaseModel):
    prompt : str # 用户 prompt
    temperature : float # 温度系数
    max_tokens : int # token 上限
    if_list : bool = False # 是否多轮对话

@app.post("/spark/")
async def get_spark_response(item: Item):
    # 实现星火大模型调用的 API 端点
    response = get_spark(item)
    print(response)
    return response


# 首先定义一个构造参数函数
def getText(role, content, text = []):
    # role 是指定角色，content 是 prompt 内容
    jsoncon = {}
    jsoncon["role"] = role
    jsoncon["content"] = content
    text.append(jsoncon)
    return text

def get_spark(item):
    # 配置 spark 秘钥
    #以下密钥信息从控制台获取
    appid = "xxx"     #填写控制台中获取的 APPID 信息
    api_secret = "xxx"   #填写控制台中获取的 APISecret 信息
    api_key ="xxx"    #填写控制台中获取的 APIKey 信息
    domain = "generalv2"    # v2.0版本
    Spark_url = "ws://spark-api.xf-yun.com/v2.1/chat"  # v2.0环境的地址

    # 构造请求参数
    if item.if_list:
        prompt = item.prompt
    else:
        prompt = getText("user", item.prompt)

    response = SparkApiSelf.main(appid,api_key,api_secret,Spark_url,domain,prompt, item.temperature, item.max_tokens)
    return response