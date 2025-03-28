from typing import List
from langchain_core.embeddings import Embeddings
from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
import streamlit as st
from langchain_openai import ChatOpenAI
from openai import OpenAI
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_community.vectorstores import Chroma
import torch
from transformers import AutoModel, AutoTokenizer
from config import *

# Embedding 封装
class Embeddings(Embeddings):
    def __init__(self):
        """
        先加载模型
        """          
        model_name_or_path = 'embedding_model_small'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, device_map="auto")

    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本列表的 embedding.
        Args:
            texts (List[str]): 要生成 embedding 的文本列表.

        Returns:
            List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """

        result = []
        for i in range(0, len(texts), 16):
            input_texts = texts[i:i+16]
            batch_dict = self.tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt').to("cuda")
            outputs = self.model(**batch_dict)
            dimension=768 # The output dimension of the output embedding, should be in [128, 768]
            embeddings = outputs.last_hidden_state[:, 0][:dimension]
            result.extend([item.cpu().detach().numpy() for item in embeddings])
        return result
    
    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.

        Args:
            texts (str): 要生成 embedding 的文本.

        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """
        batch_dict = self.tokenizer([text], max_length=8192, padding=True, truncation=True, return_tensors='pt').to("cuda")
        outputs = self.model(**batch_dict)
        dimension=768 # The output dimension of the output embedding, should be in [128, 768]
        embeddings = outputs.last_hidden_state[:, 0][:dimension]
        return embeddings[0].cpu().detach().numpy()

# LLM 封装

class DeepSeekLLM(LLM):
    # 基于 SCNet 的 DeepSeek 接口自定义 LLM 类
    api_key : str = None
    model: str = None
    client: OpenAI = None

    def __init__(self, api_key=None, model="DeepSeek-R1-Distill-Qwen-32B"):
        # model：选择使用的模型
        # 从本地初始化模型
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.scnet.cn/api/llm/v1")

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        # 重写调用函数
        response = self.client.chat.completions.create(
        model=self.model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
        )
        
        return response.choices[0].message.content
        
    @property
    def _llm_type(self) -> str:
        return "SCNet"

# 1. 定义 get_retriever 函数，该函数返回一个检索器
def get_retriever():
    # 定义 Embeddings
    embedding = Embeddings()
    # 向量数据库持久化路径
    persist_directory = 'data/chroma'
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    return vectordb.as_retriever()

# 2. 定义combine_docs函数， 该函数处理检索器返回的文本
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"])

# 3. 定义get_qa_history_chain函数，该函数可以返回一个检索问答链
def get_qa_history_chain():
    retriever = get_retriever()
    llm = llm = DeepSeekLLM(api_key = DEEPSEEK_API_KEY)
    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题，"
        "如果没有多余的聊天记录则返回用户的问题。"
    )
    condense_question_prompt = ChatPromptTemplate([
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

    retrieve_docs = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    system_prompt = (
        "你是一个问答任务的助手。 "
        "请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说不知道。 "
        "请使用简洁的话语回答用户。"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = RunnablePassthrough().assign(
        context = retrieve_docs, 
        ).assign(answer=qa_chain)
    return qa_history_chain

# 4. 定义gen_response函数，它接受检索问答链、用户输入及聊天历史，并以流式返回该链输出
def gen_response(chain, input, chat_history):
    response = chain.stream({
        "input": input,
        "chat_history": chat_history
    })
    for res in response:
        if "answer" in res.keys():
            yield res["answer"]

# 5. 定义main函数，该函数制定显示效果与逻辑
def main():
    st.markdown('### 🦜🔗 基于 DeepSeek R1 搭建 RAG 应用')
    # st.session_state可以存储用户与应用交互期间的状态与数据
    # 存储对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 存储检索问答链
    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = get_qa_history_chain()
    # 建立容器 高度为500 px
    messages = st.container(height=550)
    # 显示整个对话历史
    for message in st.session_state.messages: # 遍历对话历史
            with messages.chat_message(message[0]): # messages指在容器下显示，chat_message显示用户及ai头像
                st.write(message[1]) # 打印内容
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append(("human", prompt))
        # 显示当前用户输入
        with messages.chat_message("human"):
            st.write(prompt)
        # 生成回复
        answer = gen_response(
            chain=st.session_state.qa_history_chain,
            input=prompt,
            chat_history=st.session_state.messages
        )
        # 流式输出
        with messages.chat_message("ai"):
            output = st.write_stream(answer)
        # 将输出存入st.session_state.messages
        st.session_state.messages.append(("ai", output))

if __name__ == "__main__":
    main()