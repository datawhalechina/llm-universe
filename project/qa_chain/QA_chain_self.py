from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

import model_to_llm
import get_vectordb


class QA_chain_self():
    """"
    不带历史记录的问答链
    - model：调用的模型名称
    - temperature：温度系数，控制生成的随机性
    - top_k：返回检索的前k个相似文档
    - file_path：建库文件所在路径
    - persist_path：向量数据库持久化路径
    - appid：星火需要输入
    - api_key：所有模型都需要
    - api_secret：星火、百度文心需要输入
    - embeddings：使用的embedding模型  
    - template：可以自定义提示模板，没有输入则使用默认的提示模板default_template_rq    
    """

    #基于召回结果和 query 结合起来构建的 prompt使用的默认提示模版
    default_template_rq = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    有用的回答:"""

    def __init__(self, model:str, temperature:float=0.0, top_k:int=4,  file_path:str=None, persist_path:str=None, appid:str=None, api_key:str=None, api_secret:str=None, embedding = "openai", template=default_template_rq):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.file_path = file_path
        self.persist_path = persist_path
        self.appid = appid
        self.api_key = api_key
        self.api_secret = api_secret
        self.embedding = embedding
        self.template = template
        

        self.vectordb = get_vectordb(self.file_path, self.persist_path, self.api_key, self.embedding)
        self.llm = model_to_llm(self.model, self.temperature, self.appid, self.api_key, self.api_secret)

        self.QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                    template=self.template)
        # 自定义 QA 链
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm,
                                        retriever=self.vectordb.as_retriever(),
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt":self.QA_CHAIN_PROMPT})

    #基于大模型的问答 prompt 使用的默认提示模版
    #default_template_llm = """请回答下列问题:{question}"""
           
    def answer(self, question:str=None, ):
        """"
        核心方法，调用问答链
        arguments: 
        - question：用户提问
        """

        if len(question) == 0:
            return ""
        
        result = self.qa_chain({"query": question, "temperature": self.temperature, "top_k":self.top_k})
        return result["result"]   
