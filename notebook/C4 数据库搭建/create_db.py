# 首先实现基本配置
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import UnstructuredFileLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from langchain.llms import OpenAI
from langchain.llms import HuggingFacePipeline

# 使用前配置自己的 api 到环境变量中如
import os
import openai
import sys

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env fileopenai.api_key  = os.environ['OPENAI_API_KEY']
openai.api_key  = os.environ['OPENAI_API_KEY']

#pdf
# 加载 PDF
loaders = [
    PyMuPDFLoader("../../data_base/knowledge_db/pumkin_book/pumpkin_book.pdf") # 机器学习,
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

#md
folder_path = "../../data_base/knowledge_db/prompt_engineering/"
files = os.listdir(folder_path)
loaders = []
for one_file in files:
    loader = UnstructuredMarkdownLoader(os.path.join(folder_path, one_file))
    loaders.append(loader)
for loader in loaders:
    docs.extend(loader.load())

#mp4-txt
loaders = [
    UnstructuredFileLoader("../../data_base/knowledge_db/easy_rl/强化学习入门指南.txt") # 机器学习,
]
for loader in loaders:
    docs.extend(loader.load())

# 切分文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)


# 定义 Embeddings
embedding = OpenAIEmbeddings() 

# 定义持久化路径
persist_directory = '../../data_base/vector_db/chroma'

# 加载数据库
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)

vectordb.persist()