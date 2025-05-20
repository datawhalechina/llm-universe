import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict, Any

from chromadb.utils import embedding_functions
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

_ = load_dotenv(find_dotenv())

class TCMRAGSystem:
    def __init__(self, persist_dir: str = "tcm_chroma_db"):
        # 使用默认的嵌入模型（或自定义）
        self.embedding_model = embedding_functions.DefaultEmbeddingFunction()
        # 初始化 Chroma 客户端，启用持久化
        self.client = chromadb.PersistentClient(path=persist_dir)
        # 创建或加载集合
        self.collection = self.client.get_or_create_collection(
            name="tcm_knowledge",
            embedding_function=self.embedding_model
        )

    def add_node(self, name: str, node_type: str) -> str:
        """添加节点到集合"""
        node_id = f"node_{len(self.collection.get()['ids']) + 1}"
        self.collection.add(
            documents=[name],
            metadatas=[{"type": node_type}],
            ids=[node_id]
        )
        return node_id

    def add_relation(self, node1_id: str, node2_id: str, relation: str) -> str:
        """添加关系到集合"""
        # 获取节点名称
        node1_name = self.collection.get(ids=[node1_id])["documents"][0]
        node2_name = self.collection.get(ids=[node2_id])["documents"][0]
        relation_text = f"{node1_name} {relation} {node2_name}"
        relation_id = f"relation_{len(self.collection.get()['ids']) + 1}"
        self.collection.add(
            documents=[relation_text],
            metadatas=[{"type": "relation", "relation": relation}],
            ids=[relation_id]
        )
        return relation_id

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """检索最相关的节点或关系"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return [
            {
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            }
            for i in range(top_k)
        ]

class TCMKnowledgeGraph:
    def __init__(self, persist_dir: str = "tcm_chroma_db", model_name: str = "text-embedding-v3"):
        """
        初始化中医药知识图谱系统，直接加载向量数据库

        参数:
            persist_dir: 向量数据库持久化目录
            model_name: 使用的大模型名称
        """
        self.persist_directory = persist_dir
        self.model_name = model_name

        # 初始化Chroma客户端
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # 直接加载集合
        self.collection = self.client.get_or_create_collection(
            name="tcm_knowledge",
            metadata={"description": "中医药知识图谱"}
        )
        
        # 初始化对话历史
        self.chat_history = []

        # 初始化OpenAI客户端
        self.openai_client = OpenAI(
            base_url=os.getenv("QWEN_BASE_URL"),
            api_key=os.environ.get("QWEN_API_KEY")
        )

    def _get_embedding(self, text: str) -> List[float]:
        """获取文本的嵌入向量"""
        try:
            response = self.openai_client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise ValueError(f"获取嵌入向量失败: {str(e)}")

    def chat_query(self, question: str) -> Dict[str, Any]:
        """
        使用大模型进行问答交互

        参数:
            question: 用户问题

        返回:
            dict: 包含答案和相关文档的字典
        """
        # 获取问题的嵌入向量
        query_embedding = self._get_embedding(question)

        # 检索相关文档
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        # 构建提示词
        context = "\n".join(results["documents"][0])
        messages = [
            {"role": "system", "content": "你是一个专业的中医知识助手。请基于提供的相关信息回答用户的问题。如果无法从给定信息中找到答案，请明确说明。"},
            {"role": "user", "content": f"基于以下相关信息回答问题:\n\n相关信息:\n{context}\n\n问题: {question}"}
        ]
        messages.extend(self.chat_history)

        # 调用OpenAI API获取回答
        response = self.openai_client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            temperature=0.7
        )

        # 更新对话历史
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": response.choices[0].message.content})

        return {
            "answer": response.choices[0].message.content,
            "source_documents": results["documents"][0]
        }

    def similarity_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        语义相似度搜索

        参数:
            query: 查询文本
            k: 返回结果数量

        返回:
            相似文档列表
        """
        query_embedding = self._get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        return [
            {
                "text": doc,
                "metadata": meta
            }
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

# 使用示例
if __name__ == "__main__":
    # 初始化系统（直接加载向量数据库）
    tcm_kg = TCMKnowledgeGraph()

    # 示例查询
    question = "鱼脑有什么功效"
    print(f"\n问题: {question}")

    # 使用大模型进行问答
    print("\n大模型回答:")
    response = tcm_kg.chat_query(question)
    print(f"答案: {response['answer']}")
    print("\n参考文档:")
    for i, doc in enumerate(response['source_documents']):
        print(f"{i + 1}. {doc}")

    # 语义搜索
    print("\n语义搜索结果:")
    sim_docs = tcm_kg.similarity_search(question, k=2)
    for i, doc in enumerate(sim_docs):
        print(f"{i + 1}. {doc['text']} (来源: {doc['metadata']['source']})")