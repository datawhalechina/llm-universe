from typing import List, Optional, Any, Dict
from langchain_core.embeddings import Embeddings
from sparkai.embedding.spark_embedding import Embeddingmodel
import os
import time

class MySparkAIEmbeddings(Embeddings):
    """讯飞星火API的Embedding封装，用于集成到LangChain"""
    
    def __init__(
        self, 
        spark_embedding_app_id: Optional[str] = None,
        spark_embedding_api_key: Optional[str] = None,
        spark_embedding_api_secret: Optional[str] = None,
        spark_embedding_domain: str = "para",
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """初始化讯飞星火Embedding接口"""
        self.spark_embedding_app_id = spark_embedding_app_id or os.environ.get("IFLYTEK_SPARK_APP_ID")
        self.spark_embedding_api_key = spark_embedding_api_key or os.environ.get("IFLYTEK_SPARK_API_KEY")
        self.spark_embedding_api_secret = spark_embedding_api_secret or os.environ.get("IFLYTEK_SPARK_API_SECRET")
        self.spark_embedding_domain = spark_embedding_domain
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.last_request_time = 0  # 记录上次请求时间
        
        if not self.spark_embedding_app_id or not self.spark_embedding_api_key or not self.spark_embedding_api_secret:
            raise ValueError("必须提供讯飞星火API的凭据，请检查环境变量配置")
        
        print(f"初始化讯飞星火Embedding，domain为: {self.spark_embedding_domain}")
        
        try:
            self.client = Embeddingmodel(
                spark_embedding_app_id=self.spark_embedding_app_id,
                spark_embedding_api_key=self.spark_embedding_api_key,
                spark_embedding_api_secret=self.spark_embedding_api_secret,
                spark_embedding_domain=self.spark_embedding_domain
            )
        except Exception as e:
            print(f"初始化讯飞星火客户端时发生错误: {str(e)}")
            raise
    
    def _throttle_request(self):
        """控制请求速率，确保QPS不超过2"""
        current_time = time.time()
        # 计算距离上次请求的时间间隔
        elapsed = current_time - self.last_request_time
        
        # 如果间隔小于0.5秒（确保QPS<=2），则等待
        if elapsed < 0.5:
            sleep_time = 0.5 - elapsed
            time.sleep(sleep_time)
        
        # 更新最后请求时间
        self.last_request_time = time.time()
    
    def _make_embedding_request_with_retry(self, text_data):
        """带重试机制的embedding请求"""
        for attempt in range(self.max_retries):
            try:
                # 控制请求速率
                self._throttle_request()
                
                # 发送请求
                embedding = self.client.embedding(text=text_data)
                return embedding
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"尝试 {attempt+1}/{self.max_retries} 失败: {str(e)}. 将在{self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay)  # 重试前等待
                    # 每次重试增加延迟
                    self.retry_delay *= 1.5
                else:
                    # 最后一次尝试也失败
                    raise e
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将文档列表转换为嵌入向量列表"""
        results = []
        failed_count = 0
        
        for i, text in enumerate(texts):
            try:
                print(f"处理文档 {i+1}/{len(texts)}...")
                embedding = self._make_embedding_request_with_retry({"content": text, "role": "user"})
                results.append(embedding)
            except Exception as e:
                failed_count += 1
                print(f"文档嵌入错误(索引 {i}): {str(e)}")
                # 如果是单个文档失败，返回一个空向量而不是完全失败
                if results:
                    results.append([0.0] * len(results[0]))
                else:
                    # 如果前几个都失败，可能是API问题，暂停一段时间后再尝试
                    if i < 3:
                        print("前几个请求均失败，可能是API问题，暂停10秒后继续...")
                        time.sleep(10)
                        # 重试当前文档
                        try:
                            embedding = self._make_embedding_request_with_retry({"content": text, "role": "user"})
                            results.append(embedding)
                        except Exception as retry_e:
                            print(f"重试仍然失败: {str(retry_e)}")
                            # 如果是第一个文档，我们创建一个默认维度的向量
                            results.append([0.0] * 1024)  # 讯飞星火通常是1024维
                    else:
                        # 否则使用之前向量的维度
                        results.append([0.0] * len(results[0]))
        
        if failed_count > 0:
            print(f"警告: 共有 {failed_count}/{len(texts)} 个文档嵌入失败")
        
        return results
    
    def embed_query(self, text: str) -> List[float]:
        """将查询文本转换为嵌入向量"""
        try:
            # 查询应该使用"query"域，临时切换域
            original_domain = self.client.spark_embedding_domain
            self.client.spark_embedding_domain = "query"
            
            # 获取embedding，应用速率限制
            embedding = self._make_embedding_request_with_retry({"content": text, "role": "user"})
            
            # 恢复原始域设置
            self.client.spark_embedding_domain = original_domain
            return embedding
        except Exception as e:
            print(f"查询嵌入错误: {str(e)}")
            raise