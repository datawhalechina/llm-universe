# Embedding封装讲解
LangChain 为基于 LLM 开发自定义应用提供了高效的开发框架，便于开发者迅速地激发 LLM 的强大能力，搭建 LLM 应用。LangChain 也同样支持多种大模型的 Embeddings，内置了 OpenAI、LLAMA 等大模型 Embeddings 的调用接口。但是，LangChain 并没有内置所有大模型，它通过允许用户自定义 Embeddings 类型，来提供强大的可扩展性。

在本部分，我们以智谱 AI 为例，讲述如何基于 LangChain 自定义 Embeddings。

本部分涉及相对更多 LangChain、大模型调用的技术细节，有精力同学可以学习部署，如无精力可以直接使用后续代码来支持调用。

要实现自定义 Embeddings，需要定义一个自定义类继承自 LangChain 的 Embeddings 基类，然后定义两个函数：① embed_query 方法，用于对单个字符串（query）进行 embedding；②embed_documents 方法，用于对字符串列表（documents）进行 embedding。

首先我们导入所需的第三方库：


```python
from typing import List
from langchain_core.embeddings import Embeddings
```

这里我们定义一个继承自 Embeddings 类的自定义 Embeddings 类：


```python
class ZhipuAIEmbeddings(Embeddings):
    """`Zhipuai Embeddings` embedding models."""
    def __init__(self):
        """
        实例化ZhipuAI为values["client"]

        Args:

            values (Dict): 包含配置信息的字典，必须包含 client 的字段.
        Returns:

            values (Dict): 包含配置信息的字典。如果环境中有zhipuai库，则将返回实例化的ZhipuAI类；否则将报错 'ModuleNotFoundError: No module named 'zhipuai''.
        """
        from zhipuai import ZhipuAI
        self.client = ZhipuAI()
```

embed_documents 是对字符串列表（List[str]）计算embedding 的方法，这里我们重写该方法，调用验证环境时实例化的`ZhipuAI`来 调用远程 API 并返回 embedding 结果。


```python
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本列表的 embedding.
        Args:
            texts (List[str]): 要生成 embedding 的文本列表.

        Returns:
            List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """
        embeddings = self.client.embeddings.create(
            model="embedding-3",
            input=texts
        )
        return [embeddings.embedding for embeddings in embeddings.data]
```

`embed_query` 是对单个文本（str）计算 embedding 的方法，这里我们调用刚才定义好的`embed_documents`方法，并返回第一个子列表即可。


```python
    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.

        Args:
            texts (str): 要生成 embedding 的文本.

        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """

        return self.embed_documents([text])[0]
```

对于以上方法可以加入一些内容处理后再请求 embedding，比如文本特别长，我们可以考虑对文本分段，防止超过最大 token 限制，这些都是可以的，靠大家发挥自己的主观能动性完善啦，这里只是给出一个简单的 demo。

通过上述步骤，我们就可以基于 LangChain 与 智谱 AI 定义 embedding 的调用方式了。我们将此代码封装在 zhipuai_embedding.py 文件中。

本文对应源代码在[此处](https://github.com/datawhalechina/llm-universe/blob/main/notebook/C3%20%E6%90%AD%E5%BB%BA%E7%9F%A5%E8%AF%86%E5%BA%93/zhipuai_embedding.py)，如需复现可下载运行源代码。