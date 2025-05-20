import os
from typing import Any, Dict, List, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
import dashscope
from dashscope import Generation
from pydantic import Field, PrivateAttr

from dotenv import load_dotenv
load_dotenv()

class QwenLLM(LLM):
    """通义千问大语言模型的LangChain封装"""
    
    model_name: str = Field(default="qwen-max")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=1024)
    top_p: float = Field(default=0.9)
    api_key: Optional[str] = Field(default=None)
    
    _model_kwargs: Dict[str, Any] = PrivateAttr(default_factory=dict)
    
    def __init__(
        self,
        model_name: str = "qwen-max",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        """初始化通义千问模型

        Args:
            model_name: 模型名称，可选 qwen-max、qwen-plus、qwen-turbo 等
            temperature: 温度参数，控制随机性，范围 0-1
            max_tokens: 最大生成token数
            top_p: 核采样阈值
            api_key: API密钥，如不提供则从环境变量获取
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        
        # 设置API密钥
        self.api_key = api_key or os.getenv("QWEN_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API密钥未找到。请通过参数传入或设置环境变量QWEN_API_KEY"
            )
            
        # 设置模型参数
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        
        # 存储额外参数
        self._model_kwargs = kwargs

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """调用通义千问API生成文本

        Args:
            prompt: 输入提示文本
            stop: 停止词列表
            run_manager: 回调管理器
            **kwargs: 其他参数

        Returns:
            生成的文本响应
        """
        # 合并默认参数和传入的参数
        params = {
            "model": self.model_name,
            "prompt": prompt,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            **self._model_kwargs,
            **kwargs,
        }
        
        if stop:
            params["stop"] = stop

        try:
            # 调用API
            response = Generation.call(**params)
            
            # 检查响应状态
            if response.status_code == 200:
                return response.output.text
            else:
                raise RuntimeError(
                    f"API调用失败: {response.code} - {response.message}"
                )
        except Exception as e:
            raise RuntimeError(f"调用通义千问API时发生错误: {str(e)}")

    @property
    def _llm_type(self) -> str:
        """返回LLM类型"""
        return "tongyi_qwen"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回模型标识参数"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            **self._model_kwargs,
        }

    def get_num_tokens(self, text: str) -> int:
        """获取文本的token数量（这里提供一个粗略估计）
        
        Args:
            text: 输入文本
            
        Returns:
            预估的token数量
        """
        # 这里使用一个简单的估算方法，实际应该使用模型的tokenizer
        return len(text) // 4

if __name__ == "__main__":
    # 创建模型实例
    llm = QwenLLM(
        model_name="qwen-max",
        temperature=0.5,
        max_tokens=512,
        api_key=os.getenv("QWEN_API_KEY")  # 从环境变量获取API密钥
    )
    
    # 测试1：基本生成
    print("测试1：基本生成")
    response = llm("请用中文解释量子计算的基本概念")
    print(f"响应:\n{response}\n")
    
    # 测试2：带参数生成
    print("测试2：带参数生成")
    response = llm(
        "写一个关于AI的短故事",
        temperature=0.8,  # 增加创造性
        max_tokens=200    # 限制长度
    )
    print(f"响应:\n{response}\n")
    
    # 测试3：错误处理
    print("测试3：错误处理")
    try:
        # 故意使用错误的参数
        response = llm("测试文本", temperature=2.0)
        print(response)
    except Exception as e:
        print(f"预期的错误: {str(e)}")