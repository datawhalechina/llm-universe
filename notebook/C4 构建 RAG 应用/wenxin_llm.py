
# 基于 LangChain 定义文心模型调用方式

from typing import Any, List, Mapping, Optional, Dict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
import qianfan

# 继承自 langchain_core.language_models.llms.LLM
class Wenxin_LLM(LLM):
    # 默认选用 ERNIE-Bot-turbo 模型，即目前一般所说的百度文心大模型
    model: str = "ERNIE-Bot-turbo"
    # 温度系数
    temperature: float = 0.1
    # API_Key
    api_key: str = None
    # Secret_Key
    secret_key : str = None
    # 系统消息
    system : str = None



    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        def gen_wenxin_messages(prompt):
            '''
            构造文心模型请求参数 messages

            请求参数：
                prompt: 对应的用户提示词
            '''
            messages = [{"role": "user", "content": prompt}]
            return messages
        
        chat_comp = qianfan.ChatCompletion(ak=self.api_key,sk=self.secret_key)
        message = gen_wenxin_messages(prompt)

        resp = chat_comp.do(messages = message, 
                            model= self.model,
                            temperature = self.temperature,
                            system = self.system)

        return resp["result"]
        
    # 首先定义一个返回默认参数的方法
    @property
    def _default_params(self) -> Dict[str, Any]:
        """获取调用Ennie API的默认参数。"""
        normal_params = {
            "temperature": self.temperature,
            }
        # print(type(self.model_kwargs))
        return {**normal_params}

    @property
    def _llm_type(self) -> str:
        return "Wenxin"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}
