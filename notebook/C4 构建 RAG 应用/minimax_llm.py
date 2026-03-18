from typing import Any, Dict, Iterator, List, Optional
from openai import OpenAI
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    SystemMessage,
    ChatMessage,
    HumanMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
import time


MINIMAX_API_BASE = "https://api.minimax.io/v1"


class MinimaxLLM(BaseChatModel):
    """自定义 MiniMax 聊天模型。

    通过 OpenAI 兼容接口调用 MiniMax 大模型 API。
    支持 MiniMax-M2.7、MiniMax-M2.7-highspeed、MiniMax-M2.5、MiniMax-M2.5-highspeed 等模型。
    """

    model_name: str = "MiniMax-M2.7"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    max_retries: int = 3
    api_key: str | None = None

    def _get_client(self) -> OpenAI:
        """创建 OpenAI 兼容客户端。"""
        return OpenAI(
            api_key=self.api_key,
            base_url=MINIMAX_API_BASE,
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """通过调用 MiniMax API 响应输入。

        Args:
            messages: 由 messages 列表组成的 prompt
            stop: 在模型生成的回答中有该字符串列表中的元素则停止响应
            run_manager: 一个为 LLM 提供回调的运行管理器
        """

        messages_dicts = [_convert_message_to_dict(message) for message in messages]
        # MiniMax 温度范围为 (0.0, 1.0]，将超范围值裁剪
        temperature = self.temperature
        if temperature is not None:
            temperature = max(0.01, min(temperature, 1.0))

        start_time = time.time()
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model_name,
            temperature=temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            stop=stop,
            messages=messages_dicts,
        )
        time_in_seconds = time.time() - start_time

        content = response.choices[0].message.content or ""

        message = AIMessage(
            content=content,
            additional_kwargs={},
            response_metadata={
                "time_in_seconds": round(time_in_seconds, 3),
            },
            usage_metadata={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """通过调用 MiniMax API 返回流式输出。

        Args:
            messages: 由 messages 列表组成的 prompt
            stop: 在模型生成的回答中有该字符串列表中的元素则停止响应
            run_manager: 一个为 LLM 提供回调的运行管理器
        """
        messages_dicts = [_convert_message_to_dict(message) for message in messages]
        temperature = self.temperature
        if temperature is not None:
            temperature = max(0.01, min(temperature, 1.0))

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model_name,
            stream=True,
            temperature=temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            stop=stop,
            messages=messages_dicts,
        )
        start_time = time.time()
        usage_metadata = None
        for res in response:
            if hasattr(res, "usage") and res.usage:
                usage_metadata = UsageMetadata(
                    {
                        "input_tokens": res.usage.prompt_tokens,
                        "output_tokens": res.usage.completion_tokens,
                        "total_tokens": res.usage.total_tokens,
                    }
                )
            if res.choices and res.choices[0].delta.content:
                content = res.choices[0].delta.content
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=content)
                )

                if run_manager:
                    run_manager.on_llm_new_token(content, chunk=chunk)

                yield chunk
        time_in_sec = time.time() - start_time
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                response_metadata={"time_in_sec": round(time_in_sec, 3)},
                usage_metadata=usage_metadata,
            )
        )
        if run_manager:
            run_manager.on_llm_new_token("", chunk=chunk)
        yield chunk

    @property
    def _llm_type(self) -> str:
        """获取此聊天模型使用的语言模型类型。"""
        return self.model_name

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回一个标识参数的字典。"""
        return {
            "model_name": self.model_name,
        }


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """把 LangChain 的消息格式转为 MiniMax 支持的格式

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any] = {"content": message.content}
    if (name := message.name or message.additional_kwargs.get("name")) is not None:
        message_dict["name"] = name

    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict


if __name__ == "__main__":
    # Test
    model = MinimaxLLM(model_name="MiniMax-M2.7")
    # invoke
    answer = model.invoke("Hello")
    print(answer)
    answer = model.invoke(
        [
            HumanMessage(content="hello!"),
            AIMessage(content="Hi there human!"),
            HumanMessage(content="Meow!"),
        ]
    )
    print(answer)
    # stream
    for chunk in model.stream(
        [
            HumanMessage(content="hello!"),
            AIMessage(content="Hi there human!"),
            HumanMessage(content="Meow!"),
        ]
    ):
        print(chunk.content, end="|")
    # batch
    print(model.batch(["hello", "goodbye"]))
