"""MiniMax LLM wrapper 的单元测试和集成测试。"""

import os
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ChatMessage,
)

# 导入待测模块
from minimax_llm import MinimaxLLM, _convert_message_to_dict, MINIMAX_API_BASE


# ======================== 单元测试 ========================


class TestConvertMessageToDict(unittest.TestCase):
    """测试消息格式转换函数。"""

    def test_human_message(self):
        msg = HumanMessage(content="你好")
        result = _convert_message_to_dict(msg)
        self.assertEqual(result, {"content": "你好", "role": "user"})

    def test_ai_message(self):
        msg = AIMessage(content="你好！有什么可以帮你？")
        result = _convert_message_to_dict(msg)
        self.assertEqual(result, {"content": "你好！有什么可以帮你？", "role": "assistant"})

    def test_system_message(self):
        msg = SystemMessage(content="你是一个助手")
        result = _convert_message_to_dict(msg)
        self.assertEqual(result, {"content": "你是一个助手", "role": "system"})

    def test_chat_message(self):
        msg = ChatMessage(content="test", role="custom_role")
        result = _convert_message_to_dict(msg)
        self.assertEqual(result, {"content": "test", "role": "custom_role"})

    def test_unknown_message_type(self):
        from langchain_core.messages import BaseMessage

        class UnknownMessage(BaseMessage):
            type: str = "unknown"

        msg = UnknownMessage(content="test")
        with self.assertRaises(TypeError):
            _convert_message_to_dict(msg)


class TestMinimaxLLMInit(unittest.TestCase):
    """测试 MinimaxLLM 初始化。"""

    def test_default_model_name(self):
        llm = MinimaxLLM(api_key="test-key")
        self.assertEqual(llm.model_name, "MiniMax-M2.7")

    def test_custom_model_name(self):
        llm = MinimaxLLM(model_name="MiniMax-M2.7-highspeed", api_key="test-key")
        self.assertEqual(llm.model_name, "MiniMax-M2.7-highspeed")

    def test_llm_type(self):
        llm = MinimaxLLM(model_name="MiniMax-M2.7", api_key="test-key")
        self.assertEqual(llm._llm_type, "MiniMax-M2.7")

    def test_identifying_params(self):
        llm = MinimaxLLM(model_name="MiniMax-M2.7", api_key="test-key")
        self.assertEqual(llm._identifying_params, {"model_name": "MiniMax-M2.7"})

    def test_temperature_and_max_tokens(self):
        llm = MinimaxLLM(
            api_key="test-key",
            temperature=0.7,
            max_tokens=512,
        )
        self.assertEqual(llm.temperature, 0.7)
        self.assertEqual(llm.max_tokens, 512)

    def test_api_base_url(self):
        self.assertEqual(MINIMAX_API_BASE, "https://api.minimax.io/v1")


class TestMinimaxLLMGetClient(unittest.TestCase):
    """测试 OpenAI 客户端创建。"""

    @patch("minimax_llm.OpenAI")
    def test_client_uses_minimax_base_url(self, mock_openai):
        llm = MinimaxLLM(api_key="test-key")
        llm._get_client()
        mock_openai.assert_called_once_with(
            api_key="test-key",
            base_url=MINIMAX_API_BASE,
        )


class TestMinimaxLLMGenerate(unittest.TestCase):
    """测试 _generate 方法。"""

    def _make_mock_response(self, content="Hello!", prompt_tokens=10, completion_tokens=5):
        """创建模拟 API 响应。"""
        response = MagicMock()
        response.choices = [
            MagicMock(message=MagicMock(content=content))
        ]
        response.usage = MagicMock(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return response

    @patch("minimax_llm.OpenAI")
    def test_generate_basic(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response()
        mock_openai_cls.return_value = mock_client

        llm = MinimaxLLM(api_key="test-key")
        result = llm._generate([HumanMessage(content="你好")])

        self.assertEqual(len(result.generations), 1)
        self.assertIsInstance(result.generations[0].message, AIMessage)
        self.assertEqual(result.generations[0].message.content, "Hello!")

    @patch("minimax_llm.OpenAI")
    def test_generate_with_system_message(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response("OK")
        mock_openai_cls.return_value = mock_client

        llm = MinimaxLLM(api_key="test-key")
        result = llm._generate([
            SystemMessage(content="你是助手"),
            HumanMessage(content="你好"),
        ])

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")

    @patch("minimax_llm.OpenAI")
    def test_generate_usage_metadata(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            prompt_tokens=20, completion_tokens=30
        )
        mock_openai_cls.return_value = mock_client

        llm = MinimaxLLM(api_key="test-key")
        result = llm._generate([HumanMessage(content="test")])

        usage = result.generations[0].message.usage_metadata
        self.assertEqual(usage["input_tokens"], 20)
        self.assertEqual(usage["output_tokens"], 30)
        self.assertEqual(usage["total_tokens"], 50)

    @patch("minimax_llm.OpenAI")
    def test_generate_time_metadata(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response()
        mock_openai_cls.return_value = mock_client

        llm = MinimaxLLM(api_key="test-key")
        result = llm._generate([HumanMessage(content="test")])

        metadata = result.generations[0].message.response_metadata
        self.assertIn("time_in_seconds", metadata)
        self.assertIsInstance(metadata["time_in_seconds"], float)

    @patch("minimax_llm.OpenAI")
    def test_temperature_clamping_low(self, mock_openai_cls):
        """温度低于 0.01 时应被裁剪到 0.01。"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response()
        mock_openai_cls.return_value = mock_client

        llm = MinimaxLLM(api_key="test-key", temperature=0.0)
        llm._generate([HumanMessage(content="test")])

        call_args = mock_client.chat.completions.create.call_args
        self.assertAlmostEqual(call_args.kwargs["temperature"], 0.01)

    @patch("minimax_llm.OpenAI")
    def test_temperature_clamping_high(self, mock_openai_cls):
        """温度高于 1.0 时应被裁剪到 1.0。"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response()
        mock_openai_cls.return_value = mock_client

        llm = MinimaxLLM(api_key="test-key", temperature=2.0)
        llm._generate([HumanMessage(content="test")])

        call_args = mock_client.chat.completions.create.call_args
        self.assertAlmostEqual(call_args.kwargs["temperature"], 1.0)

    @patch("minimax_llm.OpenAI")
    def test_temperature_none_passthrough(self, mock_openai_cls):
        """温度为 None 时应原样传递。"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response()
        mock_openai_cls.return_value = mock_client

        llm = MinimaxLLM(api_key="test-key")
        llm._generate([HumanMessage(content="test")])

        call_args = mock_client.chat.completions.create.call_args
        self.assertIsNone(call_args.kwargs["temperature"])

    @patch("minimax_llm.OpenAI")
    def test_generate_empty_content(self, mock_openai_cls):
        """API 返回 None content 时应转为空字符串。"""
        mock_client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content=None))]
        response.usage = MagicMock(prompt_tokens=5, completion_tokens=0, total_tokens=5)
        mock_client.chat.completions.create.return_value = response
        mock_openai_cls.return_value = mock_client

        llm = MinimaxLLM(api_key="test-key")
        result = llm._generate([HumanMessage(content="test")])
        self.assertEqual(result.generations[0].message.content, "")

    @patch("minimax_llm.OpenAI")
    def test_generate_with_stop(self, mock_openai_cls):
        """测试 stop 参数传递。"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response()
        mock_openai_cls.return_value = mock_client

        llm = MinimaxLLM(api_key="test-key")
        llm._generate([HumanMessage(content="test")], stop=["END"])

        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args.kwargs["stop"], ["END"])

    @patch("minimax_llm.OpenAI")
    def test_generate_model_name_passthrough(self, mock_openai_cls):
        """测试模型名称正确传递。"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response()
        mock_openai_cls.return_value = mock_client

        llm = MinimaxLLM(api_key="test-key", model_name="MiniMax-M2.7-highspeed")
        llm._generate([HumanMessage(content="test")])

        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args.kwargs["model"], "MiniMax-M2.7-highspeed")

    @patch("minimax_llm.OpenAI")
    def test_legacy_model_name_passthrough(self, mock_openai_cls):
        """测试旧版模型名称正确传递。"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response()
        mock_openai_cls.return_value = mock_client

        llm = MinimaxLLM(api_key="test-key", model_name="MiniMax-M2.5")
        llm._generate([HumanMessage(content="test")])

        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args.kwargs["model"], "MiniMax-M2.5")


class TestMinimaxLLMStream(unittest.TestCase):
    """测试 _stream 方法。"""

    @patch("minimax_llm.OpenAI")
    def test_stream_basic(self, mock_openai_cls):
        mock_client = MagicMock()

        # 模拟流式响应
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="你"))]
        chunk1.usage = None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content="好"))]
        chunk2.usage = None

        chunk3 = MagicMock()
        chunk3.choices = [MagicMock(delta=MagicMock(content="！"))]
        chunk3.usage = MagicMock(prompt_tokens=5, completion_tokens=3, total_tokens=8)

        mock_client.chat.completions.create.return_value = iter([chunk1, chunk2, chunk3])
        mock_openai_cls.return_value = mock_client

        llm = MinimaxLLM(api_key="test-key")
        chunks = list(llm._stream([HumanMessage(content="test")]))

        # 3 content chunks + 1 final metadata chunk
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0].message.content, "你")
        self.assertEqual(chunks[1].message.content, "好")
        self.assertEqual(chunks[2].message.content, "！")
        self.assertEqual(chunks[3].message.content, "")  # final chunk

    @patch("minimax_llm.OpenAI")
    def test_stream_calls_create_with_stream_true(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([])
        mock_openai_cls.return_value = mock_client

        llm = MinimaxLLM(api_key="test-key")
        list(llm._stream([HumanMessage(content="test")]))

        call_args = mock_client.chat.completions.create.call_args
        self.assertTrue(call_args.kwargs["stream"])

    @patch("minimax_llm.OpenAI")
    def test_stream_temperature_clamping(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([])
        mock_openai_cls.return_value = mock_client

        llm = MinimaxLLM(api_key="test-key", temperature=0.0)
        list(llm._stream([HumanMessage(content="test")]))

        call_args = mock_client.chat.completions.create.call_args
        self.assertAlmostEqual(call_args.kwargs["temperature"], 0.01)


class TestMinimaxLLMLangChainInterface(unittest.TestCase):
    """测试 LangChain 接口兼容性。"""

    @patch("minimax_llm.OpenAI")
    def test_invoke(self, mock_openai_cls):
        mock_client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="回复"))]
        response.usage = MagicMock(prompt_tokens=5, completion_tokens=2, total_tokens=7)
        mock_client.chat.completions.create.return_value = response
        mock_openai_cls.return_value = mock_client

        llm = MinimaxLLM(api_key="test-key")
        result = llm.invoke("你好")
        self.assertIsInstance(result, AIMessage)
        self.assertEqual(result.content, "回复")

    @patch("minimax_llm.OpenAI")
    def test_invoke_with_message_list(self, mock_openai_cls):
        mock_client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="Meow back!"))]
        response.usage = MagicMock(prompt_tokens=10, completion_tokens=3, total_tokens=13)
        mock_client.chat.completions.create.return_value = response
        mock_openai_cls.return_value = mock_client

        llm = MinimaxLLM(api_key="test-key")
        result = llm.invoke([
            HumanMessage(content="hello!"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="Meow!"),
        ])
        self.assertEqual(result.content, "Meow back!")

    @patch("minimax_llm.OpenAI")
    def test_batch(self, mock_openai_cls):
        mock_client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="回复"))]
        response.usage = MagicMock(prompt_tokens=5, completion_tokens=2, total_tokens=7)
        mock_client.chat.completions.create.return_value = response
        mock_openai_cls.return_value = mock_client

        llm = MinimaxLLM(api_key="test-key")
        results = llm.batch(["hello", "goodbye"])
        self.assertEqual(len(results), 2)


# ======================== 集成测试 ========================


@unittest.skipUnless(
    os.environ.get("MINIMAX_API_KEY"),
    "Set MINIMAX_API_KEY to run integration tests",
)
class TestMinimaxLLMIntegration(unittest.TestCase):
    """集成测试：需要真实 API 密钥。"""

    def setUp(self):
        self.api_key = os.environ["MINIMAX_API_KEY"]
        self.llm = MinimaxLLM(
            model_name="MiniMax-M2.7",
            api_key=self.api_key,
            temperature=0.7,
        )

    def test_invoke_returns_content(self):
        result = self.llm.invoke("Say hello in one word.")
        self.assertIsInstance(result, AIMessage)
        self.assertTrue(len(result.content) > 0)

    def test_stream_returns_chunks(self):
        chunks = list(self.llm.stream("Say hi in one word."))
        self.assertTrue(len(chunks) > 0)
        full_content = "".join(c.content for c in chunks)
        self.assertTrue(len(full_content) > 0)

    def test_highspeed_model(self):
        llm = MinimaxLLM(
            model_name="MiniMax-M2.7-highspeed",
            api_key=self.api_key,
            temperature=0.5,
        )
        result = llm.invoke("What is 2+2? Answer with just the number.")
        self.assertIsInstance(result, AIMessage)
        self.assertIn("4", result.content)

    def test_legacy_m25_model(self):
        """旧版 M2.5 模型仍可正常使用。"""
        llm = MinimaxLLM(
            model_name="MiniMax-M2.5",
            api_key=self.api_key,
            temperature=0.5,
        )
        result = llm.invoke("Say hello in one word.")
        self.assertIsInstance(result, AIMessage)
        self.assertTrue(len(result.content) > 0)


if __name__ == "__main__":
    unittest.main()
