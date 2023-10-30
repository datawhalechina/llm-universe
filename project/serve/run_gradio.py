# 导入必要的库
import os                # 用于操作系统相关的操作，例如读取环境变量
import io                # 用于处理流式数据（例如文件流）
import IPython.display   # 用于在 IPython 环境中显示数据，例如图片
import requests          # 用于进行 HTTP 请求，例如 GET 和 POST 请求
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import gradio as gr


# 设置请求的默认超时时间为60秒
requests.adapters.DEFAULT_TIMEOUT = 60
# 导入 dotenv 库的函数
# dotenv 允许您从 .env 文件中读取环境变量
# 这在开发时特别有用，可以避免将敏感信息（如API密钥）硬编码到代码中
from dotenv import load_dotenv, find_dotenv
from llm.call_llm import get_completion
from database import create_db
from qa_chain.Chat_QA_chain_self import Chat_QA_chain_self
from qa_chain.QA_chain_self import QA_chain_self

# 寻找 .env 文件并加载它的内容
# 这允许您使用 os.environ 来读取在 .env 文件中设置的环境变量
_ = load_dotenv(find_dotenv())
llm_model_list = ['zhipuai', 'chatgpt', 'wenxin']
init_llm = "zhipuai"
init_embedding_model = "zhipuai"

block = gr.Blocks()

# 定义一个函数，用于格式化聊天 prompt。
def format_chat_prompt(message, chat_history):
    """
    该函数用于格式化聊天 prompt。

    参数:
    message: 当前的用户消息。
    chat_history: 聊天历史记录。

    返回:
    prompt: 格式化后的 prompt。
    """
    # 初始化一个空字符串，用于存放格式化后的聊天 prompt。
    prompt = ""
    # 遍历聊天历史记录。
    for turn in chat_history:
        # 从聊天记录中提取用户和机器人的消息。
        user_message, bot_message = turn
        # 更新 prompt，加入用户和机器人的消息。
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    # 将当前的用户消息也加入到 prompt中，并预留一个位置给机器人的回复。
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    # 返回格式化后的 prompt。
    return prompt

# 定义一个函数，用于生成机器人的回复。
def respond(message, chat_history, llm, history_len=3, temperature=0.1, max_tokens=2048):
    """
    该函数用于生成机器人的回复。

    参数:
    message: 当前的用户消息。
    chat_history: 聊天历史记录。

    返回:
    "": 空字符串表示没有内容需要显示在界面上，可以替换为真正的机器人回复。
    chat_history: 更新后的聊天历史记录
    """
        # 限制 history 的记忆长度
    chat_history = chat_history[-history_len:] if history_len > 0 else []
    # 调用上面的函数，将用户的消息和聊天历史记录格式化为一个 prompt。
    formatted_prompt = format_chat_prompt(message, chat_history)
    # 使用llm对象的predict方法生成机器人的回复（注意：llm对象在此代码中并未定义）。
    bot_message = get_completion(formatted_prompt, llm, temperature=temperature, max_tokens=max_tokens)
    # 将用户的消息和机器人的回复加入到聊天历史记录中。
    chat_history.append((message, bot_message))
    # 返回一个空字符串和更新后的聊天历史记录（这里的空字符串可以替换为真正的机器人回复，如果需要显示在界面上）。
    return "", chat_history

    # 调用大模型获取回复，支持上述三种模型+gpt
with block as demo:
    gr.Markdown("""<h1><center>Chat Robot</center></h1>
    <center>Local Knowledge Base Q&A with llm</center>
    """)
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=480) 
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="Prompt/问题")

            with gr.Row():
                # 创建提交按钮。
                db_with_his_btn = gr.Button("Chat db with history")
                db_wo_his_btn = gr.Button("Chat db without history")
                llm_btn = gr.Button("Chat with llm")
            with gr.Row():
                # 创建一个清除按钮，用于清除文本框和聊天机器人组件的内容。
                clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

        with gr.Column(scale=1):
            file = gr.File(label='请选择知识库目录',file_count='directory',
                file_types=['.txt', '.md', '.docx', '.pdf'])
            with gr.Row():
                init_db = gr.Button("知识库文件向量化")
            model_argument = gr.Accordion("参数配置", open=False)
            with model_argument:
                temperature = gr.Slider(0,
                        1,
                        value=0.00,
                        step=0.01,
                        label="llm temperature",
                        interactive=True)

                top_k = gr.Slider(1,
                                10,
                                value=3,
                                step=1,
                                label="vector db search top k",
                                interactive=True)

                history_len = gr.Slider(0,
                                        5,
                                        value=3,
                                        step=1,
                                        label="history length",
                                        interactive=True)

            model_select = gr.Accordion("模型选择")
            with model_select:
                llm = gr.Dropdown(
                    llm_model_list,
                    label="large language model",
                    value=init_llm,
                    interactive=True)

                embeddings = gr.Dropdown(llm_model_list,
                                                label="Embedding model",
                                                value=init_embedding_model)

        # 设置初始化向量数据库按钮的点击事件。当点击时，调用 create_db 函数，并传入用户的文件和希望使用的 Embedding 模型。
        init_db.click(
            create_db,
            show_progress=True,
            inputs=[file, embeddings],
            outputs=[],
        )
        
        # 设置按钮的点击事件。当点击时，调用上面定义的 Chat_QA_chain_self 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        # db_with_his_btn.click(Chat_QA_chain_self.answer, inputs=[msg, chatbot,  llm, embeddings, history_len, top_k, temperature], outputs=[msg, chatbot])
        # # 设置按钮的点击事件。当点击时，调用上面定义的 QA_chain_self 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        # db_wo_his_btn.click(QA_chain_self.answer, inputs=[msg, chatbot, llm, embeddings, top_k, temperature], outputs=[msg, chatbot])
        # # 设置按钮的点击事件。当点击时，调用上面定义的 get_completion 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        # llm_btn.click(get_completion, inputs=[msg, chatbot, llm, history_len, top_k, temperature], outputs=[msg, chatbot])

        # # 设置文本框的提交事件（即按下Enter键时）。功能与上面的 llm_btn 按钮点击事件相同。
        # msg.submit(get_completion, inputs=[msg, chatbot,  llm, embeddings, history_len, top_k, temperature], outputs=[msg, chatbot]) 
        # 点击后清空后端存储的聊天记录
        # clear.click(clear_history)
    gr.Markdown("""提醒：<br>
    1. 使用时请先上传自己的知识文件，并且文件中不含某些特殊字符，否则将返回error. <br>
    """)
# threads to consume the request
gr.close_all()
# 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
# demo.launch(share=True, server_port=int(os.environ['PORT1']))
# 直接启动
demo.launch()