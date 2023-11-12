# 个人知识库助手项目

## 项目简介

本项目是基于本大模型应用开发教程内容，进行一定的封装完善得到的个人知识库助手项目，核心是针对四种大模型 API 实现了底层封装，基于 Langchain 搭建了可切换模型的检索问答链，并实现 API 以及 Gradio 部署的个人轻量大模型应用。

本项目的核心开发内容都已在教程内容中讲解，包括项目大体架构、如何调用 LLM API、基于个人知识库的向量数据库搭建、基于自定义 LLM 的检索问答链搭建以及前后端部署。在教程的每一章内容中，都会深入浅出讲解每一部分的原理、技术细节，并结合本项目内容进行一定的延展讲解。

我们欢迎每一位学习者进行教程学习的过程中或完成教程学习之后，来仔细阅读项目源码，并结合项目源码和学习内容搭建一个属于自己的、具有个人风格的大模型应用。

项目开发组成员精力、时间有限，难免有所错漏，如发现问题，欢迎提 Issue 或直接联系项目负责人。

## 项目架构

### 整体架构

本项目为一个基于大模型的个人知识库助手，基于 LangChain 框架搭建，核心技术包括 LLM API 调用、向量数据库、检索问答链等。项目整体架构如下：

![](../figures/structure.jpg)

如上，本项目从底向上依次分为 LLM 层、数据层、数据库层、应用层与服务层。

① LLM 层主要基于四种流行 LLM API 进行了 LLM 调用封装，支持用户以统一的入口、方式来访问不同的模型，支持随时进行模型的切换；

② 数据层主要包括个人知识库的源数据以及 Embedding API，源数据经过 Embedding 处理可以被向量数据库使用；

③ 数据库层主要为基于个人知识库源数据搭建的向量数据库，在本项目中我们选择了 Chroma；

④ 应用层为核心功能的最顶层封装，我们基于 LangChain 提供的检索问答链基类进行了进一步封装，从而支持不同模型切换以及便捷实现基于数据库的检索问答；

⑤ 最顶层为服务层，我们分别实现了 Gradio 搭建 Demo 与 FastAPI 组建 API 两种方式来支持本项目的服务访问。

### 代码结构

本项目的完整代码存放于 project 目录下，实现了项目的全部功能及封装，代码结构如下：

    -project
        -readme.md 项目说明
        -llm LLM调用封装
            -self_llm.py 自定义 LLM 基类
            -wenxin_llm.py 自定义百度文心 LLM
            -spark_llm.py 自定义讯飞星火 LLM
            -zhipuai_llm.py 自定义智谱AI LLM
            -call_llm.py 将各个 LLM 的原生接口封装在一起
        -embedding embedding调用封装
            -zhipuai_embedding.py 自定义智谱AI embedding
        -data 源数据路径
        -database 数据库层封装
            -create_db.py 处理源数据及初始化数据库封装
        -chain 应用层封装
            -qa_chain.py 封装检索问答链，返回一个检索问答链对象
            -chat_qa_chian.py：封装对话检索链，返回一个对话检索链对象
            -prompt_template.py 存放多个版本的 Template
        -serve 服务层封装
            -run_gradio.py 启动 Gradio 界面
            -api.py 封装 FastAPI
            -run_api.sh 启动 API

### 项目逻辑

1. 用户：可以通过 run_gradio 或者 run_api 启动整个服务；
2. 服务层调用 qa_chain.py 或 chat_qa_chain 实例化对话检索链对象，实现全部核心功能；
3. 服务层和应用层都可以调用、切换 prompt_template.py 中的 prompt 模板来实现 prompt 的迭代；
4. 也可以直接调用 call_llm 中的 get_completion 函数来实现不使用数据库的 LLM；
5. 应用层调用已存在的数据库和 llm 中的自定义 LLM 来构建检索链；
6. 如果数据库不存在，应用层调用 create_db.py 创建数据库，该脚本可以使用 openai embedding 也可以使用 embedding.py 中的自定义 embedding

### 各层简析

### 1. LLM 层

LLM 层主要功能为将国内外四种知名 LLM API（OpenAI-ChatGPT、百度文心、讯飞星火、智谱GLM）进行封装，隐藏不同 API 的调用差异，实现在同一个对象或函数中通过不同的 model 参数来使用不同来源的 LLM。

在 LLM 层，我们首先构建了一个 Self_LLM 基类，基类定义了所有 API 的一些共同参数（如 API_Key，temperature 等）；然后我们在该基类基础上继承实现了上述四种 LLM API 的自定义 LLM。同时，我们也将四种 LLM 的原生 API 封装在了统一的 get_completion 函数中。

想要学习每一种 LLM 的调用方式、封装方式，请阅读教程第二章《调用大模型 API》。

### 4.2 数据层

数据层主要包括个人知识库的源数据（包括 pdf、txt、md 等）和 Embedding 对象。源数据需要经过 Embedding 处理才能进入向量数据库，我们在数据层自定义了智谱提供的 Embedding API 的封装，支持上层以统一方式调用智谱 Embedding 或 OpenAI Embedding。


### 4.3 数据库层

数据库层主要存放了向量数据库文件。同时，我们在该层实现了源数据处理、创建向量数据库的方法。

想要学习向量数据库的原理、源数据处理方法以及构建向量数据库的具体实现，请阅读教程第四章《数据库搭建》。

### 4.4 应用层

应用层封装了整个项目的全部核心功能。我们基于 LangChain 提供的检索问答链，在 LLM 层、数据库层的基础上，实现了本项目检索问答链的封装。自定义的检索问答链除具备基本的检索问答功能外，也支持通过 model 参数来灵活切换使用的 LLM。我们实现了两个检索问答链，分别是有历史记录的 Chat_QA_Chain 和没有历史记录的 QA_Chain。

想要学习 Prompt 的构造与检索问答链的构建细节，请阅读教程第五章《Prompt 设计》。

### 4.5 服务层

服务层主要是基于应用层的核心功能封装，实现了 Demo 的搭建或 API 的封装。在本项目中，我们分别实现了通过 Gradio 搭建前端界面与 FastAPI 进行封装，支持多样化的项目调用。

想要学习如何使用 Gradio 以及 FastAPI 来实现服务层的设计，请阅读教程第七章《前后端搭建》。