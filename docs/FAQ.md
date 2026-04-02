# 常见问题与快速排障指南 (FAQ & Troubleshooting)

欢迎来到 LLM-Universe 项目！作为面向小白的教程，我们整理了学习过程中最常见的“坑”和解决办法。**如果你遇到了问题，先来这里找答案！**

---

##  环境配置相关

### 1. Python 版本不匹配怎么办？

**问题现象**：运行代码时提示 `Python version not supported` 或某些库报错。  
**可能原因**：项目需要 Python 3.8-3.10，而你的版本过高或过低。  
**解决方案**：
- 使用 Conda 创建指定版本的虚拟环境：
  ```bash
  conda create -n llm python=3.9
  conda activate llm
  ```
- 或使用 `pyenv` 管理多版本 Python。

### 2. 依赖安装失败（pip install 报错）

**问题现象**：`pip install -r requirements.txt` 时卡住、超时或报红字错误。  
**可能原因**：默认的 pip 源在国外，下载慢或被墙；或者某个包版本冲突。  
**解决方案**：
- 换国内镜像源，例如清华源：
  ```bash
  pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```
- 如果还是报错，可以尝试逐个安装，比如先安装 `langchain`，看具体错误信息。

### 3. 虚拟环境配置问题

**问题现象**：激活虚拟环境后，`pip list` 仍显示全局包。  
**可能原因**：没有正确激活，或者环境变量未生效。  
**解决方案**：
- 确保在项目目录下运行 `conda activate llm`（或 `source activate llm`）。
- 在 VS Code 中，按 `Ctrl+Shift+P`，选择 `Python: Select Interpreter`，然后选你刚创建的环境。

### 4. requirements.txt 中的包版本冲突

**问题现象**：安装时报 `ERROR: pip's dependency resolver does not currently take into account...`  
**可能原因**：某些包之间存在版本冲突。  
**解决方案**：
- 先创建新的虚拟环境，然后尝试：
  ```bash
  pip install -r requirements.txt --no-deps
  ```
  再手动安装缺失的依赖。

---

##  API 配置相关

### 1. API Key 在哪里获取？

- **智谱 AI**：访问 [智谱AI开放平台](https://open.bigmodel.cn/)，注册后获取 API Key。
- **OpenAI**：访问 [OpenAI平台](https://platform.openai.com/)，申请 API Key（需要海外支付方式）。
- **百度文心**：访问 [百度智能云千帆平台](https://console.bce.baidu.com/qianfan/overview) 申请。
- **讯飞星火**：访问 [讯飞开放平台](https://www.xfyun.cn/) 申请。

### 2. API Key 配置后仍然报错（401 / Invalid key）

**问题现象**：运行代码时提示 `401 Unauthorized` 或 `Invalid API key`。  
**可能原因**：
- `.env` 文件未放在项目根目录。
- 变量名写错（例如应该是 `ZHIPUAI_API_KEY` 而不是 `API_KEY`）。
- Key 本身无效或已过期。

**解决方案**：
- 在项目根目录（与 `README.md` 同级）创建 `.env` 文件，内容格式：
  ```
  ZHIPUAI_API_KEY=你的真实key
  ```
- 确认变量名与教程中一致。如果使用 OpenAI，则变量名是 `OPENAI_API_KEY`。
- 检查 key 是否有多余空格，或重新生成一个。

### 3. API 额度用完/欠费问题

**问题现象**：调用时返回 `429 Too Many Requests` 或 `Quota exceeded`。  
**解决方案**：
- 查看对应平台的余额，如果是免费额度用完了，可以申请新的 key，或者换用其他大模型。
- 也可以暂时用本地模型（如通过 `llama.cpp`）替代。

---

##  数据库相关

### 1. 向量数据库初始化失败（Chroma / FAISS）

**问题现象**：运行代码时提示 `Chroma` 或 `FAISS` 相关错误。  
**可能原因**：版本不兼容或未正确安装。  
**解决方案**：
- 尝试固定版本安装：
  ```bash
  pip install chromadb==0.4.22
  ```
- 如果使用 FAISS，确保已安装 `faiss-cpu`：
  ```bash
  pip install faiss-cpu
  ```

### 2. 知识库文件读取错误

**问题现象**：加载 PDF 或 Markdown 时报错，例如 `FileNotFoundError` 或 `UnicodeDecodeError`。  
**可能原因**：文件路径不对，或编码问题。  
**解决方案**：
- 检查文件路径是否相对于项目根目录。
- 确保文档是 UTF-8 编码，可以在 Python 中用 `encoding='utf-8'` 打开。

---

##  运行相关

### 1. Gradio / Streamlit 启动失败

**问题现象**：运行 `streamlit run app.py` 后浏览器打不开或报错。  
**可能原因**：端口被占用，或依赖未装全。  
**解决方案**：
- 换一个端口：
  ```bash
  streamlit run app.py --server.port 8502
  ```
- 确保已安装 `streamlit`：
  ```bash
  pip install streamlit
  ```

### 2. Notebook 无法运行（Jupyter kernel 崩溃）

**问题现象**：Notebook 打开后内核一直重启，或运行单元格时自动停止。  
**可能原因**：内存不足，或某个包有冲突。  
**解决方案**：
- 重启内核，只运行一个单元格试试。
- 如果持续崩溃，可以尝试在 Colab 上运行（上传 notebook 到 Google Drive 后用 Colab 打开）。

---

##  其他常见错误

### 1. ModuleNotFoundError / ImportError

**问题现象**：`ModuleNotFoundError: No module named 'xxx'`。  
**解决方案**：
- 缺什么包就安装什么，例如：
  ```bash
  pip install langchain
  ```
- 如果是在 notebook 里，可以在单元格开头加 `!pip install xxx`。

### 2. 中文乱码

**问题现象**：输出的中文显示为乱码。  
**解决方案**：
- 在 Python 文件开头添加：
  ```python
  # -*- coding: utf-8 -*-
  ```
- 或设置系统默认编码（仅限 Windows）：
  ```python
  import sys
  import io
  sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
  ```

### 3. GitHub 访问慢或打不开

**问题现象**：克隆仓库时极慢，或网页打不开。  
**解决方案**：
- 使用 Gitee 镜像：将 GitHub 地址中的 `github.com` 替换为 `gitee.com`（如 `https://gitee.com/datawhalechina/llm-universe`）。
- 使用 Steam++（Watt Toolkit）或 FastGithub 加速工具。

---

如果以上没有解决你的问题，欢迎在 [Issues](https://github.com/datawhalechina/llm-universe/issues) 提问，我们会尽力帮助你！
```
