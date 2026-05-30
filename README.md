# AI 模拟面试官 Agent

一个基于 Streamlit、LangChain 和大模型 API 的垂直领域 AI Agent 第一版项目。

## 功能

- 支持选择目标岗位：Java 后端、Python 后端、Golang、前端、算法工程师、AI Agent
- 支持学习模式和面试模式
- 支持本地题库、AI 动态生成、混合模式三种题目来源
- 支持 DeepSeek Flash / Pro 模型切换
- 支持 DeepSeek 思考模式开关和思考强度选择
- AI 一次提出一个技术面试问题
- 根据用户回答给出评价、问题、评分、改进建议和参考思路
- 支持多轮上下文对话
- 支持查看和导出当前面试记录
- 支持生成结构化面试总结报告，包含总体评分、薄弱知识点和复习建议

## 技术栈

- Python
- Streamlit
- LangChain
- DeepSeek API 或其他 OpenAI 兼容 API

## 本地运行

1. 创建虚拟环境

```bash
python -m venv .venv
```

2. 激活虚拟环境

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

3. 安装依赖

```bash
pip install -r requirements.txt
```

4. 配置环境变量

复制 `.env.example` 为 `.env`，然后填入你的 API Key：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
MODEL_NAME=deepseek-v4-flash
BASE_URL=https://api.deepseek.com
```

5. 启动项目

```bash
streamlit run app.py
```

## 部署到 Streamlit Community Cloud

1. 将本项目上传到 GitHub。
2. 打开 Streamlit Community Cloud 并选择该仓库。
3. 入口文件选择 `app.py`。
4. 在 Secrets 中添加：

```toml
DEEPSEEK_API_KEY = "your_deepseek_api_key_here"
MODEL_NAME = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
```

5. 点击 Deploy，等待生成在线访问链接。

## 后续可扩展方向

- 加入 Redis，实现跨浏览器的持久化 Session 记忆
- 增加简历解析功能，根据用户简历生成针对性问题
- 接入搜索工具，让 Agent 查询知识点或岗位要求
- 接入 RAG 面经知识库，让 Agent 从面经和岗位 JD 中检索问题
- 使用 Docker 和 Zeabur 部署

## 简历描述示例

基于 Python、LangChain 与 Streamlit 独立开发 AI 模拟面试官系统，支持岗位选择、学习/面试双模式、本地题库与 LLM 动态出题、DeepSeek Flash/Pro 模型切换、思考模式控制、多轮上下文记忆、回答质量评分、面试记录导出和结构化总结报告生成，并部署至 Streamlit Cloud 提供在线访问。
