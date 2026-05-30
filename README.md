# AI 模拟面试官 Agent

一个基于 Streamlit、LangChain 和 DeepSeek API 的垂直领域 AI Agent 应用，用于模拟技术面试、辅助学习复盘，并生成结构化面试总结报告。

- 在线体验：https://ai-interview-agent-8dywxqicugfvm25wglebn6.streamlit.app/
- GitHub：https://github.com/2192635878/ai-interview-agent

> 在线演示版不会内置 API Key。使用时需要在侧边栏输入自己的 DeepSeek API Key，避免公开链接消耗个人额度。

## 项目截图

建议放 3 张截图，保存到 `docs/images/` 目录：

1. `start-page.png`：开始页，展示岗位、模式、模型配置和开始面试按钮。
2. `interview-chat.png`：面试对话页，展示 AI 提问、用户回答、评分和追问。
3. `summary-report.png`：总结报告页，展示总体评分、薄弱知识点和复习建议。

截图放好后，取消下面三行的注释即可展示：


![开始页](docs/images/start-page.png)
![面试对话](docs/images/interview-chat.png)
![总结报告](docs/images/summary-report.png)


## 核心功能

- 支持 Java 后端、Python 后端、Golang、前端、算法工程师、AI Agent 等岗位方向。
- 支持学习模式和面试模式，分别面向知识补齐和真实面试训练。
- 支持结构化本地面试题库、AI 动态生成、混合模式三种题目来源。
- 支持上传 PDF/TXT/MD 简历，AI 可结合项目经历和技能栈生成针对性问题。
- 支持 DeepSeek Flash / Pro 模型切换。
- 支持 DeepSeek 思考模式开关和思考强度选择。
- 支持多轮上下文对话，AI 可根据历史回答继续追问。
- 支持回答质量评分，包括准确性、完整性、表达清晰度。
- 支持查看和导出当前面试记录。
- 支持生成结构化面试总结报告，包含总体评分、薄弱知识点、改进建议和复习路线。

## 技术亮点

- 使用 LangChain 的 `ChatOpenAI` 接入 DeepSeek OpenAI 兼容接口，统一管理模型调用。
- 通过 Prompt Engineering 约束 AI 面试官的角色、提问方式、评分格式和总结报告结构。
- 基于 `st.session_state` 实现单次会话内的多轮上下文记忆。
- 将题库从代码中解耦为 `data/question_bank.json`，按岗位、难度、题型、标签和参考要点组织问题。
- 将题目来源抽象为本地题库、LLM 动态生成和混合模式，提升出题灵活性。
- 使用 `pypdf` 解析用户上传的 PDF 简历，并将简历摘要作为上下文注入面试 Prompt。
- 对在线演示版的 API Key 做输入校验，避免占位符或中文内容触发请求错误。
- 使用 Streamlit Community Cloud 部署，提供可公开访问的在线演示地址。

## 技术栈

- Python
- Streamlit
- LangChain
- pypdf
- DeepSeek API

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

Windows 用户也可以直接运行：

```powershell
.\run_app.bat
```

## 部署方式

本项目已部署到 Streamlit Community Cloud。

部署步骤：

1. 将项目推送到 GitHub。
2. 在 Streamlit Community Cloud 中选择该仓库。
3. Branch 选择 `main`，入口文件填写 `app.py`。
4. 如需使用服务端 API Key，可在 Secrets 中添加：

```toml
DEEPSEEK_API_KEY = "your_deepseek_api_key_here"
MODEL_NAME = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
```

当前在线演示版默认要求用户自行输入 API Key。

## 后续规划

- 增加固定面试轮次和“继续下一题”按钮，让流程更结构化。
- 接入 RAG 面经知识库，从面经、岗位 JD 和学习笔记中检索问题。
- 使用 SQLite 或 Redis 保存历史面试记录。
- 增加 Dockerfile，支持 Zeabur 等平台容器化部署。

## 简历描述示例

基于 Python、LangChain 与 Streamlit 独立开发 AI 模拟面试官系统，支持岗位选择、学习/面试双模式、本地题库与 LLM 动态出题、DeepSeek Flash/Pro 模型切换、思考模式控制、多轮上下文记忆、回答质量评分、面试记录导出和结构化总结报告生成，并部署至 Streamlit Community Cloud 提供在线访问。
