from __future__ import annotations

import os
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


load_dotenv()

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
TRAINING_MODES = ["学习模式", "面试模式"]
QUESTION_SOURCES = ["本地题库", "AI 动态生成", "混合模式"]
MODEL_OPTIONS: Dict[str, str] = {
    "DeepSeek V4 Flash（便宜，适合日常练习）": "deepseek-v4-flash",
    "DeepSeek V4 Pro（质量更高，适合演示）": "deepseek-v4-pro",
}
REASONING_EFFORTS = ["high", "max"]

QUESTION_BANK: Dict[str, List[str]] = {
    "Java 后端开发": [
        "请解释 Java 中 HashMap 的扩容机制，以及为什么容量通常是 2 的幂。",
        "Spring Boot 自动配置的核心原理是什么？",
        "MySQL 索引失效常见原因有哪些？请结合实际 SQL 举例。",
        "如何设计一个简单的登录鉴权流程？",
    ],
    "Python 后端开发": [
        "Python 的 GIL 是什么？它对多线程性能有什么影响？",
        "FastAPI 和 Flask 的主要区别是什么？",
        "如何定位一个接口响应变慢的问题？",
        "请说明常见的数据库事务隔离级别。",
    ],
    "前端开发": [
        "React 中 useEffect 的依赖数组有什么作用？",
        "浏览器从输入 URL 到页面展示经历了哪些步骤？",
        "如何优化首屏加载速度？",
        "请解释事件循环、宏任务和微任务。",
    ],
    "算法工程师": [
        "请解释过拟合和欠拟合，以及常见解决方法。",
        "Transformer 中 self-attention 的核心思想是什么？",
        "如何评估一个二分类模型的效果？",
        "请说明梯度下降的基本过程。",
    ],
    "Golang 开发": [
        "Go 中 goroutine 和线程有什么区别？",
        "请解释 channel 的作用，以及无缓冲 channel 和有缓冲 channel 的区别。",
        "Go 的 interface 是什么？空接口一般用在什么场景？",
        "如何定位一个 Go 服务中的 goroutine 泄漏问题？",
    ],
    "AI Agent 开发": [
        "什么是 AI Agent？它和普通聊天机器人的区别是什么？",
        "请解释 Prompt、Memory 和 Tool Use 在 Agent 中分别负责什么。",
        "LangChain 在 Agent 项目中通常解决什么问题？",
        "如果要做一个 AI 面试官 Agent，你会如何设计它的核心流程？",
    ],
}


def read_setting(name: str, default: str = "") -> str:
    """Read config from Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or os.getenv(name, default))


def validate_api_key(api_key: str) -> Optional[str]:
    cleaned_key = api_key.strip()
    if not cleaned_key:
        return "请先配置 DeepSeek API Key。"
    if "your_" in cleaned_key.lower() or "你的" in cleaned_key:
        return "当前 API Key 还是示例占位符，请替换成真实的 DeepSeek API Key。"
    try:
        cleaned_key.encode("ascii")
    except UnicodeEncodeError:
        return "API Key 只能包含英文、数字和符号，请不要填中文说明文字。"
    return None


def build_llm(
    api_key: str,
    model: str,
    base_url: str,
    temperature: float,
    thinking_enabled: bool,
    reasoning_effort: str,
) -> ChatOpenAI:
    if thinking_enabled:
        return ChatOpenAI(
            api_key=api_key,
            model=model,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
            extra_body={"thinking": {"type": "enabled"}},
        )

    return ChatOpenAI(
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=temperature,
        extra_body={"thinking": {"type": "disabled"}},
    )


def default_model_label(configured_model: str) -> str:
    for label, model_name in MODEL_OPTIONS.items():
        if configured_model == model_name:
            return label
    return next(iter(MODEL_OPTIONS))


def build_mode_instruction(training_mode: str) -> str:
    if training_mode == "学习模式":
        return """
当前是学习模式。你的目标是帮助用户补基础，而不是制造压力。
- 提问前可以先用通俗语言解释相关知识点。
- 用户回答后，先指出做得好的地方，再指出缺口。
- 如果用户不会，先讲清楚概念，再给一个更简单的小问题。
- 输出格式：知识点讲解、回答点评、评分、建议补充、下一道练习题。
""".strip()

    return """
当前是面试模式。你的目标是模拟真实技术面试。
- 提问要更接近真实面试官，可以进行追问。
- 用户回答后要直接指出问题，不要过度安慰。
- 输出格式：回答评价、存在问题、评分、改进建议、参考思路、下一题。
""".strip()


def build_source_instruction(question_source: str) -> str:
    if question_source == "本地题库":
        return "题目来源：优先从本地题库中选择或改写问题，保证问题稳定、可控。"
    if question_source == "AI 动态生成":
        return "题目来源：不要依赖本地题库，请根据岗位、难度和历史回答动态生成新问题。"
    return "题目来源：结合本地题库和 AI 动态生成。可以参考题库，但要根据用户回答灵活追问。"


def build_system_prompt(
    role: str,
    difficulty: str,
    training_mode: str,
    question_source: str,
    question_context: str,
) -> str:
    mode_instruction = build_mode_instruction(training_mode)
    source_instruction = build_source_instruction(question_source)

    return f"""
你是一名严格但友好的技术面试官，正在面试候选人的岗位是：{role}。

面试难度：{difficulty}
训练模式：{training_mode}
题目来源：{question_source}

你需要遵守以下规则：
1. 一次只问一个问题，不要连续输出多个问题。
2. 如果用户刚回答了问题，先评价回答，再给出改进建议，最后继续追问一个相关问题。
3. 每次评价用户回答时都必须给出评分，评分包含：准确性 x/10、完整性 x/10、表达清晰度 x/10。
4. 如果用户回答太短，要指出缺少的关键点，并用追问引导用户补充。
5. 语言使用中文，语气像真实技术面试官，不要过度夸奖。
6. 如果用户表示“不会”“不懂”“没学过”，先用通俗语言讲解基础概念，再给一个更简单的追问，帮助用户循序渐进。

{mode_instruction}

{source_instruction}

可参考的本地题库：
{question_context}
""".strip()


def question_context_for(role: str) -> str:
    questions = QUESTION_BANK.get(role, [])
    return "\n".join(f"- {question}" for question in questions)


def ensure_messages(
    role: str,
    difficulty: str,
    training_mode: str,
    question_source: str,
) -> None:
    current_config = (role, difficulty, training_mode, question_source)
    question_context = question_context_for(role)
    system_prompt = build_system_prompt(
        role=role,
        difficulty=difficulty,
        training_mode=training_mode,
        question_source=question_source,
        question_context=question_context,
    )
    if (
        "messages" not in st.session_state
        or st.session_state.get("interview_config") != current_config
    ):
        st.session_state.messages = [SystemMessage(content=system_prompt)]
        st.session_state.interview_config = current_config
    else:
        st.session_state.messages[0] = SystemMessage(content=system_prompt)


def render_chat(messages: List[BaseMessage]) -> None:
    for message in messages:
        if isinstance(message, SystemMessage):
            continue
        role = "assistant" if isinstance(message, AIMessage) else "user"
        with st.chat_message(role):
            st.markdown(str(message.content))


def ask_llm(llm: ChatOpenAI, messages: List[BaseMessage]) -> AIMessage:
    response = llm.invoke(messages)
    return AIMessage(content=str(response.content))


def export_chat_history(messages: List[BaseMessage]) -> str:
    lines = ["# AI 模拟面试官对话记录", ""]
    for message in messages:
        if isinstance(message, SystemMessage):
            continue
        speaker = "面试官" if isinstance(message, AIMessage) else "我"
        lines.append(f"## {speaker}")
        lines.append(str(message.content))
        lines.append("")
    return "\n".join(lines)


def has_user_answers(messages: List[BaseMessage]) -> bool:
    return any(isinstance(message, HumanMessage) for message in messages)


def build_report_prompt(
    role: str,
    difficulty: str,
    training_mode: str,
    question_source: str,
    history_text: str,
) -> List[BaseMessage]:
    system_prompt = """
你是一名专业的技术面试复盘教练。请根据用户本轮模拟面试记录，生成结构化中文总结报告。

要求：
1. 不要继续提新问题，只做总结和复盘。
2. 不要虚构没有出现在记录中的表现。
3. 评分要结合记录，给出简短理由。
4. 建议要具体，可执行，适合用户下一轮练习。
5. 使用 Markdown 输出。

报告格式：
# 面试总结报告

## 基本信息
- 目标岗位：
- 面试难度：
- 训练模式：
- 题目来源：

## 总体评分
- 综合评分：x/100
- 准确性：x/10
- 完整性：x/10
- 表达清晰度：x/10

## 表现亮点
- ...

## 薄弱知识点
- ...

## 高频问题
- ...

## 改进建议
- ...

## 推荐复习路线
- ...

## 下一轮训练建议
- ...
""".strip()
    user_prompt = f"""
目标岗位：{role}
面试难度：{difficulty}
训练模式：{training_mode}
题目来源：{question_source}

以下是本轮面试记录：

{history_text}
""".strip()
    return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]


def generate_interview_report(
    llm: ChatOpenAI,
    role: str,
    difficulty: str,
    training_mode: str,
    question_source: str,
    history_text: str,
) -> str:
    report_messages = build_report_prompt(
        role=role,
        difficulty=difficulty,
        training_mode=training_mode,
        question_source=question_source,
        history_text=history_text,
    )
    return str(llm.invoke(report_messages).content)


def start_interview() -> None:
    st.session_state.interview_started = True
    st.session_state.pop("messages", None)
    st.session_state.pop("interview_config", None)
    st.session_state.pop("interview_report", None)


st.set_page_config(page_title="AI 模拟面试官", page_icon="🎙️", layout="centered")

st.title("AI 模拟面试官")
st.caption("基于 Streamlit、LangChain 和大模型 API 的垂直领域 AI Agent 第一版")

with st.sidebar:
    st.header("面试设置")
    role = st.selectbox("目标岗位", list(QUESTION_BANK.keys()))
    difficulty = st.selectbox("面试难度", ["基础", "中等", "进阶"])
    training_mode = st.selectbox("训练模式", TRAINING_MODES)
    question_source = st.selectbox("题目来源", QUESTION_SOURCES)

    st.divider()
    st.header("模型配置")
    st.caption("在线演示版不会预填 API Key。请使用自己的 DeepSeek API Key 体验，避免误用他人额度。")
    api_key = st.text_input(
        "API Key",
        value="",
        type="password",
        placeholder="请输入你的 DeepSeek API Key",
    )
    configured_model = read_setting("MODEL_NAME", DEFAULT_MODEL)
    model_label = st.selectbox(
        "模型",
        list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.keys()).index(default_model_label(configured_model)),
    )
    model = MODEL_OPTIONS[model_label]
    thinking_enabled = st.toggle("开启思考模式", value=False)
    reasoning_effort = st.selectbox(
        "思考强度",
        REASONING_EFFORTS,
        disabled=not thinking_enabled,
    )
    temperature = st.slider(
        "模型创造性",
        0.0,
        1.0,
        0.3,
        0.1,
        disabled=thinking_enabled,
        help="思考模式开启时，DeepSeek 不支持 temperature，该参数不会生效。",
    )
    base_url = read_setting("BASE_URL", DEFAULT_BASE_URL)
    st.caption(f"Base URL: `{base_url}`")

    st.button("开始面试", type="primary", use_container_width=True, on_click=start_interview)

    if st.button("重新开始面试", use_container_width=True):
        st.session_state.pop("messages", None)
        st.session_state.pop("interview_config", None)
        st.session_state.pop("interview_started", None)
        st.session_state.pop("interview_report", None)
        st.rerun()

ensure_messages(role, difficulty, training_mode, question_source)

if not st.session_state.get("interview_started"):
    st.info("请先在侧边栏填写 DeepSeek API Key，然后点击 `开始面试`。")
    st.markdown(
        """
### 你可以用它做什么
- 选择目标岗位，进行技术面试训练。
- 在学习模式下先补基础，再做练习题。
- 在面试模式下模拟真实追问和评分。
- 切换本地题库、AI 动态生成或混合出题。
- 导出本次面试记录，方便复盘。
""".strip()
    )
    st.stop()

api_key_error = validate_api_key(api_key)
if api_key_error:
    st.warning(f"{api_key_error} 请在侧边栏输入有效的 DeepSeek API Key。")
    st.stop()

llm = build_llm(
    api_key=api_key,
    model=model,
    base_url=base_url,
    temperature=temperature,
    thinking_enabled=thinking_enabled,
    reasoning_effort=reasoning_effort,
)

if len(st.session_state.messages) == 1:
    with st.spinner("面试官正在准备第一题..."):
        if training_mode == "学习模式":
            start_text = "请以学习模式开始，先讲一个入门知识点，再提出一道简单练习题。"
        else:
            start_text = "请以面试模式开始，根据题目来源提出第一道技术问题。"
        first_prompt = HumanMessage(content=start_text)
        st.session_state.messages.append(ask_llm(llm, [*st.session_state.messages, first_prompt]))

render_chat(st.session_state.messages)

answer = st.chat_input("请输入你的回答...")
if answer:
    st.session_state.messages.append(HumanMessage(content=answer))
    st.session_state.pop("interview_report", None)
    with st.chat_message("user"):
        st.markdown(answer)

    with st.chat_message("assistant"):
        with st.spinner("面试官正在评价你的回答..."):
            ai_message = ask_llm(llm, st.session_state.messages)
            st.session_state.messages.append(ai_message)
            st.markdown(str(ai_message.content))

history_text = export_chat_history(st.session_state.get("messages", []))

st.divider()
st.subheader("面试总结报告")
if not has_user_answers(st.session_state.get("messages", [])):
    st.info("完成至少一轮回答后，可以生成面试总结报告。")
elif st.button("生成面试总结报告", type="primary"):
    with st.spinner("正在生成面试总结报告..."):
        st.session_state.interview_report = generate_interview_report(
            llm=llm,
            role=role,
            difficulty=difficulty,
            training_mode=training_mode,
            question_source=question_source,
            history_text=history_text,
        )

if st.session_state.get("interview_report"):
    st.markdown(st.session_state.interview_report)
    st.download_button(
        "下载面试总结报告",
        data=st.session_state.interview_report,
        file_name="interview-report.md",
        mime="text/markdown",
    )

with st.sidebar:
    st.divider()
    st.header("对话记录")
    st.caption("当前对话保存在浏览器会话里，刷新或重新开始后可能清空。")

    with st.expander("打开/收起当前面试记录"):
        st.text_area(
            "当前记录内容",
            value=history_text,
            height=260,
            label_visibility="collapsed",
        )

    st.download_button(
        "导出本次面试记录",
        data=history_text,
        file_name="interview-history.md",
        mime="text/markdown",
        use_container_width=True,
    )
