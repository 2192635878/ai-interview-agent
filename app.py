from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pypdf import PdfReader

import agent.core as agent_core
import agent.memory as agent_memory
from agent.skills import load_skill
from agent.state import AgentState
from agent.tools import build_default_tools
from rag import KeywordRetriever, load_knowledge_chunks


load_dotenv()

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
QUESTION_BANK_PATH = Path(__file__).parent / "data" / "question_bank.json"
HISTORY_DB_PATH = Path(__file__).parent / "data" / "interview_history.db"
TRAINING_MODES = ["学习模式", "面试模式"]
QUESTION_SOURCES = ["本地题库", "AI 动态生成", "混合模式"]
MODEL_OPTIONS: Dict[str, str] = {
    "DeepSeek V4 Flash（便宜，适合日常练习）": "deepseek-v4-flash",
    "DeepSeek V4 Pro（质量更高，适合演示）": "deepseek-v4-pro",
}
REASONING_EFFORTS = ["high", "max"]
MAX_RESUME_CONTEXT_CHARS = 4000
INTERVIEW_ROUND_OPTIONS = ["不限", "3", "5", "8"]


def read_setting(name: str, default: str = "") -> str:
    """Read config from Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or os.getenv(name, default))


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_client_id() -> str:
    query_client_id = st.query_params.get("client_id", "")
    if query_client_id:
        st.session_state.client_id = query_client_id
        return query_client_id

    if "client_id" not in st.session_state:
        st.session_state.client_id = uuid.uuid4().hex
        st.query_params["client_id"] = st.session_state.client_id

    return str(st.session_state.client_id)


def connect_history_db() -> sqlite3.Connection:
    connection = sqlite3.connect(HISTORY_DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_history_db() -> None:
    with connect_history_db() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id TEXT,
                title TEXT NOT NULL,
                role TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                training_mode TEXT NOT NULL,
                question_source TEXT NOT NULL,
                report TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(sessions)").fetchall()
        }
        if "client_id" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN client_id TEXT DEFAULT ''")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
            """
        )
        agent_memory.init_memory_db(connection)


def create_history_session(
    client_id: str,
    role: str,
    difficulty: str,
    training_mode: str,
    question_source: str,
) -> int:
    created_at = now_text()
    title = f"{role} - {created_at[5:16]}"
    with connect_history_db() as connection:
        cursor = connection.execute(
            """
            INSERT INTO sessions (client_id, title, role, difficulty, training_mode, question_source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (client_id, title, role, difficulty, training_mode, question_source, created_at, created_at),
        )
        return int(cursor.lastrowid)


def ensure_history_session(
    client_id: str,
    role: str,
    difficulty: str,
    training_mode: str,
    question_source: str,
) -> int:
    session_id = st.session_state.get("current_session_id")
    if session_id:
        return int(session_id)

    session_id = create_history_session(client_id, role, difficulty, training_mode, question_source)
    st.session_state.current_session_id = session_id
    return session_id


def message_role(message: BaseMessage) -> str:
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, HumanMessage):
        return "user"
    return "system"


def save_history_message(session_id: int, message: BaseMessage) -> None:
    role = message_role(message)
    if role == "system":
        return

    with connect_history_db() as connection:
        connection.execute(
            """
            INSERT INTO messages (session_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, role, str(message.content), now_text()),
        )
        connection.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (now_text(), session_id),
        )


def save_history_report(session_id: int, report: str) -> None:
    with connect_history_db() as connection:
        connection.execute(
            "UPDATE sessions SET report = ?, updated_at = ? WHERE id = ?",
            (report, now_text(), session_id),
        )


def list_history_sessions(client_id: str, limit: int = 10) -> List[sqlite3.Row]:
    with connect_history_db() as connection:
        return connection.execute(
            """
            SELECT id, title, role, difficulty, training_mode, question_source, updated_at
            FROM sessions
            WHERE client_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (client_id, limit),
        ).fetchall()


def load_history_session(session_id: int) -> tuple[sqlite3.Row, List[sqlite3.Row]]:
    with connect_history_db() as connection:
        session = connection.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        messages = connection.execute(
            """
            SELECT role, content
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()
    return session, messages


def restore_history_session(session_id: int) -> None:
    session, messages = load_history_session(session_id)
    if session is None:
        return

    st.session_state.selected_role = session["role"]
    st.session_state.selected_difficulty = session["difficulty"]
    st.session_state.selected_training_mode = session["training_mode"]
    st.session_state.selected_question_source = session["question_source"]
    st.session_state.current_session_id = session_id
    st.session_state.interview_started = True
    st.session_state.interview_report = session["report"] or ""
    st.session_state.pending_history_messages = [
        {"role": message["role"], "content": message["content"]} for message in messages
    ]
    with connect_history_db() as connection:
        restored_state = agent_memory.load_agent_state(connection, session_id)
    if restored_state is not None:
        st.session_state.pending_agent_state = restored_state


def rows_to_messages(rows: List[dict]) -> List[BaseMessage]:
    restored_messages: List[BaseMessage] = []
    for row in rows:
        if row["role"] == "assistant":
            restored_messages.append(AIMessage(content=row["content"]))
        elif row["role"] == "user":
            restored_messages.append(HumanMessage(content=row["content"]))
    return restored_messages


@st.cache_data
def load_question_bank() -> Dict[str, List[dict]]:
    with QUESTION_BANK_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_rag_chunks() -> List[dict]:
    return [chunk.__dict__ for chunk in load_knowledge_chunks()]


def build_retriever() -> KeywordRetriever:
    from rag import KnowledgeChunk

    return KeywordRetriever([KnowledgeChunk(**chunk) for chunk in load_rag_chunks()])


def validate_question_bank(question_bank: object) -> tuple[bool, str]:
    if not isinstance(question_bank, dict) or not question_bank:
        return False, "题库必须是非空 JSON 对象。"

    required_fields = {"question", "difficulty", "type", "tags", "expected_points"}
    for role, questions in question_bank.items():
        if not isinstance(role, str) or not role.strip():
            return False, "岗位名称必须是非空字符串。"
        if not isinstance(questions, list) or not questions:
            return False, f"{role} 的题目列表不能为空。"
        for index, question in enumerate(questions, start=1):
            if not isinstance(question, dict):
                return False, f"{role} 第 {index} 道题必须是 JSON 对象。"
            missing_fields = required_fields - set(question)
            if missing_fields:
                return False, f"{role} 第 {index} 道题缺少字段：{', '.join(sorted(missing_fields))}。"
            if not isinstance(question["tags"], list) or not isinstance(question["expected_points"], list):
                return False, f"{role} 第 {index} 道题的 tags 和 expected_points 必须是数组。"
    return True, "题库格式校验通过。"


def question_bank_stats(question_bank: Dict[str, List[dict]]) -> tuple[int, int]:
    role_count = len(question_bank)
    question_count = sum(len(questions) for questions in question_bank.values())
    return role_count, question_count


def question_bank_to_json(question_bank: Dict[str, List[dict]]) -> str:
    return json.dumps(question_bank, ensure_ascii=False, indent=2)


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


def extract_resume_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    raw_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    return raw_text.strip()


def trim_resume_text(resume_text: str) -> str:
    cleaned_text = "\n".join(line.strip() for line in resume_text.splitlines() if line.strip())
    return cleaned_text[:MAX_RESUME_CONTEXT_CHARS]


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


def build_source_instruction(question_source: str, has_resume: bool) -> str:
    resume_rule = ""
    if has_resume:
        resume_rule = "候选人已上传简历，本地题库只作为知识点参考，前 1-2 个问题必须优先围绕简历中的项目经历、技术栈或求职方向展开。"

    if question_source == "本地题库":
        return f"题目来源：优先从本地题库中选择或改写问题，保证问题稳定、可控，但不要机械按题库顺序逐题照问。{resume_rule}"
    if question_source == "AI 动态生成":
        return f"题目来源：不要依赖本地题库，请根据岗位、难度和历史回答动态生成新问题。{resume_rule}"
    return f"题目来源：结合本地题库和 AI 动态生成。可以参考题库，但要根据用户回答灵活追问。{resume_rule}"


def build_system_prompt(
    role: str,
    difficulty: str,
    training_mode: str,
    question_source: str,
    question_context: str,
    resume_context: str,
) -> str:
    mode_instruction = build_mode_instruction(training_mode)
    source_instruction = build_source_instruction(question_source, bool(resume_context.strip()))

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

候选人简历内容（如果为空则忽略）：
{resume_context}

如果候选人上传了简历，你需要优先结合简历中的项目经历、技术栈和求职方向进行提问与追问，避免只问泛泛的概念题。
如果简历中包含项目经历，第一道题应优先从项目经历切入，例如询问项目背景、技术选型、实现细节、难点或可改进点。
""".strip()


def question_context_for(
    question_bank: Dict[str, List[dict]],
    role: str,
    difficulty: str,
) -> str:
    questions = question_bank.get(role, [])
    filtered_questions = [
        question
        for question in questions
        if question.get("difficulty") in {difficulty, "基础"}
    ]
    selected_questions = filtered_questions or questions

    lines = []
    for question in selected_questions:
        expected_points = "；".join(question.get("expected_points", []))
        tags = "、".join(question.get("tags", []))
        lines.append(
            "- "
            f"题目：{question.get('question', '')}\n"
            f"  难度：{question.get('difficulty', '')}；类型：{question.get('type', '')}；标签：{tags}\n"
            f"  参考要点：{expected_points}"
        )
    return "\n".join(lines)


def ensure_messages(
    question_bank: Dict[str, List[dict]],
    role: str,
    difficulty: str,
    training_mode: str,
    question_source: str,
    resume_context: str,
) -> None:
    current_config = (role, difficulty, training_mode, question_source, resume_context)
    previous_config = st.session_state.get("interview_config")
    is_config_changed = previous_config is not None and previous_config != current_config
    is_restoring_history = st.session_state.get("pending_history_messages") is not None
    state = st.session_state.get("agent_state")
    if state is None or previous_config != current_config:
        state = AgentState(
            role=role,
            difficulty=difficulty,
            training_mode=training_mode,
            question_source=question_source,
        )
        st.session_state.agent_state = state
    else:
        state.role = role
        state.difficulty = difficulty
        state.training_mode = training_mode
        state.question_source = question_source

    skill = load_skill(role)
    retriever = build_retriever()
    initial_chunks = retriever.search(
        query=f"{role} {difficulty} {' '.join(skill.focus_areas)}",
        role=role,
        tags=skill.rag_tags,
        top_k=3,
    )
    with connect_history_db() as connection:
        profile = agent_memory.load_user_profile(connection, client_id)
    question_context = agent_core.question_context_for(question_bank, role, difficulty)
    system_prompt = agent_core.build_interview_system_prompt(
        state=state,
        question_context=question_context,
        resume_context=resume_context,
        profile=profile,
        skill=skill,
        retrieved_chunks=initial_chunks,
    )
    st.session_state.agent_skill = skill
    st.session_state.agent_profile = profile
    st.session_state.agent_rag_chunks = initial_chunks
    if (
        "messages" not in st.session_state
        or previous_config != current_config
    ):
        st.session_state.messages = [SystemMessage(content=system_prompt)]
        st.session_state.interview_config = current_config
        if is_config_changed and not is_restoring_history:
            st.session_state.pop("current_session_id", None)
            st.session_state.pop("interview_report", None)
            st.session_state.interview_started = False
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


def count_user_answers(messages: List[BaseMessage]) -> int:
    return sum(isinstance(message, HumanMessage) for message in messages)


def is_round_limit_reached(messages: List[BaseMessage], round_limit: str) -> bool:
    if round_limit == "不限":
        return False
    return count_user_answers(messages) >= int(round_limit)


def build_next_question_prompt(round_limit: str) -> HumanMessage:
    if round_limit == "不限":
        content = "请根据当前岗位、难度、题目来源和已有回答，换一道新的面试题。不要重复当前对话中已经出现过的问题，尤其不要重复最后一道题。直接输出一个不同知识点的新问题。"
    else:
        content = f"请换一道新的面试题。本轮面试目标题数是 {round_limit} 道，请保持问题聚焦。不要重复当前对话中已经出现过的问题，尤其不要重复最后一道题。"
    return HumanMessage(content=content)


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
    state = st.session_state.get("agent_state")
    if state is not None:
        with connect_history_db() as connection:
            profile = agent_memory.load_user_profile(connection, client_id)
        report_messages = agent_core.build_report_prompt(state=state, history_text=history_text, profile=profile)
    else:
        report_messages = build_report_prompt(
            role=role,
            difficulty=difficulty,
            training_mode=training_mode,
            question_source=question_source,
            history_text=history_text,
        )
    return str(llm.invoke(report_messages).content)


def save_or_generate_report(
    llm: ChatOpenAI,
    role: str,
    difficulty: str,
    training_mode: str,
    question_source: str,
    history_text: str,
) -> str:
    report = generate_interview_report(
        llm=llm,
        role=role,
        difficulty=difficulty,
        training_mode=training_mode,
        question_source=question_source,
        history_text=history_text,
    )
    st.session_state.interview_report = report
    if st.session_state.get("current_session_id"):
        save_history_report(int(st.session_state.current_session_id), report)
        state = st.session_state.get("agent_state")
        if state is not None:
            retriever = build_retriever()
            tools = build_default_tools(question_bank, retriever)
            profile_delta = agent_memory.extract_profile_update(state)
            with connect_history_db() as connection:
                review_result = tools.call(
                    "generate_review_plan",
                    profile={
                        "weaknesses": profile_delta["weaknesses"],
                        "strengths": profile_delta["strengths"],
                        "average_score": profile_delta["average_score"],
                    },
                    role=role,
                )
                agent_memory.update_user_profile(
                    connection=connection,
                    client_id=client_id,
                    role=role,
                    average_score=profile_delta["average_score"],
                    weaknesses=profile_delta["weaknesses"],
                    strengths=profile_delta["strengths"],
                    review_plan=review_result.observation.get("review_plan", []),
                )
                agent_memory.save_agent_state(connection, int(st.session_state.current_session_id), state)
    return report


def start_interview() -> None:
    st.session_state.interview_started = True
    st.session_state.pop("messages", None)
    st.session_state.pop("interview_config", None)
    st.session_state.pop("interview_report", None)
    st.session_state.pop("current_session_id", None)
    st.session_state.pop("pending_history_messages", None)
    st.session_state.pop("agent_state", None)
    st.session_state.pop("pending_agent_state", None)


st.set_page_config(page_title="AI 模拟面试官", page_icon="🎙️", layout="centered")

st.title("AI 模拟面试官")
st.caption("基于 Streamlit、LangChain 和大模型 API 的垂直领域 AI Agent 第一版")

default_question_bank = load_question_bank()
question_bank = st.session_state.get("custom_question_bank", default_question_bank)
init_history_db()
client_id = get_client_id()

with st.sidebar:
    st.header("题库管理")
    role_count, question_count = question_bank_stats(question_bank)
    st.caption(f"当前题库：{role_count} 个岗位，{question_count} 道题。")

    question_bank_file = st.file_uploader(
        "上传自定义题库 JSON（可选）",
        type=["json"],
        help="上传后仅在当前浏览器会话生效，不会覆盖项目默认题库。",
    )
    if question_bank_file is not None:
        try:
            uploaded_bank = json.loads(question_bank_file.getvalue().decode("utf-8"))
            is_valid, message = validate_question_bank(uploaded_bank)
            if is_valid:
                uploaded_bank_json = question_bank_to_json(uploaded_bank)
                if st.session_state.get("custom_question_bank_json") != uploaded_bank_json:
                    st.session_state.pop("messages", None)
                    st.session_state.pop("interview_config", None)
                    st.session_state.pop("interview_started", None)
                    st.session_state.pop("interview_report", None)
                    st.session_state.pop("current_session_id", None)
                    st.session_state.pop("pending_history_messages", None)
                st.session_state.custom_question_bank = uploaded_bank
                st.session_state.custom_question_bank_json = uploaded_bank_json
                question_bank = uploaded_bank
                role_count, question_count = question_bank_stats(question_bank)
                st.success(f"{message} 已加载 {role_count} 个岗位、{question_count} 道题。")
            else:
                st.warning(message)
        except Exception as error:
            st.warning(f"题库解析失败：{error}")

    if st.session_state.get("preview_role") not in question_bank:
        st.session_state.preview_role = next(iter(question_bank))
    preview_role = st.selectbox("预览岗位题目", list(question_bank.keys()), key="preview_role")
    with st.expander("查看题库预览"):
        for index, question in enumerate(question_bank.get(preview_role, []), start=1):
            st.markdown(
                f"**{index}. {question.get('question', '')}**\n\n"
                f"- 难度：{question.get('difficulty', '')}\n"
                f"- 类型：{question.get('type', '')}\n"
                f"- 标签：{', '.join(question.get('tags', []))}"
            )

    st.download_button(
        "下载当前题库 JSON",
        data=question_bank_to_json(question_bank),
        file_name="question_bank.json",
        mime="application/json",
        use_container_width=True,
    )

    if st.button("恢复默认题库", use_container_width=True):
        st.session_state.pop("custom_question_bank", None)
        st.session_state.pop("custom_question_bank_json", None)
        st.session_state.pop("messages", None)
        st.session_state.pop("interview_config", None)
        st.session_state.pop("interview_started", None)
        st.session_state.pop("interview_report", None)
        st.session_state.pop("current_session_id", None)
        st.session_state.pop("pending_history_messages", None)
        st.session_state.pop("agent_state", None)
        st.session_state.pop("pending_agent_state", None)
        st.rerun()

    st.divider()
    st.header("面试设置")
    if st.session_state.get("selected_role") not in question_bank:
        st.session_state.selected_role = next(iter(question_bank))
    role = st.selectbox("目标岗位", list(question_bank.keys()), key="selected_role")
    difficulty = st.selectbox("面试难度", ["基础", "中等", "进阶"], key="selected_difficulty")
    training_mode = st.selectbox("训练模式", TRAINING_MODES, key="selected_training_mode")
    question_source = st.selectbox("题目来源", QUESTION_SOURCES, key="selected_question_source")

    st.divider()
    st.header("简历上传")
    resume_file = st.file_uploader(
        "上传简历（可选）",
        type=["pdf", "txt", "md"],
        help="上传后，AI 会结合简历中的项目经历和技能进行针对性提问。",
    )
    resume_context = ""
    if resume_file is not None:
        try:
            resume_context = trim_resume_text(extract_resume_text(resume_file))
            if resume_context:
                st.success(f"已解析简历内容，约 {len(resume_context)} 个字符。")
                with st.expander("预览解析结果"):
                    st.text_area(
                        "简历文本",
                        value=resume_context,
                        height=180,
                        label_visibility="collapsed",
                    )
            else:
                st.warning("没有从简历中解析到有效文本，请尝试上传文本版 PDF、TXT 或 MD 文件。")
        except Exception as error:
            st.warning(f"简历解析失败：{error}")

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

    st.divider()
    st.header("流程控制")
    round_limit = st.selectbox(
        "面试题数",
        INTERVIEW_ROUND_OPTIONS,
        help="选择“不限”时，可手动决定何时结束面试。",
        key="selected_round_limit",
    )

    st.button("开始面试", type="primary", use_container_width=True, on_click=start_interview)

    if st.button("重新开始面试", use_container_width=True):
        st.session_state.pop("messages", None)
        st.session_state.pop("interview_config", None)
        st.session_state.pop("interview_started", None)
        st.session_state.pop("interview_report", None)
        st.session_state.pop("current_session_id", None)
        st.session_state.pop("pending_history_messages", None)
        st.session_state.pop("agent_state", None)
        st.session_state.pop("pending_agent_state", None)
        st.rerun()

    st.divider()
    st.header("历史面试记录")
    st.caption("保存最近 10 次面试，可点击恢复对话。")
    history_sessions = list_history_sessions(client_id)
    if not history_sessions:
        st.caption("暂无历史记录。")
    for history_session in history_sessions:
        label = f"{history_session['title']}｜{history_session['difficulty']}"
        st.button(
            label,
            key=f"history_session_{history_session['id']}",
            use_container_width=True,
            on_click=restore_history_session,
            args=(int(history_session["id"]),),
        )

ensure_messages(question_bank, role, difficulty, training_mode, question_source, resume_context)

if st.session_state.get("pending_history_messages") is not None:
    restored_messages = rows_to_messages(st.session_state.pending_history_messages)
    st.session_state.messages = [st.session_state.messages[0], *restored_messages]
    st.session_state.pop("pending_history_messages", None)

if st.session_state.get("pending_agent_state") is not None:
    st.session_state.agent_state = st.session_state.pending_agent_state
    st.session_state.pop("pending_agent_state", None)

if not st.session_state.get("interview_started"):
    st.info("请先在侧边栏填写 DeepSeek API Key，然后点击 `开始面试`。如果刚切换了岗位、难度或题目来源，需要重新开始一轮新面试。")
    st.markdown(
        """
### 你可以用它做什么
- 选择目标岗位，进行技术面试训练。
- 在学习模式下先补基础，再做练习题。
- 在面试模式下模拟真实追问和评分。
- 切换本地题库、AI 动态生成或混合出题。
- 上传自定义题库 JSON，按自己的题库进行训练。
- 上传简历，让 AI 根据项目经历和技能进行针对性提问。
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
        session_id = ensure_history_session(client_id, role, difficulty, training_mode, question_source)
        if resume_context:
            start_text = "请先基于候选人简历中的项目经历或技术栈开始面试，提出一道有针对性的项目问题。"
        elif training_mode == "学习模式":
            start_text = "请以学习模式开始，先讲一个入门知识点，再提出一道简单练习题。"
        else:
            start_text = "请以面试模式开始，根据题目来源提出第一道技术问题。"
        first_prompt = HumanMessage(content=start_text)
        first_ai_message = ask_llm(llm, [*st.session_state.messages, first_prompt])
        st.session_state.messages.append(first_ai_message)
        if st.session_state.get("agent_state"):
            st.session_state.agent_state.record_question(str(first_ai_message.content))
            with connect_history_db() as connection:
                agent_memory.save_agent_state(connection, session_id, st.session_state.agent_state)
        save_history_message(session_id, first_ai_message)

render_chat(st.session_state.messages)

answered_count = count_user_answers(st.session_state.messages)
round_limit_reached = is_round_limit_reached(st.session_state.messages, round_limit)

st.caption(
    f"当前已回答 {answered_count} 道题"
    + ("，面试题数不限。" if round_limit == "不限" else f"，目标题数 {round_limit} 道。")
)

control_col_1, control_col_2 = st.columns(2)
with control_col_1:
    if st.button("换/继续下一题", use_container_width=True):
        session_id = ensure_history_session(client_id, role, difficulty, training_mode, question_source)
        with st.spinner("面试官正在准备下一题..."):
            retriever = build_retriever()
            skill = st.session_state.get("agent_skill") or load_skill(role)
            state = st.session_state.get("agent_state")
            if state is not None:
                tools = build_default_tools(question_bank, retriever)
                question_result = tools.call(
                    "search_question_bank",
                    role=role,
                    difficulty=difficulty,
                    tags=skill.rag_tags,
                    top_k=3,
                )
                state.record_tool_call(question_result)
            next_ai_message = ask_llm(
                llm,
                [*st.session_state.messages, agent_core.build_next_question_prompt(round_limit)],
            )
            st.session_state.messages.append(next_ai_message)
            if state is not None:
                state.record_question(str(next_ai_message.content))
                with connect_history_db() as connection:
                    agent_memory.save_agent_state(connection, session_id, state)
            save_history_message(session_id, next_ai_message)
            st.rerun()

with control_col_2:
    if st.button("结束面试并生成报告", type="primary", use_container_width=True):
        if has_user_answers(st.session_state.messages):
            with st.spinner("正在生成面试总结报告..."):
                save_or_generate_report(
                    llm=llm,
                    role=role,
                    difficulty=difficulty,
                    training_mode=training_mode,
                    question_source=question_source,
                    history_text=export_chat_history(st.session_state.messages),
                )
            st.rerun()
        else:
            st.warning("请至少完成一轮回答后再生成报告。")

if round_limit_reached:
    st.info("已达到本轮面试题数。你可以结束面试并生成报告，也可以点击“换/继续下一题”加练。")

answer = st.chat_input("请输入你的回答...")
if answer:
    session_id = ensure_history_session(client_id, role, difficulty, training_mode, question_source)
    st.session_state.messages.append(HumanMessage(content=answer))
    save_history_message(session_id, st.session_state.messages[-1])
    st.session_state.pop("interview_report", None)
    save_history_report(session_id, "")
    with st.chat_message("user"):
        st.markdown(answer)

    with st.chat_message("assistant"):
        with st.spinner("面试官正在评价你的回答..."):
            state = st.session_state.get("agent_state")
            if state is None:
                state = AgentState(role=role, difficulty=difficulty, training_mode=training_mode, question_source=question_source)
                st.session_state.agent_state = state
            retriever = build_retriever()
            tools = build_default_tools(question_bank, retriever)
            skill = st.session_state.get("agent_skill") or load_skill(role)
            ai_message = agent_core.run_answer_turn(
                llm=llm,
                messages=st.session_state.messages,
                state=state,
                question_bank=question_bank,
                tools=tools,
                skill=skill,
                round_limit_reached=round_limit_reached,
            )
            st.session_state.messages.append(ai_message)
            with connect_history_db() as connection:
                agent_memory.save_agent_state(connection, session_id, state)
            save_history_message(session_id, ai_message)
            st.markdown(str(ai_message.content))
    st.rerun()

history_text = export_chat_history(st.session_state.get("messages", []))

st.divider()
st.subheader("面试总结报告")
if st.session_state.get("interview_report"):
    st.markdown(st.session_state.interview_report)
    st.download_button(
        "下载面试总结报告",
        data=st.session_state.interview_report,
        file_name="interview-report.md",
        mime="text/markdown",
    )
elif has_user_answers(st.session_state.get("messages", [])):
    st.info("点击上方“结束面试并生成报告”后，会在这里展示总结报告。")
else:
    st.info("完成至少一轮回答后，可以结束面试并生成报告。")

with st.sidebar:
    st.divider()
    st.header("Agent 观测")
    current_state = st.session_state.get("agent_state")
    if current_state is None:
        st.caption("开始面试后会展示 AgentState、长期记忆和工具调用。")
    else:
        state_summary = {
            "current_question": current_state.current_question,
            "asked_count": len(current_state.asked_questions),
            "next_action": current_state.next_action,
            "weaknesses": current_state.weaknesses[:5],
            "strengths": current_state.strengths[:5],
            "tool_calls": len(current_state.tool_calls),
        }
        st.json(state_summary)
        recent_tools = [tool_call.__dict__ for tool_call in current_state.tool_calls[-3:]]
        with st.expander("最近工具调用"):
            st.json(recent_tools)
        profile = st.session_state.get("agent_profile", {})
        with st.expander("长期记忆画像"):
            st.json(profile)
        rag_chunks = st.session_state.get("agent_rag_chunks", [])
        with st.expander("RAG 检索上下文"):
            for chunk in rag_chunks:
                st.markdown(f"**{chunk.get('title', '')}**")
                st.caption(f"source: {chunk.get('source', 'local')} | score: {chunk.get('score', 0)}")
                st.write(chunk.get("content", ""))

    st.divider()
    st.header("对话记录")
    st.caption("当前对话会自动保存到历史面试记录，也可以导出为 Markdown。")

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
