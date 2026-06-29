from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .skills import Skill
from .state import AgentState, ScoreRecord
from .tools import ToolRegistry


NEXT_ACTIONS = {"ask_followup", "ask_new_question", "explain_concept", "lower_difficulty", "finish_report"}


def question_context_for(question_bank: Dict[str, List[dict]], role: str, difficulty: str) -> str:
    questions = question_bank.get(role, [])
    selected_questions = [
        question for question in questions if question.get("difficulty") in {difficulty, "基础"}
    ] or questions
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


def build_mode_instruction(training_mode: str) -> str:
    if training_mode == "学习模式":
        return """
当前是学习模式。目标是帮助用户补基础。
- 提问前可以先解释一个相关知识点。
- 用户回答后先指出可取之处，再指出缺口。
- 如果用户不会，先讲概念，再给更简单的小问题。
""".strip()
    return """
当前是面试模式。目标是模拟真实技术面试。
- 提问直接、聚焦，可以追问取舍、边界和项目细节。
- 用户回答后要直接指出问题，给出评分和可改进方向。
""".strip()


def build_source_instruction(question_source: str, has_resume: bool) -> str:
    resume_rule = ""
    if has_resume:
        resume_rule = "候选人已上传简历，前 1-2 个问题必须优先围绕简历中的项目经历、技术栈或求职方向展开。"
    if question_source == "本地题库":
        return f"题目来源：优先从工具检索到的本地题库中选择或改写问题。{resume_rule}"
    if question_source == "AI 动态生成":
        return f"题目来源：根据岗位、难度、Skill、RAG 上下文和历史回答动态生成新问题。{resume_rule}"
    return f"题目来源：结合本地题库、RAG 知识和动态生成，避免重复。{resume_rule}"


def build_agent_context(state: AgentState, profile: Dict[str, Any], skill: Skill, retrieved_chunks: List[Dict[str, Any]]) -> str:
    profile_block = json.dumps(profile, ensure_ascii=False, indent=2)
    state_block = json.dumps(state.to_dict(), ensure_ascii=False, indent=2)
    knowledge_lines = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        knowledge_lines.append(
            f"[{index}] 来源：{chunk.get('source', 'local')} / {chunk.get('title', '')}\n{chunk.get('content', '')}"
        )
    knowledge_block = "\n\n".join(knowledge_lines) or "暂无检索结果。"
    return f"""
<agent_state>
{state_block}
</agent_state>

<long_term_memory>
{profile_block}
</long_term_memory>

<skill>
{skill.to_prompt_block()}
</skill>

<retrieved_knowledge>
{knowledge_block}
</retrieved_knowledge>
""".strip()


def build_interview_system_prompt(
    state: AgentState,
    question_context: str,
    resume_context: str,
    profile: Dict[str, Any],
    skill: Skill,
    retrieved_chunks: List[Dict[str, Any]],
) -> str:
    return f"""
你是一名严格但友好的技术面试官。系统采用轻量 Agent 架构：Skill 负责岗位策略，Tool Use 负责题库/评分/知识检索，Memory 负责历史画像，RAG 负责提供领域上下文。

面试岗位：{state.role}
面试难度：{state.difficulty}
训练模式：{state.training_mode}
题目来源：{state.question_source}

规则：
1. 一次只问一个问题。
2. 如果用户刚回答问题，必须先评价，再给评分，再根据 AgentState 决定追问或换题。
3. 评分包含准确性、完整性、表达清晰度，各 0-10 分。
4. 不展示隐藏推理过程，不输出 Thought/Action/Observation 字样。
5. 可引用 RAG 内容，但必须用自己的话表达，不虚构来源没有支持的细节。
6. 如果用户表示不会，先解释概念，再降低难度追问。

{build_mode_instruction(state.training_mode)}

{build_source_instruction(state.question_source, bool(resume_context.strip()))}

可参考题库：
{question_context}

候选人简历：
{resume_context}

{build_agent_context(state, profile, skill, retrieved_chunks)}
""".strip()


def build_next_question_prompt(round_limit: str) -> HumanMessage:
    if round_limit == "不限":
        content = "请根据 AgentState、Skill、RAG 上下文和已问问题，换一道不重复的新面试题。直接输出问题。"
    else:
        content = f"请换一道不重复的新面试题。本轮目标题数是 {round_limit} 道。直接输出问题。"
    return HumanMessage(content=content)


def build_report_prompt(
    state: AgentState,
    history_text: str,
    profile: Dict[str, Any],
) -> List[BaseMessage]:
    structured_scores = json.dumps([score.__dict__ for score in state.scores], ensure_ascii=False, indent=2)
    system_prompt = """
你是一名专业的技术面试复盘教练。请生成 Markdown 报告，并在末尾附加一个结构化评分 JSON 代码块。

要求：
1. 不要继续提新问题。
2. 不要虚构记录中没有出现的表现。
3. 结合 AgentState、评分记录和长期记忆给出具体建议。
4. JSON 代码块字段为 overall_score, accuracy, completeness, clarity, weaknesses, strengths, next_plan。
""".strip()
    user_prompt = f"""
AgentState：
{json.dumps(state.to_dict(), ensure_ascii=False, indent=2)}

结构化评分记录：
{structured_scores}

长期记忆：
{json.dumps(profile, ensure_ascii=False, indent=2)}

面试记录：
{history_text}
""".strip()
    return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]


def select_next_action(score_observation: Dict[str, Any], answer: str, round_limit_reached: bool = False) -> str:
    if round_limit_reached:
        return "finish_report"
    if len(answer.strip()) < 20:
        return "explain_concept"
    average = (
        int(score_observation.get("accuracy", 0))
        + int(score_observation.get("completeness", 0))
        + int(score_observation.get("clarity", 0))
    ) / 3
    if average < 5:
        return "lower_difficulty"
    if score_observation.get("missing_points"):
        return "ask_followup"
    return "ask_new_question"


def _find_latest_question(messages: List[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = str(message.content).strip()
            question_lines = [line.strip() for line in text.splitlines() if "？" in line or "?" in line]
            return question_lines[-1] if question_lines else text[-300:]
    return ""


def _expected_points_for_question(question_bank: Dict[str, List[dict]], role: str, question_text: str) -> List[str]:
    best_match = None
    for question in question_bank.get(role, []):
        title = question.get("question", "")
        if title and (title in question_text or question_text in title):
            best_match = question
            break
    if best_match is None:
        return []
    return list(best_match.get("expected_points", []))


def _extract_question_from_response(text: str) -> str:
    question_lines = [line.strip(" -") for line in text.splitlines() if "？" in line or "?" in line]
    return question_lines[-1] if question_lines else text.strip()[:300]


def run_answer_turn(
    llm: Any,
    messages: List[BaseMessage],
    state: AgentState,
    question_bank: Dict[str, List[dict]],
    tools: ToolRegistry,
    skill: Skill,
    round_limit_reached: bool = False,
) -> AIMessage:
    answer = str(messages[-1].content) if messages else ""
    latest_question = state.current_question or _find_latest_question(messages)
    expected_points = _expected_points_for_question(question_bank, state.role, latest_question)

    score_result = tools.call(
        "score_answer",
        question=latest_question,
        answer=answer,
        expected_points=expected_points,
        skill_rubric=skill.rubric,
    )
    state.record_tool_call(score_result)
    observation = score_result.observation
    state.record_score(
        ScoreRecord(
            question=latest_question,
            answer=answer,
            accuracy=int(observation.get("accuracy", 0)),
            completeness=int(observation.get("completeness", 0)),
            clarity=int(observation.get("clarity", 0)),
            missing_points=list(observation.get("missing_points", [])),
            strengths=list(observation.get("strengths", [])),
        )
    )
    action = select_next_action(observation, answer, round_limit_reached)
    if action not in NEXT_ACTIONS:
        action = "ask_followup"
    state.next_action = action

    profile_result = tools.call(
        "update_user_profile",
        scores={
            "accuracy": int(observation.get("accuracy", 0)),
            "completeness": int(observation.get("completeness", 0)),
            "clarity": int(observation.get("clarity", 0)),
        },
        weaknesses=observation.get("missing_points", []),
        strengths=observation.get("strengths", []),
    )
    state.record_tool_call(profile_result)

    retrieved = tools.call(
        "retrieve_knowledge",
        query=f"{latest_question}\n{answer}\n{';'.join(observation.get('missing_points', []))}",
        role=state.role,
        tags=skill.rag_tags,
        top_k=3,
    )
    state.record_tool_call(retrieved)

    decision_prompt = HumanMessage(
        content=f"""
请基于以下有限步 ReAct observation 继续面试，但不要展示工具调用过程。

当前问题：{latest_question}
用户回答：{answer}
评分 observation：{json.dumps(observation, ensure_ascii=False)}
下一步动作：{action}
RAG 片段：{json.dumps(retrieved.observation.get('chunks', []), ensure_ascii=False)}

输出要求：
- 先给回答评价。
- 给出准确性/完整性/表达清晰度评分。
- 根据 next_action 执行：追问、换题、解释概念、降低难度或建议结束。
- 最多提出一个问题。
""".strip()
    )
    response = llm.invoke([*messages, decision_prompt])
    ai_message = AIMessage(content=str(response.content))
    state.record_question(_extract_question_from_response(ai_message.content))
    return ai_message
