from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.memory import init_memory_db, load_user_profile, save_agent_state
from agent.skills import load_skill
from agent.state import AgentState
from agent.tools import build_default_tools
from rag import KeywordRetriever, KnowledgeChunk


class AgentModuleTests(unittest.TestCase):
    def test_skill_loader_falls_back(self) -> None:
        skill = load_skill("不存在的岗位")
        self.assertEqual(skill.role, "不存在的岗位")
        self.assertTrue(skill.rubric)

    def test_agent_state_records_question_and_score_tool(self) -> None:
        state = AgentState(role="AI Agent 开发", difficulty="基础", training_mode="面试模式", question_source="混合模式")
        state.record_question("什么是 Tool Use？")
        self.assertEqual(state.current_question, "什么是 Tool Use？")
        self.assertEqual(state.asked_questions, ["什么是 Tool Use？"])

    def test_tools_return_structured_observation(self) -> None:
        question_bank = {
            "AI Agent 开发": [
                {
                    "question": "什么是 Tool Use？",
                    "difficulty": "基础",
                    "type": "概念题",
                    "tags": ["Tool Use"],
                    "expected_points": ["模型调用外部工具完成任务"],
                }
            ]
        }
        retriever = KeywordRetriever(
            [KnowledgeChunk(title="Tool Use", content="工具调用让模型获得外部能力。", role="AI Agent 开发", tags=["Tool Use"])]
        )
        tools = build_default_tools(question_bank, retriever)
        result = tools.call("search_question_bank", role="AI Agent 开发", difficulty="基础", tags=["Tool Use"])
        self.assertEqual(result.tool_name, "search_question_bank")
        self.assertTrue(result.observation["questions"])
        score = tools.call(
            "score_answer",
            question="什么是 Tool Use？",
            answer="模型可以调用外部工具完成任务。",
            expected_points=["模型调用外部工具完成任务"],
            skill_rubric={},
        )
        self.assertTrue({"accuracy", "completeness", "clarity"} <= set(score.observation))

    def test_rag_handles_empty_and_matching_chunks(self) -> None:
        self.assertEqual(KeywordRetriever([]).search("Agent"), [])
        retriever = KeywordRetriever([KnowledgeChunk(title="RAG", content="检索增强生成减少幻觉。", tags=["RAG"])])
        self.assertTrue(retriever.search("RAG 检索"))

    def test_memory_tables_store_state(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        init_memory_db(connection)
        state = AgentState(role="AI Agent 开发", difficulty="基础", training_mode="面试模式", question_source="混合模式")
        save_agent_state(connection, 1, state)
        profile = load_user_profile(connection, "client-1")
        self.assertEqual(profile["client_id"], "client-1")


if __name__ == "__main__":
    unittest.main()
