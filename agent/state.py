from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class ScoreRecord:
    question: str
    answer: str
    accuracy: int
    completeness: int
    clarity: int
    missing_points: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=now_text)

    def average(self) -> float:
        return round((self.accuracy + self.completeness + self.clarity) / 3, 2)


@dataclass
class ToolCallRecord:
    tool_name: str
    input: Dict[str, Any]
    observation: Dict[str, Any]
    confidence: float
    created_at: str = field(default_factory=now_text)


@dataclass
class AgentState:
    role: str
    difficulty: str
    training_mode: str
    question_source: str
    current_question: str = ""
    asked_questions: List[str] = field(default_factory=list)
    scores: List[ScoreRecord] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    next_action: str = "ask_new_question"
    tool_calls: List[ToolCallRecord] = field(default_factory=list)

    def record_question(self, question: str) -> None:
        cleaned = question.strip()
        if not cleaned:
            return
        self.current_question = cleaned
        if cleaned not in self.asked_questions:
            self.asked_questions.append(cleaned)

    def record_score(self, score: ScoreRecord) -> None:
        self.scores.append(score)
        for weakness in score.missing_points:
            if weakness and weakness not in self.weaknesses:
                self.weaknesses.append(weakness)
        for strength in score.strengths:
            if strength and strength not in self.strengths:
                self.strengths.append(strength)

    def record_tool_call(self, tool_result: "ToolResultLike") -> None:
        self.tool_calls.append(
            ToolCallRecord(
                tool_name=tool_result.tool_name,
                input=dict(tool_result.input),
                observation=dict(tool_result.observation),
                confidence=float(tool_result.confidence),
                created_at=tool_result.created_at,
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        state = cls(
            role=data.get("role", ""),
            difficulty=data.get("difficulty", ""),
            training_mode=data.get("training_mode", ""),
            question_source=data.get("question_source", ""),
            current_question=data.get("current_question", ""),
            asked_questions=list(data.get("asked_questions", [])),
            weaknesses=list(data.get("weaknesses", [])),
            strengths=list(data.get("strengths", [])),
            next_action=data.get("next_action", "ask_new_question"),
        )
        state.scores = [ScoreRecord(**item) for item in data.get("scores", [])]
        state.tool_calls = [ToolCallRecord(**item) for item in data.get("tool_calls", [])]
        return state


class ToolResultLike:
    tool_name: str
    input: Dict[str, Any]
    observation: Dict[str, Any]
    confidence: float
    created_at: str
