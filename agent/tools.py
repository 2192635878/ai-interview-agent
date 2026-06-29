from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List

from rag import KeywordRetriever


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class ToolResult:
    tool_name: str
    input: Dict[str, Any]
    observation: Dict[str, Any]
    confidence: float = 1.0
    created_at: str = field(default_factory=now_text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "input": self.input,
            "observation": self.observation,
            "confidence": self.confidence,
            "created_at": self.created_at,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., ToolResult]] = {}

    def register(self, name: str, func: Callable[..., ToolResult]) -> None:
        self._tools[name] = func

    def call(self, name: str, **kwargs: Any) -> ToolResult:
        if name not in self._tools:
            return ToolResult(
                tool_name=name,
                input=kwargs,
                observation={"error": f"tool {name} is not registered"},
                confidence=0.0,
            )
        return self._tools[name](**kwargs)

    def names(self) -> List[str]:
        return sorted(self._tools)


def search_question_bank_tool(question_bank: Dict[str, List[dict]], role: str, difficulty: str, tags: Iterable[str] | None = None, top_k: int = 5) -> ToolResult:
    wanted_tags = {tag.lower() for tag in (tags or [])}
    candidates = []
    for question in question_bank.get(role, []):
        difficulty_match = question.get("difficulty") in {difficulty, "基础"}
        tag_overlap = wanted_tags & {tag.lower() for tag in question.get("tags", [])}
        if difficulty_match or tag_overlap:
            candidates.append(
                {
                    "question": question.get("question", ""),
                    "difficulty": question.get("difficulty", ""),
                    "type": question.get("type", ""),
                    "tags": question.get("tags", []),
                    "expected_points": question.get("expected_points", []),
                    "score": len(tag_overlap) + (1 if difficulty_match else 0),
                }
            )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return ToolResult(
        tool_name="search_question_bank",
        input={"role": role, "difficulty": difficulty, "tags": list(tags or []), "top_k": top_k},
        observation={"questions": candidates[:top_k]},
        confidence=0.9 if candidates else 0.2,
    )


def score_answer_tool(question: str, answer: str, expected_points: Iterable[str] | None = None, skill_rubric: Dict[str, str] | None = None) -> ToolResult:
    points = [point for point in (expected_points or []) if point]
    answer_text = answer.strip()
    matched_points = []
    missing_points = []
    for point in points:
        keywords = [word for word in point.replace("、", " ").replace("，", " ").replace("。", " ").split() if len(word) >= 2]
        if any(keyword.lower() in answer_text.lower() for keyword in keywords[:4]):
            matched_points.append(point)
        else:
            missing_points.append(point)

    length = len(answer_text)
    if points:
        coverage = len(matched_points) / max(len(points), 1)
        accuracy = max(3, min(10, round(4 + coverage * 6)))
        completeness = max(3, min(10, round(3 + coverage * 7)))
    else:
        accuracy = 6 if length >= 40 else 4
        completeness = 6 if length >= 80 else 4
    clarity = 8 if any(marker in answer_text for marker in ["首先", "其次", "最后", "第一", "第二", "例如"]) else 6
    if length < 25:
        clarity = min(clarity, 5)
        completeness = min(completeness, 4)

    strengths = []
    if matched_points:
        strengths.append("覆盖了部分参考要点")
    if clarity >= 7:
        strengths.append("表达有一定结构")
    if not strengths:
        strengths.append("能够尝试回答问题")

    observation = {
        "accuracy": accuracy,
        "completeness": completeness,
        "clarity": clarity,
        "matched_points": matched_points,
        "missing_points": missing_points[:5] or ["需要补充更具体的关键点和例子"],
        "strengths": strengths,
        "rubric_used": skill_rubric or {},
    }
    return ToolResult(
        tool_name="score_answer",
        input={"question": question, "answer": answer, "expected_points": list(points)},
        observation=observation,
        confidence=0.72 if points else 0.45,
    )


def retrieve_knowledge_tool(retriever: KeywordRetriever, query: str, role: str = "", tags: Iterable[str] | None = None, top_k: int = 3) -> ToolResult:
    results = retriever.search(query=query, role=role, tags=tags, top_k=top_k)
    return ToolResult(
        tool_name="retrieve_knowledge",
        input={"query": query, "role": role, "tags": list(tags or []), "top_k": top_k},
        observation={"chunks": results},
        confidence=0.85 if results else 0.2,
    )


def update_user_profile_tool(scores: Dict[str, int], weaknesses: Iterable[str], strengths: Iterable[str]) -> ToolResult:
    observation = {
        "scores": dict(scores),
        "weaknesses": [item for item in weaknesses if item],
        "strengths": [item for item in strengths if item],
    }
    return ToolResult(
        tool_name="update_user_profile",
        input=observation,
        observation={"profile_delta": observation},
        confidence=0.8,
    )


def generate_review_plan_tool(profile: Dict[str, Any], role: str) -> ToolResult:
    weaknesses = list(profile.get("weaknesses", []))[:5]
    focus = weaknesses or ["基础概念", "项目表达", "场景题排查"]
    plan = [f"围绕「{item}」复习概念、准备 1 个项目例子，并完成 2 道追问题。" for item in focus]
    return ToolResult(
        tool_name="generate_review_plan",
        input={"profile": profile, "role": role},
        observation={"review_plan": plan},
        confidence=0.75,
    )


def build_default_tools(question_bank: Dict[str, List[dict]], retriever: KeywordRetriever) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("search_question_bank", lambda **kwargs: search_question_bank_tool(question_bank=question_bank, **kwargs))
    registry.register("score_answer", score_answer_tool)
    registry.register("retrieve_knowledge", lambda **kwargs: retrieve_knowledge_tool(retriever=retriever, **kwargs))
    registry.register("update_user_profile", update_user_profile_tool)
    registry.register("generate_review_plan", generate_review_plan_tool)
    return registry
