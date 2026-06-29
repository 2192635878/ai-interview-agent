from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


SKILL_DIR = Path(__file__).resolve().parents[1] / "data" / "skills"


@dataclass
class Skill:
    name: str
    role: str
    focus_areas: List[str] = field(default_factory=list)
    rubric: Dict[str, str] = field(default_factory=dict)
    followup_strategies: List[str] = field(default_factory=list)
    rag_tags: List[str] = field(default_factory=list)
    difficulty_rules: Dict[str, str] = field(default_factory=dict)

    def to_prompt_block(self) -> str:
        focus = "、".join(self.focus_areas) or "岗位基础能力"
        rag_tags = "、".join(self.rag_tags) or "通用技术知识"
        rubric = "\n".join(f"- {key}: {value}" for key, value in self.rubric.items())
        followups = "\n".join(f"- {item}" for item in self.followup_strategies)
        rules = "\n".join(f"- {key}: {value}" for key, value in self.difficulty_rules.items())
        return f"""
岗位 Skill：{self.name}
考察重点：{focus}
RAG 标签：{rag_tags}
评分 rubric：
{rubric}
追问策略：
{followups}
难度规则：
{rules}
""".strip()


def _skill_filename(role: str) -> str:
    safe = role.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return f"{safe}.json"


def load_skill(role: str, skill_dir: Path = SKILL_DIR) -> Skill:
    path = skill_dir / _skill_filename(role)
    if not path.exists():
        return Skill(
            name=f"{role} 默认 Skill",
            role=role,
            focus_areas=["基础概念", "项目经验", "问题排查", "表达结构"],
            rubric={
                "accuracy": "回答是否符合核心事实和工程常识。",
                "completeness": "是否覆盖关键点、边界条件和实际场景。",
                "clarity": "是否结构清晰、表达具体、有例子支撑。",
            },
            followup_strategies=["围绕回答缺口追问一个具体场景", "必要时降低难度并解释概念"],
            rag_tags=[role, "面经", "复习"],
            difficulty_rules={
                "基础": "优先确认概念理解。",
                "中等": "要求结合真实场景或项目经验。",
                "进阶": "追问取舍、排查、性能和边界。",
            },
        )

    with path.open("r", encoding="utf-8") as file:
        data: Dict[str, Any] = json.load(file)
    return Skill(
        name=data.get("name", f"{role} Skill"),
        role=data.get("role", role),
        focus_areas=list(data.get("focus_areas", [])),
        rubric=dict(data.get("rubric", {})),
        followup_strategies=list(data.get("followup_strategies", [])),
        rag_tags=list(data.get("rag_tags", [])),
        difficulty_rules=dict(data.get("difficulty_rules", {})),
    )
