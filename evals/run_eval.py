from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.skills import load_skill
from agent.tools import score_answer_tool


SAMPLES_PATH = ROOT / "evals" / "sample_answers.json"
REPORT_PATH = ROOT / "evals" / "eval_report.md"


def main() -> None:
    samples = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
    rows = []
    tool_success = 0
    weak_hit = 0
    total_score = 0.0
    for item in samples:
        skill = load_skill(item["role"])
        result = score_answer_tool(
            question=item["question"],
            answer=item["answer"],
            expected_points=[item.get("expected_weakness", "")],
            skill_rubric=skill.rubric,
        )
        obs = result.observation
        average = round((obs["accuracy"] + obs["completeness"] + obs["clarity"]) / 3, 2)
        total_score += average
        tool_success += int("accuracy" in obs and "missing_points" in obs)
        expected = item.get("expected_weakness", "")
        weak_hit += int(any(expected in point for point in obs.get("missing_points", [])))
        rows.append(
            f"| {item['role']} | {item['difficulty']} | {average} | {', '.join(obs.get('missing_points', [])[:2])} |"
        )

    count = len(samples)
    report = [
        "# AI 面试官 Agent 离线 Eval Report",
        "",
        "该评测不调用真实模型，用确定性评分工具检查 Tool Use、评分 JSON 和弱点识别链路是否可用。",
        "",
        f"- 样例数量：{count}",
        f"- 工具调用成功率：{tool_success / count:.0%}",
        f"- 弱点命中率：{weak_hit / count:.0%}",
        f"- 平均评分：{total_score / count:.2f}/10",
        "",
        "| 岗位 | 难度 | 平均分 | 识别薄弱点 |",
        "| --- | --- | ---: | --- |",
        *rows,
        "",
        "## 结论",
        "",
        "当前 eval 重点验证工程闭环：样例输入 -> Tool Use -> 结构化 observation -> 指标报告。后续可以接入真实 LLM，增加重复提问率、追问合理性和报告可用性指标。",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
