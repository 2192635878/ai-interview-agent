# AI 面试官 Agent：面试讲法与高含金量升级路线

这份文档不是普通 README，而是用于回答面试官追问：

- 这个项目到底是不是 Agent？
- Tool Use、ReAct、RAG、Memory 分别落在哪里？
- 当前版本有什么边界？
- 如果继续做，哪些升级最有技术含量？

核心原则：不要把项目吹成复杂多智能体平台，而是把它讲成一个从 LLM workflow 向轻量 Agent 演进的工程化项目。

## 1. 当前项目够不够面试？

够，但要用正确说法。

当前项目已经不是“只调一个大模型 API 的聊天 demo”，因为它已经具备：

- 面试业务闭环：岗位选择、题库、简历解析、问答、评分、报告、历史记录。
- Agent 状态：用 `AgentState` 显式记录当前题目、已问问题、评分、薄弱点、下一步动作和工具调用。
- Skill：不同岗位有独立 skill 配置，包含考察重点、评分标准、追问策略和 RAG 标签。
- Tool Use：有本地工具注册表，支持题库检索、回答评分、知识检索、画像更新和复习计划生成。
- 有限步 ReAct：每轮回答后执行“评分工具 -> 用户画像工具 -> RAG 工具 -> 下一步动作决策”。
- RAG：从本地知识库检索相关面经、岗位知识和 Agent 概念，再注入 prompt。
- Memory：SQLite 保存短期会话、AgentState 和长期用户画像。
- Eval Harness：离线样例验证工具调用、评分 JSON 和薄弱点识别链路。

面试时建议这样说：

> 这个项目第一版是可控的 LLM workflow，后来我把它升级成轻量 Agent：显式加入 AgentState、Skill、Tool Use、RAG、Memory 和有限步 ReAct。它不是复杂多 Agent 系统，但已经具备 Agent 工程化的关键骨架。

## 2. 当前项目不要怎么说

不要说：

- “我实现了完整 AutoGPT 类 Agent。”
- “模型可以任意自主调用工具。”
- “我做了向量数据库 RAG。”
- “我完整接入了 MCP 生态。”
- “我的 eval 能证明模型效果很好。”

更稳的说法：

- 当前是轻量 Agent，不是无限循环 Agent。
- Tool Use 是本地 Python 工具编排，不是模型原生 function calling。
- RAG 是关键词检索版本，不是向量数据库版本。
- MCP 目前是工具边界示例，主流程仍走本地工具。
- Eval 目前验证工程链路，不代表真实模型综合效果。

这种说法反而更像认真做过项目的人。

## 3. Tool Use 为什么重要

Tool Use 是这个项目最值得讲的点之一。

如果没有 Tool Use，模型只是根据聊天上下文直接生成评价，容易出现：

- 评分不稳定。
- 不知道参考题库要点。
- 追问容易重复。
- 很难保存结构化结果。
- 很难做 eval。

当前项目里的工具在 `agent/tools.py`：

- `search_question_bank`：按岗位、难度、标签检索题库。
- `score_answer`：根据参考要点和 skill rubric 输出结构化评分。
- `retrieve_knowledge`：从本地知识库检索 RAG 片段。
- `update_user_profile`：生成用户画像增量。
- `generate_review_plan`：根据薄弱点生成复习计划。

每个工具统一输出：

```json
{
  "tool_name": "score_answer",
  "input": {},
  "observation": {},
  "confidence": 0.72,
  "created_at": "2026-06-20 12:00:00"
}
```

面试表达：

> 我理解 Tool Use 的价值不是“看起来高级”，而是把不稳定的自然语言生成拆成可控的工具 observation。比如评分、检索和用户画像更新都应该变成结构化工具输出，这样 UI 能展示，数据库能保存，eval 也能检查。

## 4. ReAct 在项目里怎么落地

ReAct = Reasoning + Acting，但工程里不一定要做无限循环。

当前项目采用有限步 ReAct：

```text
用户回答
-> 调用 score_answer 工具
-> 得到 observation：分数、命中点、缺失点、优势
-> 调用 update_user_profile 工具
-> 调用 retrieve_knowledge 工具
-> select_next_action 决定下一步
-> LLM 生成评价和追问
```

下一步动作限定为：

- `ask_followup`
- `ask_new_question`
- `explain_concept`
- `lower_difficulty`
- `finish_report`

为什么要有限步？

- 面试场景本身是一问一答，不需要无限探索。
- 在线 demo 要稳定，不能让模型无限调用工具。
- 便于调试和保存 trace。
- 便于做 eval。

面试表达：

> 我没有一上来做开放式无限 ReAct loop，而是先做了有限步 ReAct。每轮回答最多经过评分、记忆更新和 RAG 检索，再根据 observation 选择下一步动作。这样保留 Agent 决策能力，同时保证系统可控。

## 5. RAG 现在是什么版本

当前 RAG 是轻量关键词检索版，不是向量数据库版。

文件位置：

- 知识库：`data/knowledge_base.json`
- 检索器：`rag/retriever.py`

当前流程：

```text
用户问题 / 当前回答 / 薄弱点
-> tokenize
-> 按关键词、岗位、标签打分
-> 返回 top-k 知识片段
-> 注入 prompt
```

为什么先用关键词 RAG？

- 不依赖 embedding API key。
- 不依赖 Chroma、FAISS、Milvus 等额外服务。
- Streamlit Cloud 更容易部署。
- 对小规模面经、JD、学习笔记足够可控。
- 适合简历项目第一阶段。

和向量数据库 RAG 的差别：

| 维度 | 当前版本 | 向量数据库版本 |
| --- | --- | --- |
| 检索方式 | 关键词 / BM25-like | embedding 相似度 |
| 存储 | JSON 文件 | Chroma / FAISS / Milvus / SQLite-vec |
| 依赖 | 低 | 中高 |
| 语义召回 | 一般 | 更强 |
| 部署难度 | 低 | 更高 |
| 面试说法 | 轻量 RAG v1 | RAG v2 升级方向 |

面试表达：

> 当前 RAG 是关键词检索版，主要为了保证在线演示稳定。下一步我会升级为 hybrid RAG：embedding 向量召回 + 关键词召回 + rerank。这样既能处理语义相似问题，也保留关键词对技术名词的精确匹配。

## 6. Skill 有什么价值

Skill 的价值是把岗位策略从大 prompt 里拆出来。

如果没有 Skill，所有岗位共用一个大 prompt，会导致：

- Java、Go、AI Agent、算法岗位考察重点混在一起。
- 评分标准不清晰。
- 追问策略不可复用。
- 后续扩展岗位很麻烦。

当前 Skill 配置在 `data/skills/`，例如：

```json
{
  "name": "AI Agent 面试 Skill",
  "role": "AI Agent 开发",
  "focus_areas": ["AgentState", "Tool Use", "ReAct", "RAG", "Memory"],
  "rubric": {},
  "followup_strategies": [],
  "rag_tags": [],
  "difficulty_rules": {}
}
```

面试表达：

> 我把岗位能力抽象成 Skill 配置，让不同岗位可以拥有不同的考察重点、评分 rubric、追问策略和 RAG 标签。这样新增岗位不需要改核心代码，只需要补 skill 和题库。

## 7. Memory 怎么讲

当前 Memory 分两层：

短期记忆：

- 当前聊天消息。
- 当前 AgentState。
- 当前题目、已问问题、评分历史、工具调用。

长期记忆：

- 用户偏好岗位。
- 历史平均分。
- 常见薄弱点。
- 表现优势。
- 推荐复习计划。

存储位置：

- `agent_states` 表：保存每轮 session 的 AgentState。
- `user_profiles` 表：保存跨轮次用户画像。

面试表达：

> 我没有把所有历史对话都塞进 prompt，而是把当前会话放在 AgentState，把跨轮次稳定信息抽取成用户画像。这样能减少上下文膨胀，也方便下次面试继续针对薄弱点训练。

## 8. Eval 怎么讲

当前 Eval 是离线工程链路评测，不是真实模型效果评测。

文件位置：

- `evals/sample_answers.json`
- `evals/run_eval.py`
- `evals/eval_report.md`

当前 eval 验证：

- 工具调用是否成功。
- 评分 JSON 是否包含必要字段。
- 薄弱点是否能被识别。
- 报告是否能生成。

面试表达：

> 我先做了一个 keyless eval，不调用真实模型，主要验证 Agent 工具链路是否稳定。后续会接入真实 LLM eval，评估重复提问率、追问合理性、评分稳定性和报告可用性。

## 9. 最值得继续做的高含金量升级

如果时间有限，优先做下面 5 件事。

### P0：把 Tool Use 改成模型可选择的 Function Calling

当前工具调用主要由代码编排，下一步可以让模型输出结构化 tool call：

```json
{
  "action": "score_answer",
  "action_input": {
    "question": "...",
    "answer": "..."
  }
}
```

升级后流程：

```text
LLM 决定 action
-> ToolRegistry 执行
-> observation 回填
-> LLM 决定 next_action
```

价值：

- 更接近真正 Agent。
- 可以展示 tool schema。
- 可以记录 tool trace。
- 更容易扩展 MCP。

面试含金量：很高。

### P1：升级为 Hybrid RAG

当前是关键词检索。下一步做：

```text
文档上传
-> chunk
-> embedding
-> 存 Chroma / FAISS
-> 向量召回
-> 关键词召回
-> rerank
-> prompt 注入
```

建议技术选型：

- 本地 demo：Chroma 或 FAISS。
- 轻量部署：SQLite + sqlite-vec。
- 中文 embedding：bge-small-zh / bge-m3 / text-embedding API。

价值：

- 面试官熟悉 RAG，一听就知道你做过主流方案。
- 可以讲 chunk size、top-k、召回、rerank、引用来源。
- 能明显区别于普通 prompt 项目。

面试含金量：很高。

### P2：增加 Trace / Replay

记录每轮 Agent 行为：

```json
{
  "step": 1,
  "state": {},
  "tool_call": {},
  "observation": {},
  "next_action": "ask_followup",
  "llm_output": "..."
}
```

价值：

- 能调试为什么追问不合理。
- 能复盘工具调用。
- 能做 eval。
- 能体现 Agent 工程化意识。

面试表达：

> Agent 的难点不是单次回答，而是行为可观测。Trace 可以让我知道模型为什么做了某个决策，工具返回了什么，下一步动作如何产生。

面试含金量：很高。

### P3：真实 LLM Eval

当前 eval 是 keyless 工具链路评测。下一步加真实模型评测：

指标：

- 重复提问率。
- 追问合理性。
- 评分稳定性。
- RAG 引用命中率。
- 报告完整性。
- 平均响应耗时。
- token 成本。

可以准备 30 条模拟候选人回答，自动跑完整流程。

价值：

- 不只是“我感觉效果不错”。
- 能用指标讲项目。
- 很适合 AI Agent / LLM 应用岗位。

面试含金量：高。

### P4：接入真实 MCP Server

当前 `mcp_server.py` 是 MCP-ready 边界示例。下一步可以接入真实 MCP SDK：

暴露工具：

- `search_question_bank`
- `retrieve_knowledge`
- `get_user_profile`
- `update_user_profile`

价值：

- 能说明你理解 MCP 是工具与上下文协议。
- 让面试官看到你不是只会在一个 app 里写函数。
- 后续可以让其他 Agent 复用你的题库和知识库工具。

面试含金量：中高。

## 10. 推荐的下一版 Roadmap

### v1.1：巩固当前轻量 Agent

- 修正 UI 展示，确保 AgentState、工具调用、RAG 片段能清楚展示。
- 给每次工具调用增加 trace id。
- 把 report 里的结构化 JSON 单独保存。
- 增加 10 个单元测试。

目标：让当前版本更稳。

### v1.2：Function Calling Tool Use

- 定义 tool schema。
- 让模型选择工具。
- 增加 tool call parser 和 fallback。
- 保存 Thought-free trace，不保存隐藏推理，只保存 action/observation。

目标：更像真正 Agent。

### v1.3：Hybrid RAG

- 支持上传 JD / 面经 / 学习笔记。
- 自动 chunk。
- embedding 入库。
- 向量召回 + 关键词召回。
- 展示引用来源。

目标：补齐主流 RAG 能力。

### v1.4：Eval Harness

- 构造固定模拟候选人回答集。
- 自动跑多轮面试。
- 统计重复率、评分波动、工具成功率、RAG 命中率。
- 生成 eval dashboard 或 markdown report。

目标：从 demo 走向可评测系统。

### v1.5：MCP Integration

- 把题库、RAG、Memory 包成 MCP tools。
- 写 MCP 使用文档。
- 保留本地工具作为 fallback。

目标：展示工具生态和标准协议意识。

## 11. 简历 bullet 推荐版本

版本一，稳妥：

> 独立开发 AI 模拟面试官 Agent 系统，基于 Streamlit、LangChain、DeepSeek API 和 SQLite 实现岗位选择、题库检索、简历解析、多轮问答、回答评分、历史记录与总结报告生成，并部署至 Streamlit Cloud。

版本二，偏 Agent：

> 将 AI 面试系统从固定 LLM workflow 升级为轻量 Agent 架构，设计 AgentState、岗位 Skill、本地 Tool Registry、轻量 RAG、长期 Memory 和有限步 ReAct 流程，实现回答评分、知识检索、用户画像更新和追问决策的结构化闭环。

版本三，偏工程化：

> 为 AI 面试 Agent 设计离线 Eval Harness，构造 20 条模拟候选人回答，验证工具调用成功率、评分 JSON 完整性和薄弱点识别链路，并生成 Markdown 评测报告，用于持续评估 Agent 行为质量。

## 12. 面试官追问时的回答模板

### Q：你这个和普通 ChatGPT 套壳有什么区别？

回答：

> 普通套壳主要是把用户输入直接发给模型。我这个项目做了业务状态和工具链路拆分：AgentState 记录当前题目、已问问题和评分历史；Skill 控制不同岗位的考察策略；Tool Use 负责题库检索、评分、RAG 检索和画像更新；模型最后基于 observation 生成反馈和追问。所以它不是完全自由聊天，而是一个有状态、有工具、有记忆的面试流程。

### Q：你的 ReAct 具体在哪里？

回答：

> 我做的是有限步 ReAct。用户回答后，系统先调用评分工具得到 observation，再更新用户画像，再检索相关知识，最后根据评分和缺失点选择下一步动作，比如追问、换题、解释概念或降低难度。因为面试是一问一答场景，所以我没有做无限循环，而是限制每轮工具调用和动作集合，保证线上稳定。

### Q：你的 RAG 为什么不用向量数据库？

回答：

> 当前版本是轻量 RAG，用关键词和标签检索本地知识库，主要考虑在线 demo 的稳定性，不依赖 embedding key 和额外数据库。它适合小规模面经和岗位知识。下一步我计划升级为 hybrid RAG，用向量召回处理语义相似问题，同时保留关键词召回处理技术名词精确匹配。

### Q：你的项目还有什么不足？

回答：

> 目前不足主要有三个。第一，Tool Use 还是代码编排为主，还没有完全让模型基于 tool schema 自主选择工具。第二，RAG 还是关键词检索，没有接向量库和 rerank。第三，eval 主要验证工程链路，还没有做真实 LLM 多轮效果评估。后续我会优先补 Function Calling、Hybrid RAG 和真实 Eval Harness。

### Q：如果继续做，你最想优化什么？

回答：

> 我会优先做三件事：第一，把工具调用升级成模型可选择的 function calling，并记录完整 action/observation trace；第二，把 RAG 升级成 embedding + keyword 的 hybrid retrieval；第三，做真实 LLM eval，评估重复提问率、追问合理性、评分稳定性和报告可用性。

## 13. 你该有的底气

这个项目确实不是大型商业 Agent 平台，但对实习面试已经有几个不错的点：

- 你能讲清楚 workflow 和 Agent 的区别。
- 你能讲清楚 Tool Use 为什么比纯 prompt 稳。
- 你能讲清楚 ReAct 为什么要做有限步。
- 你能讲清楚 RAG 为什么先做关键词版，后续怎么升级向量版。
- 你能讲清楚 Memory 为什么要分短期和长期。
- 你能讲清楚 Eval 为什么是 Agent 工程化的一部分。

这比“我调用了某某 API 做聊天机器人”强很多。

真正要补的不是一口气做巨复杂，而是把后续两三个高价值点做扎实：

1. Function Calling Tool Use。
2. Hybrid RAG。
3. Trace + Eval。

把这三块补上，这个项目就会从“能讲”变成“比较有竞争力”。
