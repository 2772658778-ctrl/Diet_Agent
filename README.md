# Diet-RAG-Agent

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/Agent-LangGraph-111827)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![ChromaDB](https://img.shields.io/badge/Retrieval-ChromaDB-6E56CF)
![PostgreSQL](https://img.shields.io/badge/Storage-PostgreSQL-4169E1?logo=postgresql&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-2EA043)

> 基于 LangGraph 多节点编排的中文饮食智能 Agent。项目围绕菜谱推荐、营养分析、食材搭配、教程问答、视频总结（可复用skill 产物）和多轮承接，提供一条可运行、可扩展、可量化评测的工程主线。
> 当前对外主入口为 `src/api/main.py`，`main.py` 保留为兼容 CLI 入口。

> 当前 GitHub 仓库采用 **mainline-only** 公开边界：保留主线代码、核心数据与关键脚本；测试、Docker 编排、运行时 artifacts 等工程辅料默认不纳入当前公开快照。

## 1. 项目简介

Diet-RAG-Agent 不是一个“只会调模型回答问题”的 Demo，而是一个面向中文饮食场景的工程化 Agent 系统。它把一次请求拆成可观测、可替换、可评估的几个阶段：

- `Router` 识别意图并选择 `active_skill`
- `Planner` 决定澄清、直接承接还是进入检索
- `Retriever` 基于 skill contract 执行检索、过滤、精排
- `Generator` 基于证据生成回答，或直接执行 tutorial / video workflow
- `Evaluator` 在需要时做质量检查和有限重试

## 2. 核心能力

| 能力 | 说明 | 关键实现路径 |
| --- | --- | --- |
| 菜谱推荐 | 基于食材、时长、健康目标、餐次等约束生成 grounded recommendation | `src/graph/nodes/router.py` `planner.py` `retriever.py` `generator.py` |
| 营养分析 | 面向减脂、控糖、高蛋白等目标给出保守型营养建议 | `src/graph/nodes/generator.py` |
| 食材搭配检查 | 区分“证据结论”和“通用建议”，避免无依据强判断 | `src/graph/nodes/generator.py` |
| 教程问答 | 用 tutorial collection 回答“怎么做 / 步骤 / 教程”类问题 | `src/graph/nodes/retriever.py` `src/vectorstore/tutorial_store.py` |
| Bilibili 视频总结 | 优先导入skill 的 `result.json`，规范化为结构化教程、PDF 和可检索 chunk，并在缺失时回退原生抓取流程 | `src/graph/nodes/generator.py` `src/tutorials/windsurf_skill_bridge.py` `src/tutorials/bilibili_summary.py` `scripts/build_bilibili_tutorial_summary.py` `scripts/render_bilibili_tutorial_pdf.py` |
| 会话承接 | 读回 history、recent feedback、recent recommendation anchors、stable preferences | `diet_agent/user/session_store.py` `diet_agent/app/service.py` |
| 同步 / 流式 API | 提供完整响应与 SSE 节点流式事件两条链路 | `POST /chat` `POST /chat/stream` |
| 会话列表与历史恢复 | 支持 session 列表、历史消息与 interaction 读回 | `GET /users/{user_id}/sessions` `GET /users/{user_id}/sessions/{session_id}/history` |

## 3. 系统主线架构

### 3.1 分层视图

| 层级 | 职责 | 关键模块 |
| --- | --- | --- |
| 接入层 | HTTP API、SSE、Demo 页面、健康检查 | `src/api/main.py` |
| 编排层 | LangGraph state、节点调度、条件边与重试策略 | `src/graph/` |
| Runtime / Contract 层 | skill registry、retrieval profile、evidence policy、fallback policy | `diet_agent/runtime/skills.py` |
| 知识层 | recipe / tutorial / bilibili tutorial 检索与入库 | `src/vectorstore/` `src/retriever/` |
| 记忆层 | session history、feedback signals、recommendation anchors、stable preferences | `diet_agent/user/` `src/context/` |
| 持久化层 | PostgreSQL 结构化数据、JSON/PDF 教程产物、离线数据集 | `src/database/` `src/tutorials/` `data/` |

### 3.2 请求执行主线

1. 客户端通过 `POST /chat` 或 `POST /chat/stream` 发起请求。
2. API 层读取显式约束，并从 session store 收集历史消息、最近反馈信号、最近推荐锚点。
3. 如果提供 `user_id`，系统还会从 PostgreSQL 读回稳定偏好，用于初始化 graph state。
4. `run_diet_agent()` 进入 `src/graph/diet_graph.py`，执行 `Router -> Planner -> Retriever -> Generator -> Evaluator` 主链路。
5. `Router` 产出 `intent` 和 `active_skill`；`Planner` 决定是澄清、direct follow-up 还是进入检索。
6. `Retriever` 根据 skill contract 选择检索 profile，并返回 `retrieved_docs`、`reranked_docs`、`retrieval_stats`。
7. `Generator` 基于证据生成回答；对 tutorial / video 场景，也可以直接执行“导入 / 抓取 -> 规范化 -> canonical 存储 -> 导出 -> 入库”工作流。
8. 若进入 `Evaluator`，系统会根据满意度判断是否有限重试；若为 `chitchat`、`clarification`、`video_summary` 或显式 `skip_graph_eval`，则跳过评估。
9. 请求结束后，API 层会把最新 history 写回 session store，并把 interaction / feedback / memory metadata 组织进响应。

### 3.3 为什么是 Skill-aware Agent

这个项目的关键不是“多了几个节点”，而是把节点行为与 `active_skill` 绑定起来：

- **clarification policy**：不同 skill 需要的缺失槽位不同
- **retrieval profile**：菜谱检索、教程检索、视频总结入库使用不同路径
- **hard filter policy**：不同技能对时长、目标、证据边界的约束不同
- **rerank bias**：inventory match、goal fit、time fit、feedback preference 可以影响排序
- **evidence boundary**：某些回答必须严格基于检索证据，不能越界扩写

## 4. 数据逻辑与存储设计

这一部分是项目最值得在 README 中讲清楚的地方：**不同数据被放进不同层，不是为了“看起来复杂”，而是为了降低耦合、控制成本，并让每一层只做自己最擅长的事。**

### 4.1 四层数据分工

| 数据层 | 典型内容 | 何时写入 | 何时读取 | 设计原因 |
| --- | --- | --- | --- | --- |
| Vector Knowledge | `recipes`、`recipe_tutorials`、`bilibili_tutorials` collection | 初始化知识库或导入 tutorial / video 后写入 ChromaDB | 检索阶段 | 适合做语义召回，不适合存强结构关系 |
| Structured Persistence | users、preferences、interactions、feedbacks、inventory、nutrition goals | 用户更新偏好、请求完成、反馈提交后写入 PostgreSQL | 请求开始读用户偏好；历史恢复时读 interaction | 适合存强结构、可追溯、可查询的数据 |
| Runtime Memory | session history、recent feedback signals、recent recommendation anchors、session index | 每次请求结束后更新内存态 | 下一轮请求开始时读回 | 低延迟承接多轮上下文，不强依赖数据库 |
| Artifact / Dataset Layer | `data/*.json`、tutorial JSON、PDF、离线 benchmark 数据集 | 初始化、导入、评测时生成 | 构建知识库、离线分析、演示回放 | 便于审计、复现实验和离线处理 |

### 4.2 一次请求的数据流

#### 请求前读什么

- **来自请求体的显式约束**
  - `available_ingredients`
  - `allergies`
  - `disliked_ingredients`
  - `max_cooking_time`
  - `health_goal`
  - `meal_type`
  - `prefer_inventory_first`

- **来自 session store 的短期运行时上下文**
  - 最近消息 history
  - 最近反馈压缩信号
  - 最近推荐过的菜谱锚点

- **来自 PostgreSQL 的稳定偏好**
  - 长期口味偏好
  - 不喜欢的食材
  - 健康目标
  - 时间限制
  - 热量/蛋白目标等结构化字段

#### 请求中怎么用

- `Router` / `Planner` 主要消费显式约束和历史上下文
- `Retriever` 主要消费 `active_skill`、用户约束、稳定偏好和反馈信号
- `Generator` 主要消费证据文档、skill contract 和 follow-up 上下文
- `Evaluator` 只关心生成结果与证据是否满足质量要求

#### 请求后写什么

- **写回 session store**
  - 最新消息 history
  - recent feedback signals
  - recent recommended recipes
  - session preview / session index

- **写回 PostgreSQL**
  - interaction 记录
  - feedback 记录
  - 用户偏好更新后的结构化字段

- **返回给客户端的 metadata**
  - `active_skill`
  - `planner_next_action`
  - `retrieval_stats`
  - `evaluation`
  - `memory` readback 摘要
  - `recommended_recipes`
  - `interaction_id`

### 4.3 降级策略

这个分层还有一个工程价值：**允许部分依赖不可用时继续运行主链路。**

- PostgreSQL 不可用时，系统仍可基于向量库和内存 session store 工作
- tutorial / bilibili 导入失败时，不会破坏普通问答链路
- `/chat` 与 `/chat/stream` 共用同一 graph 主线，只是在返回方式上不同

## 5. Tutorial / Video Pipeline

项目里除了问答主链路，还有一条“把外部内容转成可检索知识”的数据生产链路。（需要自行将原版skill适配json输出）（最先的版本有非skill版，不过仅字幕及元数据，字幕、关键帧识别等功能需自行扩展）
参考来源：`https://github.com/wdkns/wdkns-skills`
### 5.1 Recipe Tutorial Pipeline

`recipes.json -> tutorial payload -> JSON -> PDF -> tutorial chunks -> Chroma collection`

关键入口：

- `scripts/build_recipe_tutorials.py`
- `src/tutorials/pipeline.py`
- `src/tutorials/storage.py`
- `src/vectorstore/tutorial_store.py`

本地执行后，系统会：

1. 从 recipe 数据生成结构化 tutorial
2. 导出 tutorial JSON
3. 导出 PDF 供人工查看
4. 切成 tutorial chunks
5. 写入 `recipe_tutorials` collection 供后续检索

### 5.2 Bilibili Video Summary Pipeline

主路径（优先复用skill 产物）：

`video url -> artifacts/bilibili_summaries/<video_id>/result.json -> bridge normalize -> canonical tutorial JSON -> tutorial chunks -> bilibili_tutorials -> PDF`

回退路径（无 `result.json` 或导入失败时）：

`video url -> subtitle / cookies / whisper fallback -> structured tutorial -> canonical tutorial JSON -> tutorial chunks -> bilibili_tutorials -> PDF`

关键入口：

- `src/graph/nodes/generator.py`
- `src/tutorials/windsurf_skill_bridge.py`
- `scripts/build_bilibili_tutorial_summary.py`
- `src/tutorials/bilibili_summary.py`
- `scripts/render_bilibili_tutorial_pdf.py`
- `src/tutorials/storage.py`
- `src/vectorstore/tutorial_store.py`

固定路径与公开边界：

- 运行时 skill 产物：`artifacts/bilibili_summaries/<video_id>/result.json`
- canonical 教程 JSON：`data/tutorials/bilibili/<video_id>/tutorial_bilibili_<video_id>.json`
- PDF：`artifacts/tutorial_pdfs/bilibili/<video_id>/<title>.pdf`
- 向量库 collection：`bilibili_tutorials`

默认行为：

1. 当 `BILIBILI_PREFER_WINDSURF_SKILL_RESULT=true` 且固定路径下存在 `result.json` 时，Diet Agent 会优先导入 Windsurf skill 的结构化结果。
2. 若 `result.json` 缺失、损坏或字段不完整，则自动回退到原生字幕 / cookies / Whisper 流程。
3. 无论走哪条路径，都会统一保存 canonical JSON，并将 tutorial chunk 写入 `bilibili_tutorials`，以便后续检索复用。
4. PDF 可由 workflow 自动生成，也可通过独立 CLI 在指定 Python 环境中重新渲染。

这个流程适合展示一个很工程化的点：**系统不仅能消费已有知识，还能把外部内容转成自己的知识资产。**

### 5.3 另一种tutorial

设计之初，还有一组基于语义+轻量知识图谱的recipe。此处保留了下来，检索若命中此类recipe时，采用混合检索，大致为：语义召回+关系推理+cross-encoder+RRF加权融合+MMR控制多样+各种外部约束。

此设计理念在小半年前，是为了解决单纯recipe语义向量可能无法覆盖如原材料、过敏源等等情况，为了避免维护知识图谱而做的尝试，可尝试后续在2026年3月发表的UniAI GraphRAG理论上进一步探索！

## 6. API 与运行接口

### 6.1 API Surface

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查，返回 graph / postgres / chromadb / langfuse 状态 |
| `GET` | `/demo` | 单文件 Web Demo |
| `POST` | `/chat` | 返回完整回答和 metadata |
| `POST` | `/chat/stream` | SSE 流式返回 chunk 与节点进度事件 |
| `GET` | `/users/{user_id}/sessions` | 返回最近会话摘要 |
| `GET` | `/users/{user_id}/sessions/{session_id}/history` | 返回会话消息与 interaction 读回视图 |

### 6.2 `/chat` 与 `/chat/stream` 的区别

- `POST /chat`
  - 更适合服务端集成或一次性拿完整结果
  - 返回完整 `response + metadata`

- `POST /chat/stream`
  - 更适合前端交互和节点级可视化
  - 返回 SSE chunk / 进度事件
  - 它和 `/chat` 的 graph 主线相同，但**返回协议并不等价**

## 7. 快速启动

### 7.1 创建环境

```powershell
conda create -n myagent python=3.11 -y
conda activate myagent
pip install -r requirements.txt
```

### 7.2 配置环境变量

复制 `.env.example` 为 `.env`，至少补齐：

- `DASHSCOPE_API_KEY`

常见可选项：

- `LLM_MODEL`
- `EMBEDDING_MODEL`
- `CHROMA_DB_PATH`
- `POSTGRES_HOST`
- `POSTGRES_PORT`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_DATABASE`
- `LANGFUSE_*`
- `BILIBILI_COOKIES_FROM_BROWSER`
- `BILIBILI_COOKIES_FILE`
- `BILIBILI_WHISPER_MODEL`
- `BILIBILI_PREFER_WINDSURF_SKILL_RESULT`
- `BILIBILI_PDF_PYTHON_EXECUTABLE`

典型 Bilibili 配置示例：

```dotenv
BILIBILI_PREFER_WINDSURF_SKILL_RESULT=true
BILIBILI_PDF_PYTHON_EXECUTABLE=D:\anaconda\envs\myagent\python.exe
```

### 7.3 初始化主知识库

```powershell
python scripts/init_database.py
```

如果希望使用更完整的数据集：

```powershell
python scripts/init_database.py --recipes-file data/recipes_v2.json
```

### 7.4 启动 API

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

启动后可访问：

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/demo`

### 7.5 可选的数据生产入口

构建 recipe tutorial：

```powershell
python scripts/build_recipe_tutorials.py
```

构建 Bilibili 视频教程总结：

```powershell
python scripts/build_bilibili_tutorial_summary.py <bilibili_url> --cookies-file <cookies.txt>
```

如果 Windsurf skill 已经在固定路径下产出 `result.json`，直接把视频 URL 交给 `/chat`、`/chat/stream` 或 `run_diet_agent()` 即可优先复用该结果；系统会自动规范化到 canonical JSON、入库到 `bilibili_tutorials`，并尝试生成 PDF。

从 canonical 教程 JSON 独立渲染 PDF：

```powershell
python scripts/render_bilibili_tutorial_pdf.py --json-path data/tutorials/bilibili/<video_id>/tutorial_bilibili_<video_id>.json
```

### 7.6 兼容 CLI 入口

```powershell
python main.py
```

> 说明：`main.py` 主要用于兼容旧使用方式；当前推荐对外入口仍是 `src/api/main.py`。

## 8. 当前公开仓库结构

```text
Diet-RAG-Agent/
├── src/
│   ├── api/             # FastAPI、SSE、Schema、Demo 接口
│   ├── graph/           # LangGraph 主图、state、nodes、edges
│   ├── context/         # 上下文拼装、记忆管理
│   ├── database/        # PostgreSQL 客户端
│   ├── evaluation/      # benchmark 与评测逻辑
│   ├── retriever/       # enhanced retriever
│   ├── tutorials/       # tutorial / video-summary 处理与导出
│   ├── vectorstore/     # Chroma 接入、tutorial ingest、检索实现
│   ├── observability/   # structured logging / Langfuse integration
│   └── utils/           # 日志、错误处理、token usage
├── diet_agent/          # runtime façade、skills、integrations、user memory
├── data/                # recipe 数据、tutorial 数据、benchmark 数据集
├── scripts/             # 初始化、tutorial/video 构建、PDF 渲染、benchmark 入口
├── web_chat_demo.html   # 单文件 Web Demo
├── main.py              # 兼容 CLI 入口
├── requirements.txt
├── README.md
└── README1.md
```

> 说明：完整本地工程工作区可能还包含测试、Docker、artifacts 和额外文档；当前公开仓库默认聚焦主线代码与关键数据，不把所有工程辅料一起暴露出来。

## 9. 工程亮点

- **多节点显式编排，而不是单轮黑盒调用**
  - 每个阶段的输入、输出和短路条件都可以单独观测和调试。

- **Skill Contract 驱动的统一框架**
  - 不同任务共享一套 graph，但通过 `active_skill` 切换检索策略、回答约束和 fallback 逻辑。

- **数据分层清晰**
  - Chroma 负责可检索知识，PostgreSQL 负责结构化状态，session store 负责低延迟短期承接，JSON/PDF 负责可审计产物。

- **从外部内容到内部知识的闭环**
  - tutorial / video 内容不仅能被总结，还能被转成可再检索的知识资产。

- **适合展示工程思维**
  - 同时覆盖 API、Graph orchestration、RAG、memory、structured persistence、observability 和 benchmark。

## 10. 维护建议

- `src/` 与 `diet_agent/` 是当前主线代码区，新增能力优先接入这两层。
- 如果你修改了路由、检索、memory contract 或 skill policy，建议同步检查：
  - `scripts/_quant_benchmark.py`
  - `data/eval/*.json`
  - tutorial / video ingest 入口是否仍匹配新的 contract
- 本地敏感文件不要提交：
  - `.env`
  - cookies 文件
  - 本地 Chroma 持久化目录
  - logs / cache / SQLite 缓存
  - tutorial/video 原始运行产物（包括 `artifacts/bilibili_summaries/<video_id>/result.json` 与导出的 PDF）
