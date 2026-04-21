# Diet-RAG-Agent

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/Agent-LangGraph-111827)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![ChromaDB](https://img.shields.io/badge/Retrieval-ChromaDB-6E56CF)
![PostgreSQL](https://img.shields.io/badge/Storage-PostgreSQL-4169E1?logo=postgresql&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-2EA043)

> 基于 LangGraph 多节点架构的垂直饮食领域智能 Agent。  
> 当前主线入口：`src/api/main.py`；`main.py` 保留为兼容 CLI 入口。

## 2. 项目简介

Diet-RAG-Agent 是一个面向中文饮食场景的工程化 Agent 系统，围绕菜谱推荐、营养分析、食材搭配、视频总结、教程问答和多轮承接，提供一条可部署、可评测、可扩展的主线能力。


## 3. 核心功能清单

| 类型 | 功能 | 说明 |
| --- | --- | --- |
| 基础功能 | 菜谱推荐 | 按食材、时长、健康目标、餐次返回 grounded recipe recommendation |
| 基础功能 | 营养分析 | 面向减脂、控糖、高蛋白等场景给出保守型营养建议 |
| 基础功能 | 食材搭配检查 | 判断食材是否适合搭配，并区分证据结论与通用建议 |
| 基础功能 | 菜谱教程问答 | 基于 tutorial collection 回答“怎么做 / 步骤 / 细节”类问题 |
| 基础功能 | Meal planning 提示 | 针对周计划类请求返回 outline / candidate 方案 |
| 基础功能 | 同步 / 流式 API | 提供 `POST /chat` 与 `POST /chat/stream` 两类接口 |
| 基础功能 | Web Demo 与会话历史 | 内置 `GET /demo`、会话列表和历史恢复接口 |
| 创新功能 | 多节点 Agent 编排 | 主链路为 `Router → Planner → Retriever → Generator → Evaluator` |
| 创新功能 | Skill-aware retrieval | `active_skill` 驱动 clarification policy、retrieval profile、rerank bias 与 evidence boundary |
| 创新功能 | 记忆闭环 | session history、recent feedback、recommended anchors、stable preferences 参与读回与写回 |
| 创新功能 | Tutorial / Video import | 支持将 recipe tutorial 与 Bilibili 视频内容转换为 JSON / PDF / Chroma chunks |


## 4. 技术栈

| 分类 | 组件 | 作用 |
| --- | --- | --- |
| Agent 框架 | LangGraph, LangChain | 多节点状态编排、LLM / tool integration |
| API 层 | FastAPI, Uvicorn, SSE | 同步接口、流式输出、Web Demo 接入 |
| 检索与 RAG | ChromaDB, DashScope Embeddings, reranker, hybrid retrieval | 菜谱 / 教程检索、过滤、精排 |
| 存储层 | PostgreSQL, in-memory session store, JSON / PDF artifacts | 用户数据、交互历史、反馈、教程产物 |
| 模型接入 | DashScope OpenAI-compatible API | LLM 与 embedding 调用 |
| 可观测性 | structured logging, Langfuse optional | 请求日志、链路 tracing |
| 评测与质量 | benchmark scripts, evaluator loop, pytest | 离线评测、回归验证、质量闭环 |
| 工具链 | Docker, docker-compose, pytest, benchmark scripts | 本地部署、服务编排与工程验证 |

## 5. 系统整体架构

### 请求处理主线

1. Web Demo 或 API Client 向 `src/api/main.py` 发起请求。
2. API 层读取显式约束、`user_id`、`session_id`，并从 session store / PostgreSQL 组装历史与偏好。
3. `run_diet_agent()` 进入 LangGraph 主图 `src/graph/diet_graph.py`。
4. `Router` 判断意图与 `active_skill`，`Planner` 决定是否先澄清、检索或直接生成。
5. `Retriever` 按 skill 合同执行（混合/非混合）向量检索、过滤、精排，并产出 retrieval stats。
6. `Generator` 基于证据生成回答或教程，`Evaluator` 负责质量判断与必要时的再生成。
7. API 层将 interaction / feedback / recommendation anchors 写回，并返回完整 metadata。
8. 若调用 `/chat/stream`，系统会通过 SSE 持续输出 chunk 和节点进度事件。

### 关键分层

- **接入层**
  - `GET /health`
  - `GET /demo`
  - `POST /chat`
  - `POST /chat/stream`
  - `GET /users/{user_id}/sessions`
  - `GET /users/{user_id}/sessions/{session_id}/history`

- **编排层**
  - `src/graph/` 负责 state、nodes、edges 与主图调度
  - `diet_agent/runtime/skills.py` 维护 skill contract 与 runtime capability

- **知识层**
  - 菜谱知识库
  - Recipe tutorial collection
  - Bilibili tutorial collection

- **记忆与持久化层**
  - 会话级短期记忆在内存中维护
  - 用户偏好、交互、反馈等结构化数据持久化到 PostgreSQL
  - 教程 JSON / PDF、benchmark 报告落盘为可审计资产

## 6. 数据库存储设计

### 数据架构总览

| 层级 | 存储介质 | 主要内容 | 典型用途 |
| --- | --- | --- | --- |
| Vector Knowledge | ChromaDB | `recipes`, `recipe_tutorials`, `bilibili_tutorials` | RAG 检索、教程问答、视频总结入库 |
| Structured Persistence | PostgreSQL | users, user preferences, interactions, feedbacks, inventory, nutrition goals | 用户画像、反馈闭环、会话级交互追溯 |
| Runtime Memory | In-memory session store | history messages, recent feedback signals, recommendation anchors | 多轮承接、最近会话加载、上下文拼装 |
| Artifact Layer | JSON / PDF / logs / cache | tutorial outputs, benchmark reports, exported assets | 演示产物、离线分析、调试与回放 |

### 设计原则

- **向量数据与结构化数据分层**
  - ChromaDB 负责可检索知识
  - PostgreSQL 负责强结构化用户数据与反馈写回

- **短期记忆与长期偏好分层**
  - 短期上下文优先从 session store 获取
  - 稳定偏好从 PostgreSQL 读回，避免每轮重复输入

- **允许降级运行**
  - PostgreSQL 不可用时，系统仍可基于向量库和内存会话继续工作
  - 这保证了 Demo 和主链路具备较好的本地可运行性

## 7. 快速启动 / 部署步骤

### 7.1 本地启动

#### 1) 创建环境并安装依赖

```powershell
conda create -n myagent python=3.11 -y
conda activate myagent
pip install -r requirements.txt
```

#### 2) 配置环境变量

复制 `.env.example` 为 `.env`，至少配置：

- `DASHSCOPE_API_KEY`

常用可选项包括：

- `LLM_MODEL`
- `EMBEDDING_MODEL`
- `CHROMA_DB_PATH`
- `POSTGRES_HOST`
- `POSTGRES_PORT`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_DATABASE`
- `LANGFUSE_*`

#### 3) 初始化 Chroma 向量库

使用最小可运行数据：

```powershell
python scripts/init_database.py
```

使用更完整的 recipe 数据：

```powershell
python scripts/init_database.py --recipes-file data/recipes_v2.json
```

#### 4) 启动 API 主线

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

启动后可直接访问：

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/demo`

#### 5) 可选能力入口

构建 recipe tutorial 并入库：

```powershell
python scripts/build_recipe_tutorials.py
```

生成 Bilibili 视频结构化教程：

```powershell
python scripts/build_bilibili_tutorial_summary.py <bilibili_url> --cookies-file <cookies.txt>
```
- **视频总结 skill 的工作流方向和产品抽象，参考了 `wdkns-skills` 项目**
- 来源仓库：`https://github.com/wdkns/wdkns-skills`
- 本仓库中的实现是面向饮食 Agent 场景的适配与工程接线，不是对原项目的逐文件复制
当前仓库里的对应实现和入口主要包括：
- `scripts/build_recipe_tutorials.py`
- `scripts/build_bilibili_tutorial_summary.py`
- `src/tutorials/`
- `src/vectorstore/tutorial_store.py`
其中 Bilibili 视频总结流程会把公开视频内容转换成结构化教程，再入到 `bilibili_tutorials` 
collection。若目标视频受平台限制，抓取阶段可能需要额外 cookies 配置


#### 6) 兼容 CLI 入口

```powershell
python main.py
```

> 说明：`main.py` 为兼容 CLI 入口；当前 API / LangGraph 主线以 `src/api/main.py` 为准。

### 7.2 Docker Compose 部署

#### 1) 准备 Docker 环境变量

将 `docker/.env.docker.example` 复制为 `docker/.env.docker`，并补齐 API Key、PostgreSQL 密码等配置。

#### 2) 启动服务

```powershell
docker compose -f docker/docker-compose.yml up --build
```

默认会启动：

- `agent-api`
- `chromadb`
- `postgres`

若需要可观测性面板，可额外启用 Langfuse profile：

```powershell
docker compose -f docker/docker-compose.yml --profile observability up --build
```

## 8. 项目目录结构

```text
Diet-RAG-Agent/
├── src/
│   ├── api/             # FastAPI、Schema、SSE、Web Demo 接口
│   ├── graph/           # LangGraph 主图、state、nodes、edges
│   ├── context/         # memory manager、上下文拼装、指令层
│   ├── database/        # PostgreSQL 客户端与结构化数据访问
│   ├── evaluation/      # benchmark 与评测逻辑
│   ├── tutorials/       # tutorial / video-summary 生成与导出
│   ├── vectorstore/     # Chroma 连接、tutorial ingest、检索相关实现
│   ├── retriever/       # enhanced retriever 与检索增强逻辑
│   ├── observability/   # structured logging、Langfuse integration
│   └── utils/           # 公共工具、日志、token usage
├── diet_agent/          # runtime façade、integrations、user memory、compat exports
├── data/                # recipe 数据、tutorial 数据、benchmark 数据集
├── scripts/             # 初始化、benchmark、tutorial/video 构建脚本
├── tests/               # 回归测试
├── docker/              # Dockerfile、compose、entrypoint、Docker env 模板
├── artifacts/           # tutorial PDF、video summary 产物、离线样例
├── web_chat_demo.html   # 单文件本地聊天 Demo
├── main.py              # 兼容 CLI 入口
├── requirements.txt
└── README.md
```

## 9. 创新亮点与价值

### 技术价值

- **多节点编排而非单轮黑盒**
  - 把路由、规划、检索、生成、评估拆成独立节点，便于调试、观测和替换。

- **Skill Contract 驱动的检索与回答**
  - 不同饮食任务共享同一框架，但拥有不同的 clarification、retrieval、evidence 与 response contract。

- **面向真实使用的记忆闭环**
  - 不只保留消息历史，还把 recent feedback、推荐锚点和稳定偏好用于下一轮生成。

- **从内容导入到知识入库的统一工作流**
  - Recipe tutorial 和视频总结都能落成 JSON / PDF / vector chunks，成为可继续调用的知识资产。

- **Benchmark-first 的工程习惯**
  - 仓库内置 benchmark 脚本和报告生成能力，便于量化 latency、token、skill execution 与约束命中。

### 产品价值

- **更贴合垂直场景**
  - 饮食推荐不只回答“吃什么”，还覆盖“为什么推荐”“怎么做”“适不适合我”。

- **更适合演示与集成**
  - 同时具备 Web Demo、同步 API、SSE 流式 API、会话历史接口和 Docker 部署脚手架。

- **更适合作为作品集与开源项目**
  - 既能展示 Agent 架构能力，也具备真实的数据、脚本、测试和部署入口。

## 10. 更新日志 & 维护信息

### 更新日志

- **Current Mainline**
  - 以 `FastAPI + LangGraph + ChromaDB + PostgreSQL + Structured Memory + Benchmarking` 为核心
  - 主入口为 `src/api/main.py`

- **Recent Additions**
  - 新增同步 / 流式 API、Web Demo、会话列表与历史恢复接口
  - 新增 recipe tutorial 与 Bilibili video-summary 知识导入流程
  - 新增 interaction / feedback / preference loopback
  - 新增离线 quantitative benchmark 脚本与报告生成能力

### 维护说明

- `src/` 与 `diet_agent/` 是当前主线代码区，新增能力优先接入这两层。
- `main.py` 保留为兼容 CLI 入口，不应替代 API 主线作为对外说明。
- 提交涉及路由、检索、生成或 memory contract 的改动时，建议同步更新：
  - `tests/`
  - `scripts/_quant_benchmark.py` 相关数据集或报告
  - 必要的 tutorial / vectorstore 构建脚本
- GitHub 默认会提交以下仓库内容：
  - `src/`、`diet_agent/`、`scripts/`、`tests/`、`data/` 等主线代码与数据目录
  - `README.md`、`web_chat_demo.html`、`.env.example`、`docker/.env.docker.example` 等公开模板与说明文件
  - 已筛除敏感信息的 tutorial / video-summary 样例 JSON、PDF 产物
  - 当前仓库根目录下保留的工程说明文档与阶段性 Markdown 文件
- 不要提交以下本地敏感或运行时文件：
  - `.env`
  - `docker/.env.docker`
  - cookies 文件
  - 本地 Chroma 持久化目录
  - logs / cache / SQLite 缓存 / 临时音视频产物
  - Bilibili 原始 metadata / transcript / whisper / audio / cover 运行产物

欢迎围绕以下方向继续演进：

- 更稳定的 meal planning 子图
- 更细粒度的 skill contract 与策略编排
- 更强的评测自动化与回归基线
- 更完整的前端产品体验与用户反馈闭环
