# Elasticsearch 入库与检索 —— 完整技术复盘

> 本文档整理自 2026-06-27 的开发会话，涵盖从环境搭建到完整实现的所有设计决策、技术分析和最终代码。

---

## 目录

1. [整体架构](#1-整体架构)
2. [环境搭建](#2-环境搭建)
3. [架构决策：单索引 vs 六索引](#3-架构决策单索引-vs-六索引)
4. [配置体系](#4-配置体系)
5. [MedicalQAStore 类设计](#5-medicalqastore-类设计)
6. [Index Mapping 设计](#6-index-mapping-设计)
7. [create_index 实现](#7-create_index-实现)
8. [数据入库脚本设计](#8-数据入库脚本设计)
9. [search 实现](#9-search-实现)
10. [性能分析](#10-性能分析)
11. [最终文件结构](#11-最终文件结构)

---

## 1. 整体架构

### 系统流程

```
用户提问
   │
   ▼
BERT 意图识别 → 6 大科室之一（儿科/内科/外科/妇产科/男科/肿瘤科）
   │
   ├─ 置信度高 → ES BM25 检索，限定 category 过滤
   │
   └─ 置信度低 → ES 全量检索（不过滤 category）
   │
   ▼
返回 top-K 结果（含 answer）
```

### 关键澄清

- BERT 分类的是 **category**（6 大科室），不是 CSV 中的 `department`（细分科室）
- `department` 是 CSV 内的字段（如"心血管科"、"新生儿科"），入库但不用于过滤
- 当 BERT 置信度不高时，全量搜索作为兜底

### 数据源

6 个 CSV 文件，共约 300MB，~44 万条 QA 对：

| 文件 | category |
|------|----------|
| 儿科.csv | 儿科 |
| 内科.csv | 内科 |
| 外科.csv | 外科 |
| 妇产科.csv | 妇产科 |
| 男科.csv | 男科 |
| 肿瘤科.csv | 肿瘤科 |

CSV 字段结构：

| 字段 | 说明 | 示例 |
|------|------|------|
| `line_id` | 行号 ID | 1 |
| `department` | 细分科室 | 心血管科 |
| `title` | 问题标题 | 高血压患者能吃党参吗？ |
| `ask` | 患者提问详情 | 我有高血压这两天... |
| `answer` | 医生回答 | 高血压病人可以口服党参的... |

---

## 2. 环境搭建

### 2.1 Python 依赖

```bash
uv add "elasticsearch>=8.0,<9.0"
uv add pandas
```

版本建议：ES Python 客户端主版本对齐服务端（8.x 客户端连 8.x 服务端）。

### 2.2 ES 服务端（Docker Compose）

```yaml
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.19.17
    container_name: es-local
    mem_limit: 2g
    environment:
      - discovery.type=single-node
      - ES_JAVA_OPTS=-Xms1g -Xmx1g
      - xpack.security.enabled=false
      - bootstrap.memory_lock=true
    ulimits:
      memlock:
        soft: -1
        hard: -1
    volumes:
      - my_es_data:/usr/share/elasticsearch/data
      - ./es-plugins:/usr/share/elasticsearch/plugins
    ports:
      - "9200:9200"
      - "9300:9300"
    networks:
      - search-net

volumes:
  my_es_data:

networks:
  search-net:
    driver: bridge
```

关键配置说明：

| 配置 | 含义 |
|------|------|
| `mem_limit: 2g` | 容器总内存上限（1GB堆 + 1GB堆外） |
| `ES_JAVA_OPTS=-Xms1g -Xmx1g` | JVM 堆内存固定 1GB |
| `xpack.security.enabled=false` | 关闭安全认证，本地开发免密 |
| `bootstrap.memory_lock=true` | 锁定内存，防止交换到磁盘 |
| `discovery.type=single-node` | 单节点模式，跳过集群检查 |

### 2.3 IK 中文分词器

**IK 是 ES 生态中中文分词的事实标准。**

| 分词器 | 特点 | 适用场景 |
|--------|------|----------|
| **IK** | 维护活跃、社区最大、支持自定义词典 | 通用中文检索，首选 |
| jieba | Python 原生，不在 ES 中 | Python 应用层分词 |
| HanLP | 学术精度高 | NLP 研究 |

IK 两个模式：

- `ik_max_word`：最细粒度切分（"高血压" → 高/血压/高血压），**索引时用**，召回率高
- `ik_smart`：粗粒度切分（"高血压" → 高血压），**搜索时用**，精准度高

安装方式：下载对应版本 zip → 解压到 `es-plugins/ik` 目录 → 重启容器。

验证分词：

```python
result = await es.indices.analyze(index='medical_qa', analyzer='ik_max_word', text='高血压患者能吃党参吗')
# 返回 6 个 token: 高血压, 血压, 患者, 能吃, 党参, 吗
```

---

## 3. 架构决策：单索引 vs 六索引

### 场景参数

| 参数 | 值 |
|------|-----|
| 总文档数 | ~44 万 |
| 单文档大小 | ~1-2 KB |
| 总数据量 | ~600 MB |
| 科室分布 | 6 个大类，不均匀 |

### 对比分析

| 维度 | 单索引 | 六索引 |
|------|--------|--------|
| **过滤查询性能** | bitset 过滤，~0 开销 | 省了 bitset，差别极小 |
| **全量查询性能** | 直接查，简单 | scatter-gather 合并，略复杂 |
| **运维复杂度** | 1 个索引，1 套配置 | 6 个索引，6 套配置要保持同步 |
| **mapping 变更** | 改一次 | 改 6 次 |
| **数据更新** | 直接更新 | 要算 category 再决定往哪个索引写 |
| **代码复杂度** | 简单 | 入库和查询都要路由逻辑 |

### 结论：采用单索引

40 万文档、600MB 数据，对 ES 来说是小规模。单索引和六索引在性能上几乎无差别——bitset 过滤和 scatter-gather 都是微秒到毫秒级的操作。但代码和运维复杂度的差别是实打实的。

---

## 4. 配置体系

### config.toml

```toml
# Elasticsearch 配置
[elasticsearch]
host = "127.0.0.1"
port = 9200
index_name = "medical_qa"
```

### config.py

```python
class ElasticsearchConfig(BaseModel):
    host: str
    port: int
    index_name: str

class Config(BaseModel):
    mysql: MysqlConfig
    redis: RedisConfig
    milvus: MilvusConfig
    elasticsearch: ElasticsearchConfig   # 新增
    llm: LlmConfig
    retrieval: RetrievalConfig
    logger: LoggerConfig
    app: AppConfig
```

对齐项目现有的 Pydantic 配置风格，统一通过 `load_config()` 加载。

---

## 5. MedicalQAStore 类设计

### 5.1 核心决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 同步 vs 异步 | **异步**（AsyncElasticsearch） | 配合 FastAPI 使用 |
| 连接管理 | **`__aenter__` / `__aexit__`** | 官方推荐异步客户端用 async context manager |
| 日志 | **`setup_logger`** | 对齐项目现有日志体系 |
| 异常处理 | **`MedicalQAStoreError`** | 统一异常，上层只需 catch 一种 |
| 字符串格式化 | **f-string** | 项目统一风格 |

### 5.2 ES 客户端连接管理（官方实际做法）

ES 客户端内部自带连接池管理（`Transport` → `ConnectionPool`），负责：
- 连接复用
- 死亡节点检测和自动复活
- 负载均衡选择

| 场景 | 官方推荐 | 原因 |
|------|----------|------|
| 同步 `Elasticsearch` | 直接实例化 | 内部连接池自动管理 |
| 异步 `AsyncElasticsearch` | context manager | aiohttp 要求显式关闭 |

### 5.3 职责划分

**MedicalQAStore 只负责运行时操作**，不负责数据入库：

```
src/es/store.py          ← 运行时接口：create_index / search / count / delete
scripts/ingest_data.py   ← 离线入库脚本：自包含，不依赖 store.py
```

### 5.4 完整代码

```python
# src/es/store.py

from elasticsearch import AsyncElasticsearch

from src.base import setup_logger
from src.base.config import ElasticsearchConfig

logger = setup_logger("MedicalQAStore")


class MedicalQAStoreError(Exception):
    """医疗问答 ES 存储层统一异常"""


class MedicalQAStore:
    """医疗问答数据的 ES 异步存储层，封装索引管理和 CRUD"""

    def __init__(self, config: ElasticsearchConfig):
        self._host = config.host
        self._port = config.port
        self._index = config.index_name
        self._es: AsyncElasticsearch | None = None

    async def connect(self) -> None:
        """建立连接并验证"""
        url = f"http://{self._host}:{self._port}"
        self._es = AsyncElasticsearch(url)
        try:
            if not await self._es.ping():
                raise MedicalQAStoreError(f"ES 无响应: {url}")
            logger.info(f"ES 连接成功: {url}")
        except MedicalQAStoreError:
            raise
        except Exception as e:
            logger.error(f"ES 连接失败: {e}", exc_info=True)
            raise MedicalQAStoreError(f"ES 连接失败: {e}") from e

    async def close(self) -> None:
        """关闭连接"""
        if self._es:
            await self._es.close()
            self._es = None
            logger.info("ES 连接已关闭")

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False

    # ── 索引生命周期 ──

    async def index_exists(self) -> bool:
        """检查索引是否存在"""
        try:
            return await self._es.indices.exists(index=self._index)
        except Exception as e:
            logger.error(f"检查索引失败: {e}", exc_info=True)
            raise MedicalQAStoreError(f"检查索引失败: {e}") from e

    async def create_index(self) -> None:
        """创建索引（含 IK 分词配置和 mapping），已存在则跳过"""
        try:
            if await self.index_exists():
                logger.info(f"索引已存在: {self._index}，跳过创建")
                return

            settings = {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "ik_analyzer": {
                            "type": "custom",
                            "tokenizer": "ik_max_word",
                            "filter": ["lowercase"],
                        }
                    }
                },
            }

            mappings = {
                "properties": {
                    "category": {"type": "keyword"},
                    "department": {"type": "keyword"},
                    "title": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "ask": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "answer": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                }
            }

            await self._es.indices.create(index=self._index, settings=settings, mappings=mappings)
            logger.info(f"索引创建成功: {self._index}")
        except Exception as e:
            logger.error(f"索引创建失败: {e}", exc_info=True)
            raise MedicalQAStoreError(f"索引创建失败: {e}") from e

    async def delete_index(self) -> None:
        """删除索引，不存在则跳过"""
        try:
            if not await self.index_exists():
                logger.info(f"索引不存在: {self._index}，跳过删除")
                return
            await self._es.indices.delete(index=self._index)
            logger.info(f"索引删除成功: {self._index}")
        except Exception as e:
            logger.error(f"索引删除失败: {e}", exc_info=True)
            raise MedicalQAStoreError(f"索引删除失败: {e}") from e

    # ── 查询 ──

    async def search(
        self,
        query: str,
        category: str | None = None,
        top_k: int = 5,
        source_includes: list[str] | None = None,
    ) -> list[dict]:
        """BM25 检索，支持 category 过滤和高亮"""
        try:
            must_clause = {
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "ask^2", "answer"],
                    "type": "best_fields",
                }
            }

            filter_clause = []
            if category:
                filter_clause.append({"term": {"category": category}})

            query_body = {"bool": {"must": [must_clause], "filter": filter_clause}}

            highlight = {
                "fields": {
                    "title": {"number_of_fragments": 1},
                    "ask": {"number_of_fragments": 1},
                    "answer": {"number_of_fragments": 1},
                }
            }

            kwargs = {}
            if source_includes:
                kwargs["_source"] = source_includes

            result = await self._es.search(
                index=self._index,
                query=query_body,
                size=top_k,
                highlight=highlight,
                **kwargs,
            )

            formatted_hits = []
            for hit in result["hits"]["hits"]:
                doc = {
                    "id": hit["_id"],
                    "score": hit["_score"],
                    **hit.get("_source", {}),
                }
                if "highlight" in hit:
                    doc["highlights"] = {
                        field: fragments[0]
                        for field, fragments in hit["highlight"].items()
                    }
                formatted_hits.append(doc)

            return formatted_hits

        except Exception as e:
            logger.error(f"ES 搜索异常: {e}", exc_info=True)
            return []

    # ── 管理 ──

    async def count(self) -> int:
        """返回索引中文档总数"""
        try:
            result = await self._es.count(index=self._index)
            return result["count"]
        except Exception as e:
            logger.error(f"查询文档总数失败: {e}", exc_info=True)
            raise MedicalQAStoreError(f"查询文档总数失败: {e}") from e
```

---

## 6. Index Mapping 设计

### 6.1 Mapping 配置

```python
settings = {
    "number_of_shards": 1,      # 44万条，1个分片足够
    "number_of_replicas": 0,    # 本地开发，单节点模式
    "analysis": {
        "analyzer": {
            "ik_analyzer": {
                "type": "custom",
                "tokenizer": "ik_max_word",
                "filter": ["lowercase"]
            }
        }
    }
}

mappings = {
    "properties": {
        "category":   {"type": "keyword"},           # 大科室，用于过滤
        "department": {"type": "keyword"},           # 细分科室，不过滤但可聚合
        "title":      {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
        "ask":        {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
        "answer":     {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
    }
}
```

### 6.2 字段类型说明

| 字段 | 类型 | 用途 |
|------|------|------|
| `category` | `keyword` | BERT 分类结果过滤，精确匹配，不走分词 |
| `department` | `keyword` | 细分科室，不过滤但可聚合统计 |
| `title` / `ask` / `answer` | `text` + IK 分词 | BM25 全文检索的核心字段 |

### 6.3 分词策略

- **索引时** `ik_max_word`："高血压" → 拆成"高/血压/高血压"，倒排索引更全，召回率高
- **搜索时** `ik_smart`："高血压怎么治疗" → 整体切分，减少噪音匹配，精确度高

### 6.4 单节点下 number_of_replicas 的选择

单节点模式下 `replicas=1` 会导致集群状态 yellow（无实际影响但有告噪），保持 `replicas=0` 即可。

---

## 7. create_index 实现

### 设计要点

- **幂等性**：可重复调用不报错，已存在则跳过
- **复用 `index_exists()`**：`create_index` 和 `delete_index` 都调用 `index_exists()` 做前置检查
- **异常处理**：捕获异常 → 记日志（含堆栈）→ 抛 `MedicalQAStoreError`

### 实现顺序

```
index_exists()  ← 基础方法
   ↑              ↑
create_index()  delete_index()
```

---

## 8. 数据入库脚本设计

### 8.1 关键决策：入库脚本与 MedicalQAStore 分离

数据入库是一次性的离线 ETL 任务，不是运行时操作。因此：

- **`scripts/ingest_data.py`**：自包含的入库脚本，不依赖 store.py
- **`src/es/store.py`**：纯运行时接口（search / count / delete）

### 8.2 架构来源：pipeline_code.py 模板

参考 `think_code/pipeline_code.py` 的流式管道模式，分为 6 个模块：

```
csv_reader()  →  validate_and_transform_pipeline()  →  GenericElasticRepository.bulk_insert_stream()
  (数据源)             (校验 + 转换，不碰网络)                   (纯写入，不懂业务)
```

### 8.3 核心设计

#### 流式处理（async_streaming_bulk vs async_bulk）

```python
# async_bulk：等全部写完才返回结果
success, errors = await async_bulk(...)

# async_streaming_bulk：逐条返回结果，边写边统计
async for success, info in async_streaming_bulk(...):
    if success: stats.es_success += 1
    else: stats.record_es_error(info)
```

流式处理，内存永远只保留 chunk_size 的数据，44 万条不会撑爆内存。

#### 写入性能调优

```python
# 写入前：关副本、放宽刷新
await self._es.indices.put_settings(body={
    "index": {"refresh_interval": "30s", "number_of_replicas": 0}
})

# 写入后：恢复标准设置
await self._es.indices.put_settings(body={
    "index": {"refresh_interval": "1s", "number_of_replicas": 0}
})
```

批量写入时，ES 频繁 refresh 和副本同步是性能杀手。临时关掉可以提速 3~5 倍。

#### PipelineStats 统计对象

不把统计数据塞到方法返回值里，而是用一个独立的统计对象在管道各阶段间传递：

```python
class PipelineStats:
    def __init__(self):
        self.total_read: int = 0
        self.validation_errors: int = 0
        self.es_success: int = 0
        self.es_failed: int = 0
        self.error_samples: List[dict] = []
```

#### Pydantic 文档校验

```python
class MedicalQADocument(BaseModel):
    category: str           # 大科室（从文件名提取）
    department: str         # 细分科室（CSV 内字段）
    title: str = Field(..., min_length=1)
    ask: str                # 允许空字符串（数据中"无"会被规范化为""）
    answer: str
```

#### 数据清洗："无" → 空字符串

```python
def _process_single_doc(doc: dict, stats: PipelineStats) -> Union[dict, None]:
    try:
        # 数据清洗：将"无"规范化为空字符串
        if doc.get("ask") == "无":
            doc["ask"] = ""
        validated = MedicalQADocument(**doc)
        return {"_op_type": "index", "_source": validated.model_dump()}
    except ValidationError as e:
        ...
```

#### chunk_size 选择

| chunk_size | 单次请求大小 | 说明 |
|------------|-------------|------|
| 500 | ~750KB | 网络往返多，总吞吐低 |
| 1000 | ~1.5MB | 保守选择 |
| **2000** | **~3MB** | **推荐默认值** |
| 5000 | ~7.5MB | 1GB 堆下可能 GC 压力 |

ES 官方建议单次 bulk 请求不超过 100MB。2000 是在 1GB 堆内存约束下的合理默认值。

### 8.4 完整入库脚本

```python
# scripts/ingest_data.py

"""
医疗问答数据入库脚本（自包含）

从 data/ 目录读取 6 个科室的 CSV 文件，经 Pydantic 校验后流式写入 ES。
基于 pipeline_code.py 模板，适配实际字段：category / department / title / ask / answer。

用法:
    python -m scripts.ingest_data
"""
import csv
import json
import asyncio
from pathlib import Path
from typing import AsyncIterable, Iterable, Union, Dict, Any, List

from pydantic import BaseModel, Field, ValidationError
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_streaming_bulk

from src.base import setup_logger, load_config

logger = setup_logger("IngestData")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

INDEX_SETTINGS = {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "analysis": {
        "analyzer": {
            "ik_analyzer": {
                "type": "custom",
                "tokenizer": "ik_max_word",
                "filter": ["lowercase"],
            }
        }
    },
}

INDEX_MAPPINGS = {
    "properties": {
        "category": {"type": "keyword"},
        "department": {"type": "keyword"},
        "title": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
        "ask": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
        "answer": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
    }
}


# ==========================================
# 模块 1: 业务数据模型 (Models)
# ==========================================
class MedicalQADocument(BaseModel):
    """医疗问答文档模型，字段对齐 CSV 实际结构"""
    category: str
    department: str
    title: str = Field(..., min_length=1)
    ask: str
    answer: str


# ==========================================
# 模块 2: 管道状态计数器 (Context/Stats)
# ==========================================
class PipelineStats:
    """用于在管道各个阶段间传递和累加统计状态的轻量对象"""

    def __init__(self):
        self.total_read: int = 0
        self.validation_errors: int = 0
        self.es_success: int = 0
        self.es_failed: int = 0
        self.error_samples: List[dict] = []

    def record_es_error(self, error_info: dict):
        self.es_failed += 1
        if len(self.error_samples) < 20:
            self.error_samples.append(error_info)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_read": self.total_read,
            "success": self.es_success,
            "es_failed": self.es_failed,
            "validation_errors": self.validation_errors,
            "error_samples": self.error_samples[:10],
        }


# ==========================================
# 模块 3: 管道清洗与转化层 (Transformers)
# ==========================================
async def validate_and_transform_pipeline(
    raw_documents: Union[Iterable[dict], AsyncIterable[dict]],
    stats: PipelineStats,
) -> AsyncIterable[dict]:
    """
    【清洗管道】
    流式读取原始数据 -> Pydantic 校验 -> 转化为 ES 标准 Action 字典。
    不知道 ES 索引叫什么，不参与网络 IO。
    """
    if isinstance(raw_documents, AsyncIterable):
        async for doc in raw_documents:
            stats.total_read += 1
            action = _process_single_doc(doc, stats)
            if action:
                yield action
    else:
        for doc in raw_documents:
            stats.total_read += 1
            action = _process_single_doc(doc, stats)
            if action:
                yield action


def _process_single_doc(doc: dict, stats: PipelineStats) -> Union[dict, None]:
    """单条数据的校验转化"""
    try:
        # 数据清洗：将"无"规范化为空字符串
        if doc.get("ask") == "无":
            doc["ask"] = ""
        validated = MedicalQADocument(**doc)
        return {
            "_op_type": "index",
            "_source": validated.model_dump(),
        }
    except ValidationError as e:
        stats.validation_errors += 1
        if stats.validation_errors <= 5:
            logger.warning(f"第 {stats.total_read} 条数据校验失败: {e}")
        return None
    except Exception as e:
        stats.validation_errors += 1
        logger.error(f"第 {stats.total_read} 条数据未知异常: {e}", exc_info=True)
        return None


# ==========================================
# 模块 4: ES 基础服务层 (Sinks/Repositories)
# ==========================================
class GenericElasticRepository:
    """
    【ES 通用数据桶】
    只认标准 Action 字典，不知道什么是 MedicalQADocument。
    负责：批量写入、网络异常控制、ES 性能调优。
    """

    def __init__(self, es_client: AsyncElasticsearch, index_name: str):
        self._es = es_client
        self._index = index_name

    async def _update_index_settings(self, refresh_interval: str, number_of_replicas: int):
        """临时调整索引设置以优化写入性能"""
        try:
            await self._es.indices.put_settings(
                index=self._index,
                body={
                    "index": {
                        "refresh_interval": refresh_interval,
                        "number_of_replicas": number_of_replicas,
                    }
                },
            )
        except Exception as e:
            logger.warning(f"调整索引设置失败(可忽略): {e}")

    async def bulk_insert_stream(
        self,
        actions: AsyncIterable[dict],
        stats: PipelineStats,
        chunk_size: int = 2000,
        log_interval: int = 20000,
    ) -> None:
        """接收动作流，高效安全地写入 ES"""

        async def _index_binding_stream():
            async for action in actions:
                action["_index"] = self._index
                yield action

        # 写入前调优
        await self._update_index_settings(refresh_interval="30s", number_of_replicas=0)

        try:
            async for success, info in async_streaming_bulk(
                client=self._es.options(request_timeout=120),
                actions=_index_binding_stream(),
                chunk_size=chunk_size,
                raise_on_error=False,
            ):
                if success:
                    stats.es_success += 1
                else:
                    stats.record_es_error(info)

                total_processed = stats.es_success + stats.es_failed
                if total_processed % log_interval == 0:
                    logger.info(
                        f"进度: 已提交 {total_processed} 条 | "
                        f"成功: {stats.es_success} | 失败: {stats.es_failed} | "
                        f"校验失败: {stats.validation_errors}"
                    )
        except Exception as e:
            logger.error(f"ES 流式写入崩溃: {e}", exc_info=True)
        finally:
            # 恢复索引标准设置
            await self._update_index_settings(refresh_interval="1s", number_of_replicas=0)


# ==========================================
# 模块 5: 数据源 (Sources)
# ==========================================
def csv_reader(data_dir: Path = DATA_DIR) -> Iterable[dict]:
    """
    【水机端】逐行读取 data_dir 下所有 CSV 文件，yield 原始 dict。
    category 从文件名提取（如 "儿科.csv" -> "儿科"）。
    """
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        logger.error(f"未找到 CSV 文件: {data_dir}")
        return

    for csv_file in csv_files:
        category = csv_file.stem
        logger.info(f"读取文件: {csv_file.name} (category={category})")

        with open(csv_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield {
                    "category": category,
                    "department": row["department"],
                    "title": row["title"],
                    "ask": row["ask"],
                    "answer": row["answer"],
                }


# ==========================================
# 模块 6: 业务运行/调度脚本
# ==========================================
async def create_index(es_client: AsyncElasticsearch, index_name: str) -> None:
    """创建索引（幂等），已存在则跳过"""
    if await es_client.indices.exists(index=index_name):
        logger.info(f"索引已存在: {index_name}，跳过创建")
        return
    await es_client.indices.create(index=index_name, settings=INDEX_SETTINGS, mappings=INDEX_MAPPINGS)
    logger.info(f"索引创建成功: {index_name}")


async def main():
    config = load_config()
    es_config = config.elasticsearch

    # 1. 初始化 ES 客户端
    es_url = f"http://{es_config.host}:{es_config.port}"
    es_client = AsyncElasticsearch(es_url)
    if not await es_client.ping():
        logger.error(f"无法连接 ES: {es_url}")
        return
    logger.info(f"ES 连接成功: {es_url}")

    # 2. 创建索引（幂等）
    await create_index(es_client, es_config.index_name)

    # 3. 初始化统计与仓储
    stats = PipelineStats()
    es_repo = GenericElasticRepository(es_client, index_name=es_config.index_name)

    # 4. 获取原始数据流
    raw_stream = csv_reader()

    # 5. 接入清洗管道
    clean_action_stream = validate_and_transform_pipeline(raw_stream, stats)

    # 6. 写入 ES
    logger.info("数据管道启动...")
    await es_repo.bulk_insert_stream(clean_action_stream, stats, chunk_size=2000)

    # 7. 打印最终报告
    logger.info("管道运行结束，报告:")
    print(json.dumps(stats.to_dict(), indent=4, ensure_ascii=False))

    # 8. 关闭连接
    await es_client.close()


if __name__ == "__main__":
    asyncio.run(main())
```

### 8.5 入库结果

```
total_read: 443,933
success: 387,327
es_failed: 0
validation_errors: 56,606  （主要原因是 ask="无" 被 min_length=2 拦截，已修复）
```

---

## 9. search 实现

### 9.1 ES 查询 DSL 结构（bool 查询）

```python
query_body = {
    "bool": {
        "must": [...],    # 全文检索，参与 BM25 打分
        "filter": [...]   # 精确过滤，不打分，有 bitset 缓存
    }
}
```

类比 MySQL：

```sql
SELECT * FROM medical_qa
WHERE MATCH(title, ask, answer) AGAINST('高血压怎么治')  -- must
  AND category = '内科'                                    -- filter
ORDER BY relevance_score DESC
LIMIT 5;
```

### 9.2 bool 查询四个子句

| 子句 | 含义 | 参与 BM25 打分 | 等价 SQL |
|------|------|:-:|----------|
| `must` | 必须匹配 | ✅ | `WHERE ... AND ...` |
| `filter` | 必须匹配 | ❌ | `WHERE ... AND ...` |
| `should` | 可选匹配（加分） | ✅ | 无直接等价 |
| `must_not` | 必须不匹配 | ❌ | `WHERE NOT ...` |

**`must` 和 `filter` 的区别**：`filter` 不打分是因为 category 过滤没有"相关性"概念。ES 对 keyword 过滤做了 bitset 缓存，第二次查询几乎零开销。**filter 先于 must 执行**，先缩小候选集再打分。

### 9.3 字段权重

| 字段 | 权重 | 理由 |
|------|------|------|
| `title` | ^3 | 标题最能概括问题 |
| `ask` | ^2 | 患者提问，语义接近用户输入 |
| `answer` | ^1 | 医生回答，可能包含关键词但不一定最相关 |

### 9.4 multi_match 类型选择

| 类型 | 行为 | 适合场景 |
|------|------|----------|
| **`best_fields`** | 按最高分的字段排序 | **采用**，"哪个字段最相关就用哪个" |
| `most_fields` | 所有字段分数求和 | 匹配字段越多越好 |
| `cross_fields` | 把多个字段当一个大字段 | 关键词跨字段分布 |

### 9.5 高亮（Highlight）

**高亮结果不覆盖原字段，单独放 `highlights`**：

```python
doc = {
    "id": "xxx",
    "score": 12.5,
    "category": "内科",
    "title": "高血压患者能吃党参吗？",           # 原文，干净
    "highlights": {
        "title": "<em>高血压</em>患者能吃党参吗？"  # 高亮，带标签
    }
}
```

好处：前端按需使用，原文不受高亮标签污染。

### 9.6 异常处理策略

search 是面向用户的查询，采用**优雅降级**：

```python
except Exception as e:
    logger.error(f"ES 搜索异常: {e}", exc_info=True)
    return []  # 返回空列表，不中断对话流程
```

与 create_index/delete_index 不同（那些是基础设施操作，需要 raise）。

### 9.7 _source 过滤

支持 `source_includes` 参数，只返回必要字段，减少网络传输：

```python
results = await store.search("高血压", source_includes=["title", "answer"])
```

---

## 10. 性能分析

### 10.1 实测数据

```
分类搜索（带 category 过滤）: 20~24 ms
全量搜索（不带过滤）: 20~23 ms
缓存命中: 0 ms
```

### 10.2 为什么过滤和全量搜索耗时几乎一样？

**44 万条对 ES 来说太小了，filter 的优化体现不出来。**

- BM25 打分是瓶颈，filter 过滤开销极小
- 从 44 万缩小到 8 万，省下的时间被网络往返和 JSON 序列化淹没
- filter 优化真正发挥作用的场景：数据量上千万、上亿

### 10.3 为什么有 0ms 结果？

**ES Node Query Cache 命中**。ES 自动缓存最近的查询结果，连续同一 query 搜索会命中缓存。

### 10.4 结论

| 现象 | 原因 |
|------|------|
| 过滤 ≈ 全量 | 44 万太小，BM25 打分是瓶颈 |
| 0.00ms | ES Node Query Cache 命中 |
| 后续查询更快 | OS 文件缓存 + ES 缓存叠加 |

search 耗时稳定在 20~25ms，已经很快。真正影响用户体验的是 BERT 分类的耗时，不是 ES 检索。

---

## 11. 最终文件结构

```
medical-RAG/
├── src/
│   ├── config.toml              # 新增 [elasticsearch] 段
│   ├── base/
│   │   ├── config.py            # 新增 ElasticsearchConfig
│   │   ├── logger.py            # 未改动
│   │   └── __init__.py          # 未改动
│   ├── es/
│   │   ├── store.py             # MedicalQAStore（运行时接口）
│   │   └── __init__.py          # 导出 MedicalQAStore, MedicalQAStoreError
│   └── logs/
│       └── app.log
├── scripts/
│   └── ingest_data.py           # 自包含入库脚本（6 模块管道）
├── data/
│   ├── 儿科.csv
│   ├── 内科.csv
│   ├── 外科.csv
│   ├── 妇产科.csv
│   ├── 男科.csv
│   └── 肿瘤科.csv
├── es-plugins/
│   └── ik/                      # IK 分词器插件
└── docker-compose.yml
```

### 职责划分

| 文件 | 职责 | 依赖 |
|------|------|------|
| `src/es/store.py` | 运行时 ES 接口（search/count/delete） | elasticsearch, src.base |
| `scripts/ingest_data.py` | 离线数据入库（自包含） | elasticsearch, pydantic, src.base |
| `src/base/config.py` | 配置模型 | pydantic |
| `src/base/logger.py` | 日志配置 | logging |

### 运行时 vs 离线的边界

| 操作 | 归属 | 原因 |
|------|------|------|
| `create_index` | store.py | 初始化时调用 |
| `search` | store.py | 运行时核心功能 |
| `count` | store.py | 运行时管理 |
| `delete_index` | store.py | 运行时管理 |
| bulk_insert | ingest_data.py | 一次性离线任务 |
| CSV 读取 | ingest_data.py | 数据源格式是脚本的事 |
| 流式管道 | ingest_data.py | ETL 逻辑 |
