# Redis Stack 向量搜索实现记录

> 从需求分析到实现完成的完整过程

---

## 目录

- [1. 背景与需求](#1-背景与需求)
- [2. 技术选型](#2-技术选型)
- [3. 架构设计](#3-架构设计)
- [4. 实现过程](#4-实现过程)
- [5. 遇到的问题与解决方案](#5-遇到的问题与解决方案)
- [6. 关键技术点](#6-关键技术点)
- [7. 测试验证](#7-测试验证)
- [8. 最终结论](#8-最终结论)

---

## 1. 背景与需求

### 1.1 核心问题

```
问题问法不同导致 Redis 缓存失效:

    "高血压怎么治疗"
    "高血压如何治疗"        ← 同一个问题，不同表述
    "怎么治疗高血压"
    "高血压  怎么治疗"      ← 多余空格
    "高血压怎么治疗？"      ← 标点符号

精确匹配:
    │
    ├─ key: "高血压怎么治疗"
    ├─ key: "高血压如何治疗"    ← 不同 key！
    └─ 结果: 缓存命中率低
```

### 1.2 需求分析

| 需求 | 说明 |
|------|------|
| **语义相似度匹配** | 解决"问题问法不同"的问题 |
| **标量过滤** | 支持按科室分类过滤 |
| **高性能** | 延迟 < 10ms |
| **简单部署** | 使用现有 Redis |

### 1.3 方案对比

| 方案 | 解决程度 | 复杂度 | 说明 |
|------|----------|--------|------|
| **精确匹配** | ❌ 差 | 低 | 无法解决 |
| **归一化** | ⚠️ 一般 | 低 | 解决格式差异 |
| **向量搜索** | ✅ 好 | 中 | 解决语义差异 |

### 1.4 归一化的局限

```python
# 归一化能解决的:
"高血压怎么治疗？"  → "高血压怎么治疗"  ✅
"高血压  怎么治疗"  → "高血压 怎么治疗"  ✅

# 归一化无法解决的:
"高血压怎么治疗"    → "高血压怎么治疗"  ✅
"高血压如何治疗"    → "高血压如何治疗"  ❌ 仍然不同！
"怎么治疗高血压"    → "怎么治疗高血压"  ❌ 仍然不同！
```

---

## 2. 技术选型

### 2.1 Redis Stack 介绍

```
Redis Stack 组件:
    │
    ├─ RediSearch: 全文搜索 + 向量搜索
    ├─ RedisJSON: JSON 数据类型
    ├─ RedisTimeSeries: 时间序列
    └─ RedisGraph: 图数据库

向量搜索应用场景:
    │
    ├─ 语义相似问题匹配
    ├─ 推荐系统
    └─ 相似文档检索
```

### 2.2 向量搜索 vs 专业向量数据库

| 维度 | Redis Stack | Milvus |
|------|-------------|--------|
| **定位** | 缓存 + 向量搜索 | 专业向量数据库 |
| **性能** | ✅ 快 (<1ms) | ⚠️ 中 (5-20ms) |
| **精度** | ⚠️ 中等 | ✅ 高 |
| **部署** | ✅ 简单 | ⚠️ 复杂 |
| **功能** | ⚠️ 基础 | ✅ 丰富 |
| **成本** | ✅ 低 | ⚠️ 高 |

### 2.3 选择 Redis Stack 的理由

```
选择 Redis Stack 的理由:
    │
    ├─ 数据量小 (几千到几万条)
    ├─ 目的是缓存，不是检索
    ├─ 解决语义相似度问题
    ├─ 部署简单，性能好
    └─ 已有 Redis，无需额外部署
```

---

## 3. 架构设计

### 3.1 决策过程

#### 决策点 1: 是否需要新建类？

| 方案 | 实现 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|----------|
| **直接扩展** | 在类中添加方法 | 简单、统一 | 类可能变大 | 功能相关 |
| **继承** | 创建子类 | 职责分离 | 增加复杂度 | 功能差异大 |
| **组合** | 包含其他类实例 | 灵活 | 需要管理多个对象 | 功能独立 |

**决定**: 直接扩展 RedisCache

**理由**:
1. 向量搜索和普通缓存都是操作 Redis
2. 功能相关，不是独立的
3. 继承会增加不必要的复杂度

#### 决策点 2: 是否需要继承？

**决定**: 不需要继承

**理由**:
1. 向量搜索和普通缓存都是操作 Redis
2. 都使用同一个连接池
3. 功能相关，不是独立的

### 3.2 最终架构

```python
class RedisCache:
    """Redis 异步缓存封装 - 支持向量搜索"""
    
    # ── 连接管理 ──
    async def connect()      # 创建连接池
    async def close()        # 关闭连接池
    
    # ── 基本操作 ──
    async def get(key)       # 获取缓存
    async def set(key, value, ttl)  # 设置缓存
    async def delete(key)    # 删除缓存
    async def exists(key)    # 检查存在
    
    # ── 向量搜索 ──
    async def create_vector_index(index_name, prefix, dim)  # 创建索引
    async def vector_search(index_name, embedding, top_k, filters)  # 向量搜索
    async def set_with_embedding(key, question, result, embedding, category, ttl)  # 存储
```

### 3.3 Redis 存储结构

```
Redis Hash 结构:
    │
    ├─ key: "qa:abc123"
    ├─ field: "question" → "高血压怎么治疗"
    ├─ field: "result" → '{"intent": {...}, "results": [...]}'
    ├─ field: "embedding" → bytes (向量数据)
    └─ field: "category" → "内科"
```

### 3.4 完整流程

```
用户问题: "高血压怎么治疗"
    │
    ├─→ BERT 意图识别 → {"label": "内科", "confidence": 0.99}
    │
    ├─→ embedding 模型 → [0.1, 0.2, ..., 0.7]
    │
    └─→ Redis 混合查询:
            @category:{内科}=>[KNN 5 @embedding $vec AS score]
            │
            ├─ 命中 → 返回缓存结果
            └─ 未命中 → ES 检索 → 缓存结果 → 返回
```

---

## 4. 实现过程

### 4.1 添加依赖导入

```python
import numpy as np
from redis.commands.search.field import VectorField, TextField, TagField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError
```

### 4.2 create_vector_index 实现

```python
async def create_vector_index(
    self,
    index_name: str,
    prefix: str = "qa:",
    dim: int = 768
) -> bool:
    """创建向量索引"""
    try:
        # 1. 定义 Schema
        schema = (
            TextField("question"),  # 普通文本字段
            TagField("category"),   # 标签字段 (用于过滤)
            VectorField("embedding",  # 向量字段
                algorithm="HNSW",
                attributes={
                    "TYPE": "FLOAT32",
                    "DIM": dim,
                    "DISTANCE_METRIC": "COSINE"
                }
            )
        )

        # 2. 创建索引
        await self._redis.ft(index_name).create_index(
            schema,
            definition=IndexDefinition(
                prefix=[prefix],
                index_type=IndexType.HASH
            )
        )

        logger.info(f"向量索引创建成功: {index_name}")
        return True

    except ResponseError as e:
        if "Index already exists" in str(e):
            logger.debug(f"向量索引已存在: {index_name}")
            return True
        logger.error(f"创建向量索引失败: {e}", exc_info=True)
        raise RedisCacheError(f"创建向量索引失败: {e}") from e
    except Exception as e:
        logger.error(f"创建向量索引失败: {e}", exc_info=True)
        raise RedisCacheError(f"创建向量索引失败: {e}") from e
```

### 4.3 set_with_embedding 实现

```python
async def set_with_embedding(
    self,
    key: str,
    question: str,
    result: dict,
    embedding: list[float],
    category: str | None = None,
    ttl: int | None = None
) -> bool:
    """存储带 embedding 的缓存"""
    try:
        # 1. 转换 embedding 为 bytes
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

        # 2. 序列化 result 为 JSON
        result_json = json.dumps(result, ensure_ascii=False)

        # 3. 构建数据
        data = {
            "question": question,
            "result": result_json,
            "embedding": embedding_bytes
        }

        # 4. 添加 category (如果有)
        if category:
            data["category"] = category

        # 5. 存储数据
        await self._redis.hset(key, mapping=data)

        # 6. 设置 TTL
        if ttl:
            await self._redis.expire(key, ttl)

        return True

    except Exception as e:
        logger.error(f"Redis SET_WITH_EMBEDDING 失败 (key={key}): {e}", exc_info=True)
        raise RedisCacheError(f"Redis SET_WITH_EMBEDDING 失败: {e}") from e
```

### 4.4 vector_search 实现

```python
async def vector_search(
    self,
    index_name: str,
    embedding: list[float],
    top_k: int = 1,
    filters: dict[str, str] | None = None
) -> list[dict]:
    """向量搜索 (支持标量过滤)"""
    try:
        # 1. 构建过滤条件
        if filters:
            filter_str = " ".join(f"@{k}:{{{v}}}" for k, v in filters.items())
            query_str = f"{filter_str}=>[KNN {top_k} @embedding $vec AS score]"
        else:
            query_str = f"*=>[KNN {top_k} @embedding $vec AS score]"

        # 2. 构建查询
        query = (
            Query(query_str)
            .sort_by("score")
            .return_fields("question", "result", "score")
            .dialect(2)
        )

        # 3. 转换 embedding 为 bytes
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

        # 4. 执行向量搜索
        results = await self._redis.ft(index_name).search(
            query,
            query_params={"vec": embedding_bytes}
        )

        # 5. 解析结果
        parsed_results = []
        for doc in results.docs:
            try:
                parsed_results.append({
                    "id": doc.id,
                    "question": doc.question,
                    "result": json.loads(doc.result),
                    "score": float(doc.score)
                })
            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning(f"解析结果失败: {e}")
                continue
        return parsed_results

    except Exception as e:
        logger.error(f"Redis VECTOR_SEARCH 失败: {e}", exc_info=True)
        raise RedisCacheError(f"Redis VECTOR_SEARCH 失败: {e}") from e
```

---

## 5. 遇到的问题与解决方案

### 5.1 Query 未导入

**问题**: `NameError: name 'Query' is not defined`

**原因**: `vector_search` 方法中使用了 `Query`，但没有导入

**解决**: 添加导入
```python
from redis.commands.search.query import Query
```

### 5.2 TagField 未导入

**问题**: `NameError: name 'TagField' is not defined`

**原因**: `create_vector_index` 方法中使用了 `TagField`，但没有导入

**解决**: 修改导入
```python
from redis.commands.search.field import VectorField, TextField, TagField
```

### 5.3 category 字段缺失

**问题**: 向量搜索返回 0 个结果

**原因**: 索引 schema 中没有 `category` 字段

**解决**: 添加 `TagField("category")` 到 schema
```python
schema = (
    TextField("question"),
    TagField("category"),  # 添加这一行
    VectorField("embedding", ...)
)
```

### 5.4 过滤语法错误

**问题**: 向量搜索返回 0 个结果

**原因**: TagField 的过滤语法不同
- 错误: `@category:内科`
- 正确: `@category:{内科}`

**解决**: 修改过滤语法
```python
# 错误
filter_str = " ".join(f"@{k}:{v}" for k, v in filters.items())

# 正确
filter_str = " ".join(f"@{k}:{{{v}}}" for k, v in filters.items())
```

### 5.5 二进制解码失败

**问题**: `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xcd`

**原因**: `decode_responses=True` 导致 embedding 二进制数据被强制 UTF-8 解码

**解决**: 使用 `return_fields` 排除 embedding 字段
```python
query = query.return_fields("question", "result", "score")
```

### 5.6 浮点精度问题

**问题**: 分数是负数 (`score: -8.34465026855e-07`)

**原因**: 浮点精度问题
- 理论值: `1.0 - 1.0 = 0.0`
- 实际值: `1.0 - 1.000000000834 = -0.000000000834`

**结论**: 这是正常的，表示"几乎完全相同"

---

## 6. 关键技术点

### 6.1 TagField 过滤语法

```
TagField 过滤语法:
    │
    ├─ 正确: @category:{内科}
    └─ 错误: @category:内科

示例:
    │
    ├─ 单条件: @category:{内科}
    ├─ 多条件: @category:{内科} @department:{心内科}
    └─ 无过滤: *
```

### 6.2 return_fields 使用

```python
# 只返回需要的字段，排除 embedding 二进制数据
query = query.return_fields("question", "result", "score")

# 对于二进制字段，使用 decode_field=False
query = query.return_field("embedding", decode_field=False)
```

### 6.3 COSINE 距离计算

```
COSINE 距离:
    │
    ├─ 公式: distance = 1 - cosine_similarity
    ├─ cosine_similarity 范围: [-1, 1]
    └─ distance 范围: [0, 2]

分数含义:
    │
    ├─ 0: 完全相同
    ├─ 接近 0: 非常相似
    ├─ 1: 正交 (不相似)
    └─ 2: 完全相反
```

### 6.4 查询语法

```python
# 无过滤
Query("*=>[KNN 5 @embedding $vec AS score]")

# 有过滤
Query("@category:{内科}=>[KNN 5 @embedding $vec AS score]")

# 带排序和分页
Query("*=>[KNN 5 @embedding $vec AS score]")
    .sort_by("score")
    .return_fields("question", "result", "score")
    .paging(0, 5)
    .dialect(2)
```

---

## 7. 测试验证

### 7.1 测试代码

```python
async def main():
    config = load_config()

    # 创建实例并连接
    cache = RedisCache(config.redis)
    await cache.connect()

    # 创建向量索引
    await cache.create_vector_index(
        index_name="qa_semantic_cache",
        prefix="qa:",
        dim=768
    )

    # 存储带 embedding 的缓存
    await cache.set_with_embedding(
        key="qa:test1",
        question="高血压怎么治疗",
        result={"intent": {"label": "内科"}, "results": [{"title": "高血压治疗"}]},
        embedding=[0.1] * 768,
        category="内科",
        ttl=3600
    )

    # 向量搜索
    results = await cache.vector_search(
        index_name="qa_semantic_cache",
        embedding=[0.1] * 768,
        top_k=5,
        filters={"category": "内科"}
    )

    print(f"找到 {len(results)} 个相似结果")
    for result in results:
        print(f"  - {result['question']} (score: {result['score']})")

    # 关闭连接
    await cache.close()
```

### 7.2 测试结果

```
向量索引创建成功
带 embedding 的缓存存储成功
找到 1 个相似结果
  - 高血压怎么治疗 (score: -8.34465026855e-07)
```

### 7.3 Redis 命令行验证

```bash
# 检查索引
redis-cli -a "password" ft.info qa_semantic_cache

# 检查数据
redis-cli -a "password" hgetall qa:test1

# 测试向量搜索
redis-cli -a "password" ft.search qa_semantic_cache "*=>[KNN 1 @embedding \$vec AS score]" PARAMS 2 vec "..." DIALECT 2
```

---

## 8. 最终结论

### 8.1 实现总结

| 方法 | 用途 | 状态 |
|------|------|------|
| `create_vector_index()` | 创建向量索引 | ✅ 完成 |
| `vector_search()` | 向量搜索 (支持标量过滤) | ✅ 完成 |
| `set_with_embedding()` | 存储带 embedding 的缓存 | ✅ 完成 |

### 8.2 使用方式

```python
# 1. 创建向量索引
await cache.create_vector_index(
    index_name="qa_semantic_cache",
    prefix="qa:",
    dim=768
)

# 2. 存储带 embedding 的缓存
await cache.set_with_embedding(
    key="qa:abc123",
    question="高血压怎么治疗",
    result={"intent": {"label": "内科"}, "results": [...]},
    embedding=[0.1, 0.2, ...],
    category="内科",
    ttl=3600
)

# 3. 向量搜索
results = await cache.vector_search(
    index_name="qa_semantic_cache",
    embedding=[0.1, 0.2, ...],
    top_k=5,
    filters={"category": "内科"}
)
```

### 8.3 核心价值

```
Redis Stack 向量搜索的核心价值:
    │
    ├─ 不是: 大规模向量检索
    ├─ 而是: 解决"问题问法不同"的问题
    └─ 目的: 提高缓存命中率
```

### 8.4 适用场景

| 场景 | 是否适用 | 说明 |
|------|----------|------|
| **数据量小** (<10 万条) | ✅ 适用 | 性能好 |
| **延迟要求高** (<10ms) | ✅ 适用 | Redis 原生支持 |
| **精度要求中等** | ✅ 适用 | 够用 |
| **数据量大** (>100 万条) | ⚠️ 考虑 Milvus | 专业向量数据库 |

---

> 文档生成时间: 2026-06-28
> 环境: Redis Stack + redis-py
