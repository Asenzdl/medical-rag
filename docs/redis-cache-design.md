# Redis 缓存封装设计文档

> 从零开始构建生产级 Redis 缓存封装的完整记录

---

## 目录

- [1. 背景与目标](#1-背景与目标)
- [2. 生产环境的核心认知](#2-生产环境的核心认知)
- [3. 设计决策过程](#3-设计决策过程)
- [4. 最终实现](#4-最终实现)
- [5. 遇到的问题与解决方案](#5-遇到的问题与解决方案)
- [6. 代码演进历史](#6-代码演进历史)
- [7. 关键总结](#7-关键总结)

---

## 1. 背景与目标

### 1.1 项目背景

医疗问答系统需要缓存 BERT 意图识别结果和 ES 检索结果，以减少 GPU 计算和 ES 查询。

### 1.2 缓存流程设计

```
用户问题: "高血压怎么治疗"
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: BERT 意图识别 (~3ms)                                   │
│                                                                 │
│  输入: "高血压怎么治疗"                                         │
│  输出: {"label": "内科", "confidence": 0.9979}                  │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: embedding 模型 (~10ms)                                 │
│                                                                 │
│  输入: "高血压怎么治疗"                                         │
│  输出: [0.1, 0.2, ..., 0.7]                                    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: Redis 混合查询 (<1ms)                                  │
│                                                                 │
│  @category:{内科}=>[KNN 5 @embedding $vec AS score]             │
│  命中 → 返回                                                    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼ 未命中
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: ES 检索 (10-50ms)                                      │
│                                                                 │
│  query: "高血压怎么治疗"                                        │
│  filter: category = "内科"                                      │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: 缓存结果 → Redis                                       │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 目标

- 构建生产级 Redis 缓存封装
- 支持连接池管理
- 支持单例模式
- 支持超时机制
- 代码简洁、易维护

---

## 2. 生产环境的核心认知

### 2.1 连接池是生产标配

**问题**: 为什么需要连接池？

**答案**:

```
没有连接池:
    │
    ├─ 每次请求: 创建连接 → 执行命令 → 关闭连接
    ├─ 开销: TCP 握手 + 认证 (~1-5ms)
    └─ 问题: 高并发时连接数爆炸

有连接池:
    │
    ├─ 启动时: 创建 N 个连接
    ├─ 每次请求: 从池中取连接 → 执行命令 → 还回连接
    ├─ 开销: 几乎为 0
    └─ 优点: 连接数可控、性能高
```

**性能对比**:

| 方式 | 每次请求开销 | 1000 并发连接数 | 适用场景 |
|------|-------------|----------------|----------|
| **无连接池** | 1-5ms | 1000 | ❌ 不推荐 |
| **有连接池** | <0.1ms | 20 (复用) | ✅ 生产标配 |

### 2.2 decode_responses=True 必须设置

**问题**: 为什么必须设置 `decode_responses=True`？

**答案**:

```python
# 没有 decode_responses=True
value = redis.get("key")  # 返回 bytes: b'{"name": "test"}'
data = json.loads(value.decode('utf-8'))  # 需要手动解码

# 有 decode_responses=True
value = redis.get("key")  # 返回 str: '{"name": "test"}'
data = json.loads(value)  # 直接使用
```

### 2.3 socket_keepalive 防止防火墙切断

**问题**: 为什么需要 `socket_keepalive=True`？

**答案**:

```
长连接可能被防火墙切断:
    │
    ├─ 防火墙会定期清理空闲连接
    ├─ 如果连接长时间没有数据传输，会被切断
    └─ socket_keepalive 会定期发送心跳包，保持连接活跃
```

### 2.4 单例模式确保全局唯一连接池

**问题**: 为什么需要单例模式？

**答案**:

```
没有单例模式:
    │
    ├─ 每次创建 RedisCache 实例都会创建新连接池
    ├─ 连接数会不断增长
    └─ 最终导致连接数爆炸

有单例模式:
    │
    ├─ 全局只有一个 RedisCache 实例
    ├─ 全局只有一个连接池
    └─ 连接数可控
```

### 2.5 不需要手动 release 连接

**问题**: 为什么不需要手动 release 连接？

**答案**:

```
redis-py 连接池的工作原理:

    ┌─────────────────────────────────────────────────────────────┐
    │  连接池 (ConnectionPool)                                    │
    │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                  │
    │  │conn1│ │conn2│ │conn3│ │conn4│ │conn5│                  │
    │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘                  │
    └─────────────────────────────────────────────────────────────┘
           │                          ▲
           │ 1. 获取连接               │ 3. 自动归还
           ▼                          │
    ┌─────────────────────────────────────────┐
    │  redis.Redis(connection_pool=pool)      │
    │                                         │
    │  result = await pool.get("key")         │
    │           │                             │
    │           ▼                             │
    │  2. 执行命令 → 返回结果                  │
    │           │                             │
    │           ▼                             │
    │  3. 自动归还连接到连接池                  │
    └─────────────────────────────────────────┘

redis-py 内部实现 (简化版):

    class Redis:
        async def execute_command(self, *args, **kwargs):
            # 1. 从连接池获取连接
            conn = await self.pool.get_connection()
            
            try:
                # 2. 执行命令
                result = await conn.send_command(*args, **kwargs)
                return result
            finally:
                # 3. 无论成功失败，都归还连接
                await self.pool.release(conn)
```

**结论**: 单个连接是自动管理的，但连接池本身需要在应用关闭时手动销毁。

### 2.6 超时机制防止网络抖动

**问题**: 为什么需要超时机制？

**答案**:

```
没有超时:
    │
    ├─ 网络抖动时，请求可能卡住 60 秒
    └─ 导致应用线程被阻塞

有超时:
    │
    ├─ 5 秒超时，快速失败
    └─ 应用可以继续处理其他请求
```

---

## 3. 设计决策过程

### 3.1 单例模式实现方式

**讨论**: `__new__` 方式 vs `get_instance()` 方式

| 方式 | 实现 | 优点 | 缺点 |
|------|------|------|------|
| **__new__** | `if not cls._instance:` | 更简洁 | 需要理解 __new__ |
| **get_instance** | `if cls._instance is None:` | 更直观 | 代码稍多 |

**最终决定**: 采用 `__new__` 方式

```python
class RedisCache:
    _instance = None

    def __new__(cls, *args, **kwargs):
        """单例模式: 确保全局只有一个实例"""
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
```

### 3.2 上下文管理器 vs 单例模式

**讨论**: 是否需要上下文管理器 (`__aenter__` / `__aexit__`)

**问题**: 单例模式和上下文管理器冲突

```python
# 单例模式: 全局只有一个实例
cache = RedisCache.get_instance(config)

# 上下文管理器: 每次都会 close()
async with cache as c:  # 进入时 connect()
    ...                 # 退出时 close() ← 问题！
```

**冲突场景**:

```
场景: 两个地方同时使用 cache

    ┌─────────────────────────────────────────────────────────────┐
    │  代码 A                                                     │
    │  async with RedisCache.get_instance(config) as cache:      │
    │      await cache.get("key1")  # 正常                        │
    │  # 退出时 close() ← 连接池被关闭！                           │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  代码 B                                                     │
    │  cache = RedisCache.get_instance(config)                    │
    │  await cache.get("key2")  # ❌ 错误！连接池已关闭            │
    └─────────────────────────────────────────────────────────────┘
```

**最终决定**: 去掉上下文管理器，只用单例模式

```python
# 应用启动时
cache = RedisCache(config.redis)
await cache.connect()

# 使用时
await cache.get("key")

# 应用关闭时
await cache.close()
```

### 3.3 重试机制是否需要

**讨论**: 是否需要重试机制

**分析**:

| 方案 | 优点 | 缺点 |
|------|------|------|
| **有重试** | 网络抖动时自动重试 | 代码复杂 |
| **无重试** | 代码简单 | 网络抖动时直接失败 |

**最终决定**: 去掉重试机制

**理由**:
1. 代码更简洁
2. 超时机制已经足够
3. 重试机制增加复杂度

### 3.4 方法精简

**讨论**: 哪些方法在当前场景不会用上

**分析**:

| 方法 | 会用上？ | 原因 |
|------|----------|------|
| `get` | ✅ 会 | 获取缓存的意图/结果 |
| `set` | ✅ 会 | 设置缓存的意图/结果 |
| `delete` | ✅ 会 | 手动清除缓存 |
| `exists` | ✅ 会 | 检查缓存是否存在 |
| `ping` | ❌ 不会 | 连接问题会通过其他方法暴露 |
| `mget` | ❌ 不会 | 没有批量获取需求 |
| `mset` | ❌ 不会 | 没有批量设置需求 |
| `incr` | ❌ 不会 | 没有计数需求 |
| `keys` | ❌ 不会 | 没有遍历 key 需求 |
| `flushdb` | ❌ 不会 | 清空数据库，太危险 |
| `reset_instance` | ❌ 不会 | 测试用，生产不需要 |

**最终决定**: 只保留 6 个方法

| 方法 | 用途 |
|------|------|
| `connect()` | 创建连接池 |
| `close()` | 关闭连接池 |
| `get(key)` | 获取缓存 |
| `set(key, value, ttl)` | 设置缓存 |
| `delete(key)` | 删除缓存 |
| `exists(key)` | 检查是否存在 |

### 3.5 文件命名问题

**问题**: `redis.py` 与系统包名冲突

**错误信息**:

```
ModuleNotFoundError: No module named 'redis.asyncio'; 'redis' is not a package
```

**原因分析**:

```
Python 导入顺序:
    │
    ├─ 1. 当前目录: src/cache/redis.py  ← 找到了这个！
    ├─ 2. 上级目录: src/redis.py
    └─ 3. 系统包: redis (pip install redis)
    
错误原因:
    Python 找到了你的 redis.py 文件，而不是系统的 redis 包
```

**最终决定**: 重命名为 `redis_cache.py`

---

## 4. 最终实现

### 4.1 文件结构

```
src/cache/
├── __init__.py          # 导出 RedisCache, RedisCacheError
└── redis_cache.py       # Redis 缓存封装
```

### 4.2 完整代码

```python
"""
Redis 异步缓存封装

特性:
    - 连接池管理 (生产标配)
    - 单例模式 (全局唯一连接池)
    - 超时机制 (防止网络抖动)
    - 自动序列化/反序列化 (JSON)
    - 统一异常处理

用法:
    from src.cache import RedisCache
    from src.base.config import load_config

    # 应用启动时
    config = load_config()
    cache = RedisCache(config.redis)
    await cache.connect()

    # 使用时
    await cache.set("key", {"data": "value"}, ttl=3600)
    result = await cache.get("key")

    # 应用关闭时
    await cache.close()
"""

import json
from typing import Any
import redis.asyncio as redis

from src.base import setup_logger
from src.base.config import RedisConfig

logger = setup_logger("RedisCache")


class RedisCacheError(Exception):
    """Redis 缓存统一异常"""


class RedisCache:
    """Redis 异步缓存封装 - 单例 + 连接池模式"""

    _instance: "RedisCache | None" = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        """单例模式: 确保全局只有一个实例"""
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        config: RedisConfig,
        max_connections: int = 20,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
    ):
        """
        初始化 Redis 缓存

        Args:
            config: Redis 配置
            max_connections: 连接池最大连接数
            socket_timeout: 读写超时 (秒)
            socket_connect_timeout: 连接超时 (秒)
        """
        # 防止重复初始化
        if self._initialized:
            return

        self._host = config.host
        self._port = config.port
        self._password = config.password
        self._db = config.db
        self._max_connections = max_connections
        self._socket_timeout = socket_timeout
        self._socket_connect_timeout = socket_connect_timeout

        self._pool: redis.ConnectionPool | None = None
        self._redis: redis.Redis | None = None

        self._initialized = True

    async def connect(self) -> None:
        """创建连接池并测试连接"""
        if self._redis is not None:
            logger.debug("Redis 连接池已存在，跳过创建")
            return

        try:
            self._pool = redis.ConnectionPool(
                host=self._host,
                port=self._port,
                password=self._password,
                db=self._db,
                max_connections=self._max_connections,
                socket_timeout=self._socket_timeout,
                socket_connect_timeout=self._socket_connect_timeout,
                decode_responses=True,  # 自动解码为字符串
                socket_keepalive=True,  # 保持长连接活跃，防止被防火墙切断
            )
            self._redis = redis.Redis(connection_pool=self._pool)

            # 测试连接
            await self._redis.ping()
            logger.info(f"Redis 连接池创建成功: {self._host}:{self._port}/{self._db}")

        except Exception as e:
            logger.error(f"Redis 连接失败: {e}", exc_info=True)
            raise RedisCacheError(f"Redis 连接失败: {e}") from e

    async def close(self) -> None:
        """关闭连接池"""
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
            self._redis = None
            RedisCache._initialized = False
            logger.info("Redis 连接池已关闭")

    # ── 基本操作 ──

    async def get(self, key: str) -> Any | None:
        """
        获取缓存

        Args:
            key: 缓存 key

        Returns:
            缓存值 (自动反序列化 JSON)，不存在返回 None
        """
        try:
            value = await self._redis.get(key)
            if value is not None:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None
        except Exception as e:
            logger.error(f"Redis GET 失败 (key={key}): {e}", exc_info=True)
            raise RedisCacheError(f"Redis GET 失败: {e}") from e

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """
        设置缓存

        Args:
            key: 缓存 key
            value: 缓存值 (自动序列化为 JSON)
            ttl: 过期时间 (秒)，None 表示永不过期

        Returns:
            是否设置成功
        """
        try:
            serialized = json.dumps(value, ensure_ascii=False)
            if ttl:
                await self._redis.set(key, serialized, ex=ttl)
            else:
                await self._redis.set(key, serialized)
            return True
        except Exception as e:
            logger.error(f"Redis SET 失败 (key={key}): {e}", exc_info=True)
            raise RedisCacheError(f"Redis SET 失败: {e}") from e

    async def delete(self, key: str) -> bool:
        """
        删除缓存

        Args:
            key: 缓存 key

        Returns:
            是否删除成功
        """
        try:
            result = await self._redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis DELETE 失败 (key={key}): {e}", exc_info=True)
            raise RedisCacheError(f"Redis DELETE 失败: {e}") from e

    async def exists(self, key: str) -> bool:
        """
        检查 key 是否存在

        Args:
            key: 缓存 key

        Returns:
            是否存在
        """
        try:
            return await self._redis.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis EXISTS 失败 (key={key}): {e}", exc_info=True)
            raise RedisCacheError(f"Redis EXISTS 失败: {e}") from e
```

### 4.3 __init__.py

```python
from .redis_cache import RedisCache, RedisCacheError

__all__ = ["RedisCache", "RedisCacheError"]
```

### 4.4 使用方式

```python
# 应用启动时
config = load_config()
cache = RedisCache(config.redis)
await cache.connect()

# 使用时
await cache.set("key", {"data": "value"}, ttl=3600)
result = await cache.get("key")

# 应用关闭时
await cache.close()
```

---

## 5. 遇到的问题与解决方案

### 5.1 文件名冲突

**问题**: `redis.py` 与系统包名冲突

**错误信息**:

```
ModuleNotFoundError: No module named 'redis.asyncio'; 'redis' is not a package
```

**解决方案**: 重命名为 `redis_cache.py`

**修改内容**:

| 文件 | 修改 |
|------|------|
| `src/cache/redis.py` | 重命名为 `src/cache/redis_cache.py` |
| `src/cache/__init__.py` | 更新导入: `from .redis_cache import ...` |

### 5.2 单例模式与上下文管理器冲突

**问题**: 单例模式和上下文管理器冲突

**解决方案**: 去掉上下文管理器，只用单例模式

**修改内容**:

| 修改 | 说明 |
|------|------|
| 删除 `__aenter__` | 去掉上下文管理器 |
| 删除 `__aexit__` | 去掉上下文管理器 |

### 5.3 重试机制过于复杂

**问题**: 重试机制增加代码复杂度

**解决方案**: 去掉重试机制

**修改内容**:

| 修改 | 说明 |
|------|------|
| 删除 `_execute_with_retry` | 去掉重试机制 |
| 删除 `import asyncio` | 不需要了 |
| 所有方法直接调用 `self._redis.xxx` | 简化代码 |

### 5.4 不需要的方法过多

**问题**: 很多方法在当前场景不会用上

**解决方案**: 删除不需要的方法

**删除的方法**:

| 方法 | 原因 |
|------|------|
| `ping` | 连接问题会通过其他方法暴露 |
| `mget` | 没有批量获取需求 |
| `mset` | 没有批量设置需求 |
| `incr` | 没有计数需求 |
| `keys` | 没有遍历 key 需求 |
| `flushdb` | 清空数据库，太危险 |
| `reset_instance` | 测试用，生产不需要 |

---

## 6. 代码演进历史

### 6.1 版本 1: 初始版本

**特点**:
- 有上下文管理器
- 有重试机制
- 有很多方法

**代码行数**: 377 行

### 6.2 版本 2: 去掉上下文管理器

**修改**:
- 删除 `__aenter__` 和 `__aexit__`
- 保留重试机制

**代码行数**: 350 行

### 6.3 版本 3: 去掉重试机制

**修改**:
- 删除 `_execute_with_retry` 方法
- 所有方法直接调用 `self._redis.xxx`

**代码行数**: 280 行

### 6.4 版本 4: 精简方法

**修改**:
- 删除不需要的方法

**代码行数**: 234 行

### 6.5 版本 5: 重命名文件

**修改**:
- `redis.py` → `redis_cache.py`
- 更新 `__init__.py` 导入

**代码行数**: 234 行

### 6.6 演进总结

| 版本 | 修改 | 代码行数 | 减少 |
|------|------|----------|------|
| V1 | 初始版本 | 377 | - |
| V2 | 去掉上下文管理器 | 350 | 27 (7%) |
| V3 | 去掉重试机制 | 280 | 70 (20%) |
| V4 | 精简方法 | 234 | 46 (16%) |
| V5 | 重命名文件 | 234 | 0 (0%) |
| **总计** | - | 234 | **143 (38%)** |

---

## 7. 关键总结

### 7.1 生产环境的核心认知

| 认知 | 说明 |
|------|------|
| **连接池是标配** | 避免每次创建/关闭连接 |
| **decode_responses=True** | 自动解码为字符串 |
| **socket_keepalive=True** | 防止防火墙切断连接 |
| **单例模式** | 全局唯一连接池 |
| **不需要手动 release** | redis-py 自动归还连接 |
| **超时机制** | 防止网络抖动 |

### 7.2 设计决策总结

| 决策点 | 决定 | 理由 |
|--------|------|------|
| **单例模式** | `__new__` 方式 | 更简洁 |
| **上下文管理器** | 去掉 | 与单例模式冲突 |
| **重试机制** | 去掉 | 简化代码 |
| **方法数量** | 6 个 | 只保留需要的 |
| **文件命名** | `redis_cache.py` | 避免与系统包冲突 |

### 7.3 最终方法列表

| 方法 | 用途 |
|------|------|
| `connect()` | 创建连接池 |
| `close()` | 关闭连接池 |
| `get(key)` | 获取缓存 |
| `set(key, value, ttl)` | 设置缓存 |
| `delete(key)` | 删除缓存 |
| `exists(key)` | 检查是否存在 |

### 7.4 代码行数对比

| 指标 | 初始版本 | 最终版本 | 减少 |
|------|----------|----------|------|
| 代码行数 | 377 | 234 | **38%** |
| 方法数量 | 12 | 6 | **50%** |

### 7.5 使用方式

```python
# 应用启动时
config = load_config()
cache = RedisCache(config.redis)
await cache.connect()

# 使用时
await cache.set("key", {"data": "value"}, ttl=3600)
result = await cache.get("key")

# 应用关闭时
await cache.close()
```

---

> 文档生成时间: 2026-06-28
> 代码版本: V5 (最终版本)
> 代码行数: 234 行
