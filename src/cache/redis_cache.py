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
import numpy as np
import redis.asyncio as redis
from redis.commands.search.field import VectorField, TextField, TagField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

from loguru import logger
from src.base.log_config import log_latency
from src.base.config import RedisConfig


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

    @log_latency
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

    @log_latency
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

    @log_latency
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

    @log_latency
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

    # ── 向量搜索 ──

    async def create_vector_index(
        self,
        index_name: str,
        prefix: str = "qa:",
        dim: int = 768
    ) -> bool:
        """
        创建向量索引

        Args:
            index_name: 索引名称
            prefix: key 前缀
            dim: embedding 维度

        Returns:
            是否创建成功
        """
        try:
            # 1. 定义 Schema
            schema = (
                TextField("question"),  # 普通文本字段
                TagField("category"),  # 标签字段 (用于过滤)
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

    async def vector_search(
        self,
        index_name: str,
        embedding: list[float],
        top_k: int = 1,
        filters: dict[str, str] | None = None
    ) -> list[dict]:
        """
        向量搜索 (支持标量过滤)

        Args:
            index_name: 索引名称
            embedding: 查询向量
            top_k: 返回结果数量
            filters: 标量过滤条件 {"category": "内科"}

        Returns:
            相似结果列表 [{id, question, result, score}, ...]
        """
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
                .dialect(2)
            )

            # 3. 转换 embedding 为 bytes
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

            # 4. 执行向量搜索 (只返回需要的字段，排除 embedding 二进制数据)
            query = query.return_fields("question", "result", "score")
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

    async def set_with_embedding(
        self,
        key: str,
        question: str,
        result: dict,
        embedding: list[float],
        category: str | None = None,
        ttl: int | None = None
    ) -> bool:
        """
        存储带 embedding 的缓存

        Args:
            key: 缓存 key
            question: 问题文本
            result: 缓存结果
            embedding: 问题向量
            category: 科室分类 (用于过滤)
            ttl: 过期时间 (秒)

        Returns:
            是否设置成功
        """
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


if __name__ == "__main__":
    import asyncio
    from src.base import load_config

    async def main():
        config = load_config()

        # 单例模式: 直接创建实例并连接
        cache = RedisCache(config.redis)
        await cache.connect()

        #############################################
        # 普通文本操作
        # # 测试基本操作
        # await cache.set("test:key1", {"name": "测试", "value": 123}, ttl=60)
        # result = await cache.get("test:key1")
        # print(f"GET 结果: {result}")

        # # 测试存在性
        # exists = await cache.exists("test:key1")
        # print(f"EXISTS: {exists}")

        # # 测试删除
        # await cache.delete("test:key1")
        # exists = await cache.exists("test:key1")
        # print(f"DELETE 后 EXISTS: {exists}")
        #############################################
        
        # 测试向量索引创建 (需要 Redis Stack)
        await cache.create_vector_index(
            index_name="qa_semantic_cache",
            prefix="qa:",
            dim=768
        )
        print("向量索引创建成功")

        # 测试存储带 embedding 的缓存 (需要 Redis Stack)
        await cache.set_with_embedding(
            key="qa:test1",
            question="高血压怎么治疗",
            result={"intent": {"label": "内科"}, "results": [{"title": "高血压治疗"}]},
            embedding=[0.1] * 768,  # 模拟 embedding
            category="内科",  # 添加 category 字段
            ttl=3600
        )
        print("带 embedding 的缓存存储成功")

        # 测试向量搜索 (需要 Redis Stack)
        results = await cache.vector_search(
            index_name="qa_semantic_cache",
            embedding=[0.1] * 768,  # 模拟 embedding
            top_k=5,
            filters={"category": "内科"}
        )
        print(f"找到 {len(results)} 个相似结果")
        for result in results:
            print(f"  - {result['question']} (score: {result['score']})")

        # 关闭连接
        await cache.close()

    asyncio.run(main())
