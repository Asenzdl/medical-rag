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


if __name__ == "__main__":
    import asyncio
    from src.base import load_config

    async def main():
        config = load_config()

        # 单例模式: 直接创建实例并连接
        cache = RedisCache(config.redis)
        await cache.connect()

        # 测试基本操作
        await cache.set("test:key1", {"name": "测试", "value": 123}, ttl=60)
        result = await cache.get("test:key1")
        print(f"GET 结果: {result}")

        # 测试存在性
        exists = await cache.exists("test:key1")
        print(f"EXISTS: {exists}")

        # 测试删除
        await cache.delete("test:key1")
        exists = await cache.exists("test:key1")
        print(f"DELETE 后 EXISTS: {exists}")

        # 关闭连接
        await cache.close()

    asyncio.run(main())
