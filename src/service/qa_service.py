"""
QA Service - 医疗问答服务层

流程：意图识别 → Redis 精准匹配 → ES 回源
"""

import hashlib
import re

from loguru import logger
from src.bert.intent import BertEngine
from src.cache.redis_cache import RedisCache
from src.es import ESClient


class MedicalQAService:
    """医疗问答服务"""

    def __init__(
        self,
        bert: BertEngine,
        redis: RedisCache,
        es: ESClient,
    ):
        self.bert = bert
        self.redis = redis
        self.es = es

    async def query(self, question: str, top_k: int = 5) -> dict:
        """
        问答查询

        Args:
            question: 用户问题
            top_k: 返回结果数量

        Returns:
            {
                "category": str,
                "results": list[dict],
            }
        """
        # 1. 意图识别
        intent_result = self.bert.predict(question)
        category = intent_result["label"]
        score = intent_result["score"]
        logger.info(f"意图识别: {category} (置信度: {score:.2f})")

        # 2. 获取知识（缓存精准匹配，ES 兜底）
        results = await self._get_knowledge(question, top_k, category)
        logger.info(f"查询完成: {question} → {len(results)} 条结果")

        return {
            "category": category,
            "results": results,
        }

    # ── 内部方法 ──

    def _build_cache_key(self, query: str) -> str:
        """
        生成缓存 key

        对 query 做标准化（strip + lower + 去标点）后取 SHA256 前 16 位，
        确保"高血压怎么治"和"高血压怎么治？"命中同一条缓存。
        """
        normalized = re.sub(r'[^\w]', '', query.strip().lower())
        digest = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        return f"qa:{digest}"

    async def _get_knowledge(self, query: str, top_k: int, category: str) -> list[dict]:
        """
        获取知识：缓存精准匹配优先，ES 回源兜底

        Args:
            query: 用户问题
            top_k: 返回结果数量
            category: 科室分类

        Returns:
            知识列表
        """
        cache_key = self._build_cache_key(query)

        # 1. 精准匹配
        cached = await self.redis.get(cache_key)
        if cached:
            logger.debug(f"缓存命中: {cache_key} → {len(cached)} 条结果")
            return cached

        # 2. miss → ES 检索
        logger.debug(f"缓存未命中: {cache_key}, 查询 ES...")
        results = await self.es.search(query, category=category, top_k=top_k)

        # 3. 写回缓存
        if results:
            await self.redis.set(cache_key, results, ttl=3600)
            logger.debug(f"缓存写入: {cache_key} → {len(results)} 条结果")

        return results


if __name__ == "__main__":
    import asyncio
    import time
    from src.bert.intent import create_bert_engine
    from src.base.config import load_config

    async def main():
        config = load_config()

        # 初始化组件
        bert = create_bert_engine(
            model_dir=config.bert.model_dir,
            tokenizer_dir=config.bert.tokenizer_dir,
            backend=config.bert.backend,
            provider=config.bert.provider,
            label_map=config.bert.label_map,
        )

        redis = RedisCache(config.redis)
        await redis.connect()

        async with ESClient(config.elasticsearch) as es:
            service = MedicalQAService(bert=bert, redis=redis, es=es)

            test_query = "高血压怎么治疗"
            cache_key = service._build_cache_key(test_query)

            # ── 测试 1: 缓存未命中 → ES 回源 → 写回 Redis ──
            logger.info("=" * 50)
            logger.info("测试 1: 缓存未命中 → ES 回源")
            logger.info("=" * 50)

            # 清理可能存在的缓存
            await redis.delete(cache_key)

            start = time.perf_counter()
            result1 = await service.query(test_query)
            elapsed1 = (time.perf_counter() - start) * 1000

            # 验证缓存已写入
            cached = await redis.get(cache_key)
            assert cached is not None, "失败: 缓存应该已写入"

            logger.info(f"问题: {test_query}")
            logger.info(f"分类: {result1['category']}")
            logger.info(f"结果数: {len(result1['results'])}")
            logger.info(f"耗时: {elapsed1:.2f} ms")
            logger.info(f"缓存已写入: {cached is not None}")

            # ── 测试 2: 缓存命中 ──
            logger.info("=" * 50)
            logger.info("测试 2: 缓存命中")
            logger.info("=" * 50)

            start = time.perf_counter()
            result2 = await service.query(test_query)
            elapsed2 = (time.perf_counter() - start) * 1000

            # 验证结果一致
            assert result1 == result2, "失败: 缓存命中结果应与首次一致"

            logger.info(f"问题: {test_query}")
            logger.info(f"结果与首次一致: {result1 == result2}")
            logger.info(f"耗时: {elapsed2:.2f} ms")
            logger.info(f"加速: {elapsed1 - elapsed2:.2f} ms")

            # ── 测试 3: 不同 query 走 ES 回源 ──
            logger.info("=" * 50)
            logger.info("测试 3: 不同 query → ES 回源")
            logger.info("=" * 50)

            another_query = "偏头痛如何治疗"
            another_key = service._build_cache_key(another_query)

            # 确保缓存不存在
            await redis.delete(another_key)

            start = time.perf_counter()
            result3 = await service.query(another_query)
            elapsed3 = (time.perf_counter() - start) * 1000

            cached3 = await redis.get(another_key)
            assert cached3 is not None, "失败: 新 query 缓存应该已写入"

            logger.info(f"问题: {another_query}")
            logger.info(f"分类: {result3['category']}")
            logger.info(f"结果数: {len(result3['results'])}")
            logger.info(f"耗时: {elapsed3:.2f} ms")

        await redis.close()
        logger.info("全部测试通过 ✓")

    asyncio.run(main())
