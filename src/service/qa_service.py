"""
QA Service - 医疗问答服务层

流程：意图识别 → 向量生成 → Redis 缓存 → ES 回源
"""

from src.bert.intent import BertEngine
from src.embeddings.embedding import EmbeddingModel
from src.cache.redis_cache import RedisCache
from src.es import ESClient


class MedicalQAService:
    """医疗问答服务"""

    def __init__(
        self,
        bert: BertEngine,
        embedding: EmbeddingModel,
        redis: RedisCache,
        es: ESClient,
    ):
        self.bert = bert
        self.embedding = embedding
        self.redis = redis
        self.es = es

    async def query(self, question: str) -> dict:
        """
        问答查询

        Args:
            question: 用户问题

        Returns:
            {
                "category": str,
                "results": list[dict],
            }
        """
        # 1. 意图识别
        intent_result = self.bert.predict(question)
        category = intent_result["label"]

        # 2. 向量生成
        # 3. Redis 向量搜索
        # 4. 未命中则 ES 搜索
        # 5. 结果写入缓存

        # TODO: 后续步骤实现后替换
        return intent_result


if __name__ == "__main__":
    import asyncio
    from src.bert.intent import create_bert_engine
    from src.base.config import load_config

    async def main():
        config = load_config()

        bert = create_bert_engine(
            model_dir=config.bert.model_dir,
            tokenizer_dir=config.bert.tokenizer_dir,
            backend=config.bert.backend,
            provider=config.bert.provider,
            label_map=config.bert.label_map,
            log_latency=True,
        )

        service = MedicalQAService(bert=bert, embedding=None, redis=None, es=None)

        questions = [
            "高血压怎么治疗",      # 医疗问题
            "高血压的症状有哪些",        # 医疗问题
            "偏头痛如何治疗",        # 闲聊
        ]
        for q in questions:
            result = await service.query(q)
            print(f"问题: {q}")
            print(f"结果: {result}")
            print()

    asyncio.run(main())
