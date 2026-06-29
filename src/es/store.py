from elasticsearch import AsyncElasticsearch

from src.base import setup_logger
from src.base.config import ElasticsearchConfig

logger = setup_logger("ESClient")


class ESClientError(Exception):
    """ES 客户端统一异常"""


class ESClient:
    """Elasticsearch 异步客户端，封装索引管理和 CRUD"""

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
                raise ESClientError(f"ES 无响应: {url}")
            logger.info(f"ES 连接成功: {url}")
        except Exception as e:
            logger.error(f"ES 连接失败: {e}", exc_info=True)
            raise ESClientError(f"ES 连接失败: {e}") from e

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
            raise ESClientError(f"检查索引失败: {e}") from e

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
            raise ESClientError(f"索引创建失败: {e}") from e

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
            raise ESClientError(f"索引删除失败: {e}") from e

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
            # 1. 全文检索条件
            must_clause = {
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "ask^2", "answer"],
                    "type": "best_fields",
                }
            }

            # 2. 过滤条件（不参与评分）
            filter_clause = []
            if category:
                filter_clause.append({"term": {"category": category}})

            # 3. 组合 DSL
            query_body = {"bool": {"must": [must_clause], "filter": filter_clause}}

            # 4. 高亮配置（覆盖所有检索字段）
            highlight = {
                "fields": {
                    "title": {"number_of_fragments": 1},
                    "ask": {"number_of_fragments": 1},
                    "answer": {"number_of_fragments": 1},
                }
            }

            # 5. 动态参数
            kwargs = {}
            if source_includes:
                kwargs["_source"] = source_includes

            # 6. 执行搜索
            result = await self._es.search(
                index=self._index,
                query=query_body,
                size=top_k,
                highlight=highlight,
                **kwargs,
            )

            # 7. 安全提取结果
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
            raise ESClientError(f"搜索失败: {e}") from e

    # ── 管理 ──

    async def count(self) -> int:
        """返回索引中文档总数"""
        try:
            result = await self._es.count(index=self._index)
            return result["count"]
        except Exception as e:
            logger.error(f"查询文档总数失败: {e}", exc_info=True)
            raise ESClientError(f"查询文档总数失败: {e}") from e


if __name__ == "__main__":
    import asyncio
    from src.base import load_config

    async def main():
        config = load_config()
        async with ESClient(config.elasticsearch) as store:
            info = await store._es.info()
            logger.info(f"ES 版本: {info['version']['number']}")

            # 测试 create_index（幂等：可重复运行）
            await store.create_index()

            # 验证索引存在
            exists = await store.index_exists()
            logger.info(f"索引 {store._index} 存在: {exists}")

            # 测试 delete_index（取消注释执行）
            # await store.delete_index()
            # logger.info(f"索引 {store._index} 存在: {await store.index_exists()}")

            # 测试 search（取消注释执行）
            # import json
            # import time
            
            # start_time = time.time()
            # results = await store.search("高血压怎么治疗", category="内科", top_k=5)
            # end_time = time.time()
            # logger.info(f"分类搜索耗时: {(end_time - start_time) * 1000:.2f} ms")
            # # print(json.dumps(results, indent=2, ensure_ascii=False))
            # start_time = time.time()
            # results = await store.search("支气管扩张怎么办", category="外科", top_k=5)
            # end_time = time.time()
            # logger.info(f"分类搜索耗时: {(end_time - start_time) * 1000:.2f} ms")
            # print(json.dumps(results, indent=2, ensure_ascii=False))
            # start_time = time.time()
            # results = await store.search("腰痛咋办", category="外科", top_k=5)
            # end_time = time.time()
            # logger.info(f"分类搜索耗时: {(end_time - start_time) * 1000:.2f} ms")
            # print(json.dumps(results, indent=2, ensure_ascii=False))

            # 全量搜索（不带 category 过滤）
            # start_time = time.time()
            # results = await store.search("高血压怎么治疗", top_k=5)
            # end_time = time.time()
            # logger.info(f"全量搜索耗时: {(end_time - start_time) * 1000:.2f} ms")
            # # print(json.dumps(results, indent=2, ensure_ascii=False))
            # start_time = time.time()
            # results = await store.search("支气管扩张怎么办", top_k=5)
            # end_time = time.time()
            # logger.info(f"全量搜索耗时: {(end_time - start_time) * 1000:.2f} ms")
            # # print(json.dumps(results, indent=2, ensure_ascii=False))
            # start_time = time.time()
            # results = await store.search("腰痛咋办", top_k=5)
            # end_time = time.time()
            # logger.info(f"全量搜索耗时: {(end_time - start_time) * 1000:.2f} ms")
            # # print(json.dumps(results, indent=2, ensure_ascii=False))

    asyncio.run(main())
