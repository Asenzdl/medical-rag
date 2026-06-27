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
    category: str           # 大科室（从文件名提取）
    department: str         # 细分科室（CSV 内字段）
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
        category = csv_file.stem  # "儿科.csv" -> "儿科"
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
