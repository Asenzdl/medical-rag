"""
Embedding 统一接口

通过工厂函数创建不同后端的 Embedding 模型。

Usage:
    from src.embeddings.embedding import create_embedding_model

    # 自动选择后端（优先 TensorRT）
    model = create_embedding_model("src/embeddings/onnx/bge-base-zh-v1.5/none")

    # 指定 SentenceTransformer + CPU
    model = create_embedding_model(
        "src/embeddings/onnx/bge-base-zh-v1.5/o1",
        backend="sentence_transformer",
        provider="cpu",
    )

    # 指定 TensorRT
    model = create_embedding_model(
        "src/embeddings/onnx/bge-base-zh-v1.5/none",
        backend="tensorrt",
        tokenizer_dir="src/embeddings/model/bge-base-zh-v1.5",
    )

    # 统一调用
    embeddings = model.encode(["高血压怎么治疗"])
"""

from pathlib import Path
from functools import lru_cache
from typing import Union, TYPE_CHECKING
from transformers import AutoTokenizer
from src.base.logger import setup_logger

if TYPE_CHECKING:
    from src.embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding
    from src.embeddings.tensorrt_embedding import TensorRTEmbedding

logger = setup_logger("Embedding")

# 类型别名
EmbeddingModel = Union["SentenceTransformerEmbedding", "TensorRTEmbedding"]


@lru_cache(maxsize=2)
def get_tokenizer(model_dir: str):
    """缓存 tokenizer，避免重复加载"""
    logger.info(f"加载 Tokenizer: {model_dir}")
    return AutoTokenizer.from_pretrained(model_dir)


def create_embedding_model(
    model_dir: str,
    backend: str = "auto",
    provider: str = "cpu",
    tokenizer_dir: str = None,
) -> EmbeddingModel:
    """
    创建 Embedding 模型（工厂函数）

    Args:
        model_dir: 模型目录
        backend: 推理后端（"auto", "sentence_transformer", "tensorrt"）
        provider: 推理设备（"cpu", "cuda", "tensorrt"），仅 SentenceTransformer 生效
        tokenizer_dir: Tokenizer 目录，TensorRT 后端必须指定
    Returns:
        EmbeddingModel 实例（SentenceTransformerEmbedding 或 TensorRTEmbedding）
    """
    model_dir = Path(model_dir)

    # 自动检测后端
    if backend == "auto":
        trt_path = model_dir / "model.trt"
        if trt_path.exists():
            backend = "tensorrt"
            logger.info(f"自动选择后端: TensorRT (发现 {trt_path})")
        else:
            backend = "sentence_transformer"
            logger.info("自动选择后端: SentenceTransformer")
    else:
        logger.info(f"指定后端: {backend}")

    # 创建模型
    if backend == "tensorrt":
        from src.embeddings.tensorrt_embedding import TensorRTEmbedding

        if not tokenizer_dir:
            raise ValueError("TensorRT 后端必须指定 tokenizer_dir")
        tokenizer = get_tokenizer(tokenizer_dir)
        return TensorRTEmbedding(model_dir, tokenizer)
    else:
        from src.embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding
        return SentenceTransformerEmbedding(model_dir, provider)


if __name__ == "__main__":
    import time

    # 配置
    MODEL_DIR = "src/embeddings/onnx/bge-base-zh-v1.5/none"
    TOKENIZER_DIR = "src/embeddings/model/bge-base-zh-v1.5"

    print("=" * 60)
    print("Embedding 统一接口测试")
    print("=" * 60)

    # 测试用例
    test_cases = [
        ["高血压怎么治疗"],
        ["高血压怎么治疗", "糖尿病的症状有哪些"],
        ["感冒了怎么办", "发烧吃什么药", "头痛的原因"],
    ]

    # 测试 TensorRT 后端
    print("\n--- TensorRT 后端 ---")
    trt_model = create_embedding_model(MODEL_DIR, backend="tensorrt", tokenizer_dir=TOKENIZER_DIR)
    for i, texts in enumerate(test_cases):
        start = time.perf_counter()
        embeddings = trt_model.encode(texts)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"测试 {i + 1}: {len(texts)} 条 | 形状={embeddings.shape} | 耗时={elapsed:.2f} ms")

    # 测试 SentenceTransformer 后端（需要包含 tokenizer 的目录，如 o3）
    print("\n--- SentenceTransformer 后端 ---")
    st_model_dir = "src/embeddings/onnx/bge-base-zh-v1.5/o3"
    st_model = create_embedding_model(st_model_dir, backend="sentence_transformer", provider="cpu")
    for i, texts in enumerate(test_cases):
        start = time.perf_counter()
        embeddings = st_model.encode(texts)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"测试 {i + 1}: {len(texts)} 条 | 形状={embeddings.shape} | 耗时={elapsed:.2f} ms")

    # 测试自动选择
    print("\n--- 自动选择后端 ---")
    auto_model = create_embedding_model(MODEL_DIR, tokenizer_dir=TOKENIZER_DIR)
    for i, texts in enumerate(test_cases):
        start = time.perf_counter()
        embeddings = auto_model.encode(texts)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"测试 {i + 1}: {len(texts)} 条 | 形状={embeddings.shape} | 耗时={elapsed:.2f} ms")

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
