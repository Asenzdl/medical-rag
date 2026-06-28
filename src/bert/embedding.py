"""
BERT 意图识别统一接口

通过工厂函数创建不同后端的推理引擎。

Usage:
    from src.bert.embedding import create_bert_engine

    # 自动选择后端（优先 TensorRT）
    engine = create_bert_engine(
        model_dir="src/bert/onnx/chinese-roberta-wwm-ext-distilled-27745/none",
        tokenizer_dir="src/bert/model/chinese-roberta-wwm-ext-distilled-27745",
    )

    # 指定 ONNX Runtime + GPU
    engine = create_bert_engine(
        model_dir="src/bert/onnx/chinese-roberta-wwm-ext-distilled-27745/none",
        tokenizer_dir="src/bert/model/chinese-roberta-wwm-ext-distilled-27745",
        backend="onnx",
        provider="cuda",
    )

    # 指定 TensorRT
    engine = create_bert_engine(
        model_dir="src/bert/onnx/chinese-roberta-wwm-ext-distilled-27745/none",
        tokenizer_dir="src/bert/model/chinese-roberta-wwm-ext-distilled-27745",
        backend="tensorrt",
    )

    # 统一调用
    results = engine.predict(["高血压怎么治疗"])
    # [{"label": "LABEL_1", "score": 0.95}, ...]
"""

from pathlib import Path
from functools import lru_cache
from typing import Union, TYPE_CHECKING
from transformers import AutoTokenizer
from src.base.logger import setup_logger

if TYPE_CHECKING:
    from src.bert.onnx_engine import OnnxRuntimeEngine
    from src.bert.tensorrt_engine import TensorRTEngine

logger = setup_logger("BertEngine")

# 类型别名
BertEngine = Union["OnnxRuntimeEngine", "TensorRTEngine"]


@lru_cache(maxsize=2)
def get_tokenizer(model_dir: str):
    """缓存 tokenizer，避免重复加载"""
    logger.info(f"加载 Tokenizer: {model_dir}")
    return AutoTokenizer.from_pretrained(model_dir)


def create_bert_engine(
    model_dir: str,
    tokenizer_dir: str,
    backend: str = "auto",
    provider: str = "cpu",
) -> BertEngine:
    """
    创建 BERT 意图识别引擎（工厂函数）

    Args:
        model_dir: 模型目录（ONNX 或 TRT 引擎所在目录）
        tokenizer_dir: Tokenizer 目录
        backend: 推理后端（"auto", "onnx", "tensorrt"）
        provider: 推理设备（"cpu", "cuda"），仅 ONNX Runtime 生效
    Returns:
        BertEngine 实例（OnnxRuntimeEngine 或 TensorRTEngine）
    """
    model_dir = Path(model_dir)

    # 加载 tokenizer
    tokenizer = get_tokenizer(tokenizer_dir)

    # 自动检测后端
    if backend == "auto":
        trt_path = model_dir / "model.trt"
        if trt_path.exists():
            backend = "tensorrt"
            logger.info(f"自动选择后端: TensorRT (发现 {trt_path})")
        else:
            backend = "onnx"
            logger.info("自动选择后端: ONNX Runtime")
    else:
        logger.info(f"指定后端: {backend}")

    # 创建引擎
    if backend == "tensorrt":
        from src.bert.tensorrt_engine import TensorRTEngine
        return TensorRTEngine(model_dir, tokenizer)
    else:
        from src.bert.onnx_engine import OnnxRuntimeEngine
        return OnnxRuntimeEngine(model_dir, tokenizer, provider)


if __name__ == "__main__":
    import time

    # 配置
    MODEL_DIR = "src/bert/onnx/chinese-roberta-wwm-ext-distilled-27745/none"
    TOKENIZER_DIR = "src/bert/model/chinese-roberta-wwm-ext-distilled-27745"

    # 测试文本
    test_texts = [
        "我高血压怎么治疗？",
        "我儿子怎么发烧了？",
        "我肚子疼怎么办？",
    ]

    print("=" * 60)
    print("BERT 意图识别统一接口测试")
    print("=" * 60)

    # 测试 ONNX Runtime CPU
    print("\n--- ONNX Runtime CPU ---")
    engine_cpu = create_bert_engine(MODEL_DIR, TOKENIZER_DIR, backend="onnx", provider="cpu")
    _ = engine_cpu.predict(["预热"])  # 预热
    start = time.perf_counter()
    results = engine_cpu.predict(test_texts)
    cpu_time = (time.perf_counter() - start) * 1000
    for text, result in zip(test_texts, results):
        print(f"  {text} → {result}")
    print(f"  耗时: {cpu_time:.2f}ms")

    # 测试 ONNX Runtime GPU
    print("\n--- ONNX Runtime GPU ---")
    engine_gpu = create_bert_engine(MODEL_DIR, TOKENIZER_DIR, backend="onnx", provider="cuda")
    _ = engine_gpu.predict(["预热"])  # 预热
    start = time.perf_counter()
    results = engine_gpu.predict(test_texts)
    gpu_time = (time.perf_counter() - start) * 1000
    for text, result in zip(test_texts, results):
        print(f"  {text} → {result}")
    print(f"  耗时: {gpu_time:.2f}ms")

    # 测试 TensorRT
    print("\n--- TensorRT ---")
    engine_trt = create_bert_engine(MODEL_DIR, TOKENIZER_DIR, backend="tensorrt")
    _ = engine_trt.predict(["预热"])  # 预热
    start = time.perf_counter()
    results = engine_trt.predict(test_texts)
    trt_time = (time.perf_counter() - start) * 1000
    for text, result in zip(test_texts, results):
        print(f"  {text} → {result}")
    print(f"  耗时: {trt_time:.2f}ms")

    # 测试自动选择
    print("\n--- 自动选择后端 ---")
    engine_auto = create_bert_engine(MODEL_DIR, TOKENIZER_DIR)
    _ = engine_auto.predict(["预热"])  # 预热
    start = time.perf_counter()
    results = engine_auto.predict(test_texts)
    auto_time = (time.perf_counter() - start) * 1000
    for text, result in zip(test_texts, results):
        print(f"  {text} → {result}")
    print(f"  耗时: {auto_time:.2f}ms")

    # 性能对比
    print("\n" + "=" * 60)
    print("性能对比:")
    print(f"  ONNX CPU:  {cpu_time:.2f}ms")
    print(f"  ONNX GPU:  {gpu_time:.2f}ms")
    print(f"  TensorRT:  {trt_time:.2f}ms")
    print(f"  自动选择:  {auto_time:.2f}ms")
