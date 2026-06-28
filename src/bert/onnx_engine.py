"""
ONNX Runtime 推理引擎（用于 BERT 意图识别）

使用 HuggingFace Optimum + Pipeline 实现优雅的推理接口。
"""

from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import pipeline, AutoTokenizer
from src.base.logger import setup_logger

logger = setup_logger("OnnxRuntimeEngine")

# Provider 映射
PROVIDER_MAP = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
}


class OnnxRuntimeEngine:
    """ONNX Runtime 推理引擎（用于 BERT 意图识别）"""

    def __init__(self, model_dir: str, tokenizer, provider: str = "cpu"):
        """
        初始化 ONNX Runtime 引擎

        Args:
            model_dir: ONNX 模型目录（包含 model.onnx）
            tokenizer: 已加载的 tokenizer 实例
            provider: 推理设备（"cpu", "cuda"）
        """
        self.model_dir = Path(model_dir)
        self.tokenizer = tokenizer
        self.provider = PROVIDER_MAP.get(provider, provider)

        logger.info(f"正在加载模型: {self.model_dir} (Provider={self.provider})")

        # 使用 Optimum 加载 ONNX 模型
        self.model = ORTModelForSequenceClassification.from_pretrained(
            str(self.model_dir),
            provider=self.provider,
        )

        # 创建 pipeline
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        logger.info("模型加载完成")

    def predict(self, texts):
        """
        预测意图

        Args:
            texts: 文本列表或单个文本
        Returns:
            results: [{"label": "intent_1", "score": 0.95}, ...]
        """
        return self.classifier(texts)


if __name__ == "__main__":
    import time

    # 模型路径
    MODEL_DIR = Path("src/bert/onnx/chinese-roberta-wwm-ext-distilled-27745/none")
    TOKENIZER_DIR = Path("src/bert/model/chinese-roberta-wwm-ext-distilled-27745")

    # 测试文本
    test_texts = [
        "我最近总是头疼，怎么办？",
        "这个药有什么副作用吗？",
        "我想预约明天的门诊",
    ]

    # 加载 tokenizer
    print("=" * 60)
    print("加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))

    # 测试 CPU 推理
    print("\n" + "=" * 60)
    print("测试 CPU 推理...")
    engine_cpu = OnnxRuntimeEngine(str(MODEL_DIR), tokenizer, provider="cpu")

    # 预热
    _ = engine_cpu.predict(["预热"])

    # 测试
    start = time.perf_counter()
    results = engine_cpu.predict(test_texts)
    cpu_time = (time.perf_counter() - start) * 1000

    print(f"\nCPU 推理结果:")
    for text, result in zip(test_texts, results):
        print(f"  文本: {text}")
        print(f"  结果: {result}")
    print(f"  耗时: {cpu_time:.2f}ms")

    # 测试 GPU 推理
    print("\n" + "=" * 60)
    print("测试 GPU 推理...")
    engine_gpu = OnnxRuntimeEngine(str(MODEL_DIR), tokenizer, provider="cuda")

    # 预热
    _ = engine_gpu.predict(["预热"])

    # 测试
    start = time.perf_counter()
    results = engine_gpu.predict(test_texts)
    gpu_time = (time.perf_counter() - start) * 1000

    print(f"\nGPU 推理结果:")
    for text, result in zip(test_texts, results):
        print(f"  文本: {text}")
        print(f"  结果: {result}")
    print(f"  耗时: {gpu_time:.2f}ms")

    # 性能对比
    print("\n" + "=" * 60)
    print("性能对比:")
    print(f"  CPU: {cpu_time:.2f}ms")
    print(f"  GPU: {gpu_time:.2f}ms")
    print(f"  加速比: {cpu_time / gpu_time:.2f}x")
