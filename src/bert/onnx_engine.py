"""
ONNX Runtime 推理引擎（用于 BERT 意图识别）

使用 HuggingFace Optimum + Pipeline 实现优雅的推理接口。
"""

import time
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

    def __init__(self, model_dir: str, tokenizer, provider: str = "cpu",
                 label_map: dict[str, int] | None = None, log_latency: bool = False):
        """
        初始化 ONNX Runtime 引擎

        Args:
            model_dir: ONNX 模型目录（包含 model.onnx）
            tokenizer: 已加载的 tokenizer 实例
            provider: 推理设备（"cpu", "cuda"）
            label_map: 标签映射 {"内科": 1, ...}，用于将 LABEL_X 转为可读名称
            log_latency: 是否记录每次推理耗时
        """
        self.model_dir = Path(model_dir)
        self.tokenizer = tokenizer
        self.provider = PROVIDER_MAP.get(provider, provider)
        self.log_latency = log_latency

        logger.info(f"正在加载模型: {self.model_dir} (Provider={self.provider})")

        # 使用 Optimum 加载 ONNX 模型
        self.model = ORTModelForSequenceClassification.from_pretrained(
            str(self.model_dir),
            provider=self.provider,
        )

        # 构建 id2label 并设置到模型 config（pipeline 会从 config 读取）
        if label_map:
            id2label = {v: k for k, v in label_map.items()}
            self.model.config.id2label = id2label
            self.model.config.label2id = {v: k for k, v in id2label.items()}

        # 创建 pipeline（显式指定 device，避免 pipeline 自动选 GPU）
        device = "cuda" if "CUDA" in self.provider else "cpu"
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
        )

        logger.info("模型加载完成")

    def predict(self, text: str) -> dict:
        """
        预测意图

        Args:
            text: 单条文本
        Returns:
            {"label": "内科", "score": 0.95}
        """
        start = time.perf_counter()
        result = self.classifier(text)[0]
        if self.log_latency:
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"推理耗时: {elapsed:.2f}ms | 输入: {text[:30]}...")
        return result


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
