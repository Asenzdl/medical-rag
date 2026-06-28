"""
HuggingFace Optimum 推理引擎

使用 ONNX Runtime 推理，代码最简洁

用法:
    from src.bert.optimum_engine import OptimumEngine
    engine = OptimumEngine()
    result = engine.predict("高血压怎么治疗")
"""

import sys
import os

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

from src.base import setup_logger

logger = setup_logger("OptimumEngine")

MODEL_DIR = Path(__file__).parent / "model"
ONNX_DIR = Path(__file__).parent / "onnx"

# 科室标签映射
ID2LABEL = {
    0: "儿科",
    1: "内科",
    2: "外科",
    3: "妇产科",
    4: "男科",
    5: "肿瘤科",
}


class OptimumEngine:
    """HuggingFace Optimum 推理引擎 (最简洁方案)"""

    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        onnx_dir: Path = ONNX_DIR,
        use_gpu: bool = True,
    ):
        self.id2label = ID2LABEL

        # 加载分词器
        logger.info(f"加载分词器: {onnx_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(onnx_dir)

        # 选择 Provider
        provider = "CUDAExecutionProvider" if use_gpu else "CPUExecutionProvider"
        logger.info(f"加载模型 (provider={provider})")

        # 从已导出的 ONNX 目录加载
        logger.info(f"加载 ONNX 模型: {onnx_dir}")
        self.model = ORTModelForSequenceClassification.from_pretrained(
            onnx_dir,
            file_name="model.onnx",
            provider=provider,
        )

        logger.info("模型加载完成")

    def predict(self, text: str) -> dict:
        """
        预测单条文本

        Args:
            text: 输入文本

        Returns:
            dict: {"label": "内科", "confidence": 0.95, ...}
        """
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        批量预测

        Args:
            texts: 输入文本列表

        Returns:
            list[dict]: 预测结果列表
        """
        # 1. Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        # 2. 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.numpy()

        # 3. 后处理
        return self._postprocess(logits)

    def _postprocess(self, logits_batch: np.ndarray) -> list[dict]:
        """后处理: softmax + argmax"""
        results = []

        for logits in logits_batch:
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            pred_id = int(np.argmax(probs))

            results.append({
                "label": self.id2label[pred_id],
                "confidence": float(probs[pred_id]),
                "logits": logits.tolist(),
                "probabilities": {self.id2label[i]: float(p) for i, p in enumerate(probs)},
            })

        return results


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("HuggingFace Optimum 推理引擎测试")
    print("=" * 60)

    # 初始化 (先用 CPU 验证，GPU 有兼容性问题)
    engine = OptimumEngine(use_gpu=False)

    test_texts = [
        "高血压怎么治疗",
        "孩子发烧怎么办",
        "骨折后怎么护理",
    ]

    # 预热
    print("\n=== 预热 ===")
    engine.predict("预热")

    # 单条推理测试
    print("\n=== 单条推理测试 ===")
    for text in test_texts:
        start = time.perf_counter()
        result = engine.predict(text)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"输入: {text}")
        print(f"预测: {result['label']} (置信度: {result['confidence']:.2%})")
        print(f"耗时: {elapsed:.2f} ms\n")

    # 批量推理测试
    print("=== 批量推理测试 ===")
    start = time.perf_counter()
    results = engine.predict_batch(test_texts)
    elapsed = (time.perf_counter() - start) * 1000

    for text, result in zip(test_texts, results):
        print(f"{text} -> {result['label']} ({result['confidence']:.2%})")
    print(f"批量耗时: {elapsed:.2f} ms (平均 {elapsed/len(test_texts):.2f} ms/条)")

    # 性能测试
    print("\n=== 性能测试 (100 次) ===")
    warmup = 10
    iterations = 100

    # 预热
    for _ in range(warmup):
        engine.predict_batch(test_texts)

    # 测试
    start = time.perf_counter()
    for _ in range(iterations):
        engine.predict_batch(test_texts)
    elapsed = (time.perf_counter() - start)

    print(f"总耗时: {elapsed:.2f} s")
    print(f"吞吐量: {iterations * len(test_texts) / elapsed:.1f} 条/秒")
    print(f"平均延迟: {elapsed / iterations * 1000:.2f} ms/批")
