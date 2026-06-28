"""
TensorRT 推理引擎封装

优化版本: Buffer 预分配 + Stream 复用 + 统一推理逻辑

用法:
    from src.bert.trt_engine import TRTInferenceEngine
    engine = TRTInferenceEngine()
    result = engine.predict("高血压怎么治疗")
    results = engine.predict_batch(["高血压", "发烧"])
"""

import sys
import os

# 修复 Windows 编码问题
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
import numpy as np
import tensorrt as trt
import torch
from transformers import AutoTokenizer

from src.base import setup_logger

logger = setup_logger("TRTEngine")

MODEL_DIR = Path(__file__).parent
ENGINE_PATH = MODEL_DIR / "model.trt"
TOKENIZER_PATH = MODEL_DIR / "model"

# 科室标签映射
ID2LABEL = {
    0: "儿科",
    1: "内科",
    2: "外科",
    3: "妇产科",
    4: "男科",
    5: "肿瘤科",
}


class TRTInferenceEngine:
    """TensorRT 推理引擎 (优化版)"""

    def __init__(
        self,
        engine_path: Path = ENGINE_PATH,
        tokenizer_path: Path = TOKENIZER_PATH,
        max_length: int = 128,
        max_batch_size: int = 32,
    ):
        self.max_length = max_length
        self.max_batch_size = max_batch_size
        self.id2label = ID2LABEL

        # 加载分词器
        logger.info(f"加载分词器: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # 加载 TensorRT Engine
        logger.info(f"加载 TensorRT Engine: {engine_path}")
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        # 预分配资源
        self.stream = torch.cuda.Stream()
        self._allocate_buffers()

        logger.info("TensorRT 引擎初始化完成")

    def _load_engine(self, engine_path: Path) -> trt.ICudaEngine:
        """加载 TensorRT Engine"""
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            raise RuntimeError(f"无法加载 Engine: {engine_path}")

        return engine

    def _allocate_buffers(self) -> None:
        """预分配 GPU buffer (按最大 batch_size)"""
        logger.info(f"预分配 GPU buffer: max_batch={self.max_batch_size}, seq_len={self.max_length}")

        # 预分配固定大小的 GPU buffer
        self.d_input_ids = torch.zeros(
            (self.max_batch_size, self.max_length), dtype=torch.int64, device="cuda"
        )
        self.d_attention_mask = torch.zeros(
            (self.max_batch_size, self.max_length), dtype=torch.int64, device="cuda"
        )
        self.d_output = torch.zeros(
            (self.max_batch_size, 6), dtype=torch.float32, device="cuda"
        )

        logger.info("GPU buffer 预分配完成")

    def predict(self, text: str) -> dict:
        """
        预测单条文本的科室分类

        Args:
            text: 输入文本

        Returns:
            dict: {"label": "内科", "confidence": 0.95, "logits": [...]}
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
        batch_size = len(texts)

        if batch_size > self.max_batch_size:
            raise ValueError(f"batch_size({batch_size}) > max_batch_size({self.max_batch_size})")

        # 1. Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="np",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)

        # 2. H2D: 拷贝到预分配 buffer 的前 batch_size 行
        self.d_input_ids[:batch_size].copy_(torch.from_numpy(input_ids))
        self.d_attention_mask[:batch_size].copy_(torch.from_numpy(attention_mask))

        # 3. 设置 tensor shape 和地址
        self.context.set_input_shape("input_ids", (batch_size, self.max_length))
        self.context.set_input_shape("attention_mask", (batch_size, self.max_length))
        self.context.set_tensor_address("input_ids", self.d_input_ids.data_ptr())
        self.context.set_tensor_address("attention_mask", self.d_attention_mask.data_ptr())
        self.context.set_tensor_address("logits", self.d_output.data_ptr())

        # 4. 执行推理 (复用 stream)
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        # 5. D2H: 只拷贝有效部分
        logits_batch = self.d_output[:batch_size].cpu().numpy()

        # 6. 后处理
        return self._postprocess(logits_batch)

    def _postprocess(self, logits_batch: np.ndarray) -> list[dict]:
        """后处理: softmax + argmax"""
        results = []

        for logits in logits_batch:
            # softmax
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

    def __del__(self):
        """释放资源"""
        if hasattr(self, "d_input_ids"):
            del self.d_input_ids
        if hasattr(self, "d_attention_mask"):
            del self.d_attention_mask
        if hasattr(self, "d_output"):
            del self.d_output


if __name__ == "__main__":
    import time

    # 测试推理
    engine = TRTInferenceEngine()

    test_texts = [
        "高血压怎么治疗",
        "孩子发烧怎么办",
        "骨折后怎么护理",
    ]

    # 预热
    print("=== 预热 ===")
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
