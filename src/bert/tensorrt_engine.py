"""
TensorRT 推理引擎（用于 BERT 意图识别）

基于 trt_sdk.py 的高性能 GPU 推理。
"""

import time
import numpy as np
import torch
from pathlib import Path
from src.bert.trt_sdk import TRT11InferenceEngine
from src.base.logger import setup_logger

logger = setup_logger("TensorRTEngine")


class TensorRTEngine:
    """TensorRT 推理引擎（仅支持 GPU）"""

    def __init__(self, model_dir: str, tokenizer, label_map: dict[str, int] | None = None,
                 log_latency: bool = False):
        """
        初始化 TensorRT 引擎

        Args:
            model_dir: TRT 引擎文件所在目录（包含 model.trt）
            tokenizer: 已加载的 tokenizer 实例
            label_map: 标签映射 {"内科": 1, ...}，用于将 LABEL_X 转为可读名称
            log_latency: 是否记录每次推理耗时
        """
        self.model_dir = Path(model_dir)
        self.tokenizer = tokenizer
        self.log_latency = log_latency
        # 反转为 {0: "内科", 1: "外科", ...}
        self.id2label = {v: k for k, v in label_map.items()} if label_map else None

        # 查找 TRT 引擎文件
        engine_path = self.model_dir / "model.trt"
        if not engine_path.exists():
            raise FileNotFoundError(f"未找到 TRT 引擎文件: {engine_path}")

        logger.info(f"正在加载 TRT 引擎: {engine_path}")
        self.engine = TRT11InferenceEngine(str(engine_path))
        logger.info(f"TRT 引擎加载完成 (输入={list(self.engine.input_specs.keys())})")

    def predict(self, text: str) -> dict:
        """
        预测意图

        Args:
            text: 单条文本
        Returns:
            {"label": "内科", "score": 0.95}
        """
        start = time.perf_counter()

        # tokenize
        inputs = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # 移到 GPU
        input_dict = {
            "input_ids": inputs.input_ids.cuda(),
            "attention_mask": inputs.attention_mask.cuda(),
            "token_type_ids": inputs.token_type_ids.cuda(),
        }

        # TRT 推理
        outputs = self.engine.predict(input_dict)
        logits = outputs["logits"]

        # softmax 获取概率
        prob = torch.softmax(logits, dim=-1)[0]
        label_id = prob.argmax().item()
        score = prob[label_id].item()
        label = self.id2label[label_id] if self.id2label else f"LABEL_{label_id}"

        if self.log_latency:
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"推理耗时: {elapsed:.2f}ms | 输入: {text[:30]}...")

        return {"label": label, "score": score}


if __name__ == "__main__":
    import time
    from transformers import AutoTokenizer

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

    # 测试 TensorRT 推理
    print("\n" + "=" * 60)
    print("测试 TensorRT 推理...")
    engine = TensorRTEngine(str(MODEL_DIR), tokenizer)

    # 预热
    _ = engine.predict(["预热"])

    # 测试
    start = time.perf_counter()
    results = engine.predict(test_texts)
    trt_time = (time.perf_counter() - start) * 1000

    print(f"\nTensorRT 推理结果:")
    for text, result in zip(test_texts, results):
        print(f"  文本: {text}")
        print(f"  结果: {result}")
    print(f"  耗时: {trt_time:.2f}ms")
