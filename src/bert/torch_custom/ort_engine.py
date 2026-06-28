"""
基于 ONNX Runtime + TensorRT 的推理引擎

用法:
    from src.bert.ort_engine import BertInferenceEngine
    engine = BertInferenceEngine()
    result = engine.predict("高血压怎么治疗")
"""

import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import onnxruntime as ort

from src.base import setup_logger

logger = setup_logger("ORTEngine")

MODEL_DIR = Path(__file__).parent / "model"
ONNX_PATH = Path(__file__).parent / "onnx" / "model.onnx"

# 科室标签映射
ID2LABEL = {
    0: "儿科",
    1: "内科",
    2: "外科",
    3: "妇产科",
    4: "男科",
    5: "肿瘤科",
}


class BertInferenceEngine:
    """基于 ONNX Runtime + TensorRT 的推理引擎"""

    def __init__(
        self,
        model_path: Path = MODEL_DIR,
        onnx_path: Path = ONNX_PATH,
        use_tensorrt: bool = True,
    ):
        self.id2label = ID2LABEL

        # 加载分词器
        logger.info(f"加载分词器: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 选择 Provider
        if use_tensorrt:
            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
            logger.info("使用 TensorRT + CUDA 加速")
        else:
            providers = ["CUDAExecutionProvider"]
            logger.info("使用 CUDA 加速")

        # 加载 ONNX 模型
        logger.info(f"加载 ONNX 模型: {onnx_path}")
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=providers,
        )

        # 获取输入输出名称
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        logger.info(f"输入: {self.input_names}")
        logger.info(f"输出: {self.output_names}")

        logger.info("推理引擎初始化完成")

    def predict(self, text: str) -> dict:
        """
        预测单条文本的科室分类

        Args:
            text: 输入文本

        Returns:
            dict: {"label": "内科", "confidence": 0.95, "probabilities": {...}}
        """
        # 1. Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=128,
        )

        # 2. 准备输入
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }

        # 3. 推理
        outputs = self.session.run(self.output_names, ort_inputs)
        logits = outputs[0][0]

        # 4. 后处理
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        pred_id = int(np.argmax(probs))

        return {
            "label": self.id2label[pred_id],
            "confidence": float(probs[pred_id]),
            "probabilities": {self.id2label[i]: float(p) for i, p in enumerate(probs)},
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """批量预测"""
        # 1. 批量 Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=128,
        )

        # 2. 准备输入
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }

        # 3. 推理
        outputs = self.session.run(self.output_names, ort_inputs)
        logits_batch = outputs[0]

        # 4. 后处理
        results = []
        for i in range(len(texts)):
            logits = logits_batch[i]
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            pred_id = int(np.argmax(probs))

            results.append({
                "label": self.id2label[pred_id],
                "confidence": float(probs[pred_id]),
                "probabilities": {self.id2label[j]: float(p) for j, p in enumerate(probs)},
            })

        return results


if __name__ == "__main__":
    # 测试推理
    engine = BertInferenceEngine(use_tensorrt=True)

    test_texts = [
        "高血压怎么治疗",
        "孩子发烧怎么办",
        "骨折后怎么护理",
    ]

    print("\n=== 单条推理测试 ===")
    for text in test_texts:
        result = engine.predict(text)
        print(f"\n输入: {text}")
        print(f"预测: {result['label']} (置信度: {result['confidence']:.2%})")

    print("\n=== 批量推理测试 ===")
    results = engine.predict_batch(test_texts)
    for text, result in zip(test_texts, results):
        print(f"{text} -> {result['label']} ({result['confidence']:.2%})")
