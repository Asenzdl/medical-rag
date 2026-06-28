"""
ERNIE 模型导出为 ONNX 格式

用法:
    python -m src.bert.export_onnx
"""

import os
import sys

# 修复 Windows GBK 编码问题（torch.onnx 内部 print Unicode 字符）
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.base import setup_logger

logger = setup_logger("ExportONNX")

MODEL_DIR = Path(__file__).parent / "model"
ONNX_PATH = Path(__file__).parent / "onnx" / "model.onnx"


def export_onnx():
    """将 HuggingFace ERNIE 模型导出为 ONNX"""
    logger.info(f"加载模型: {MODEL_DIR}")

    # 加载模型和分词器
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model.eval()

    logger.info(f"模型架构: {model.config.architectures}")
    logger.info(f"分类数: {model.config.num_labels}")

    # 构造 dummy 输入（用于 trace）
    dummy_text = "这是一个测试输入，用于导出 ONNX 模型"
    dummy = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True,
    )

    logger.info(f"Dummy 输入 shape: input_ids={dummy['input_ids'].shape}")

    # 导出 ONNX（PyTorch 2.9+ 新 API）
    logger.info(f"开始导出 ONNX: {ONNX_PATH}")

    # 使用 dynamic_shapes 替代已弃用的 dynamic_axes
    # batch_size 和 seq_len 都设为动态
    batch_size = torch.export.Dim("batch_size", min=1, max=32)
    seq_len = torch.export.Dim("seq_len", min=1, max=128)

    onnx_program = torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(ONNX_PATH),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_shapes={
            "input_ids": {0: batch_size, 1: seq_len},
            "attention_mask": {0: batch_size, 1: seq_len},
        },
        dynamo=True,
        opset_version=18,
    )

    logger.info(f"ONNX 导出成功: {ONNX_PATH}")
    logger.info(f"文件大小: {ONNX_PATH.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    export_onnx()
