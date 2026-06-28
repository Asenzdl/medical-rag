"""
TensorRT Embedding 后端

基于 trt_sdk.py 的高性能 GPU 推理。
"""

import numpy as np
import torch
from pathlib import Path
from src.embeddings.trt_sdk import TRT11InferenceEngine
from src.base.logger import setup_logger

logger = setup_logger("TensorRTEmbedding")


class TensorRTEmbedding:
    """TensorRT Embedding 后端（仅支持 GPU）"""

    def __init__(self, model_dir: str, tokenizer):
        """
        初始化 TensorRT 模型

        Args:
            model_dir: TRT 引擎文件所在目录（包含 model.trt）
            tokenizer: 已加载的 tokenizer 实例
        """
        self.model_dir = Path(model_dir)
        self.tokenizer = tokenizer

        # 查找 TRT 引擎文件
        engine_path = self.model_dir / "model.trt"
        if not engine_path.exists():
            raise FileNotFoundError(f"未找到 TRT 引擎文件: {engine_path}")

        logger.info(f"正在加载 TRT 引擎: {engine_path}")
        self.engine = TRT11InferenceEngine(str(engine_path))
        logger.info(f"TRT 引擎加载完成 (输入={list(self.engine.input_specs.keys())})")

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        将文本转换为 embedding 向量

        Args:
            texts: 文本列表
        Returns:
            embeddings: shape (n, dim) 的 numpy 数组
        """
        # tokenize
        inputs = self.tokenizer(
            texts,
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
        last_hidden_state = outputs["last_hidden_state"]

        # mean pooling：对 token embeddings 按 attention_mask 加权平均
        attention_mask = inputs.attention_mask.unsqueeze(-1).cuda()
        embeddings = (last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

        return embeddings.cpu().numpy()
