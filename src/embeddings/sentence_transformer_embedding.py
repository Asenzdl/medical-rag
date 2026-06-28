"""
SentenceTransformer Embedding 后端

支持 ONNX / PyTorch 推理，通过 provider 参数切换 CPU/GPU。
"""

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.base.logger import setup_logger

logger = setup_logger("SentenceTransformerEmbedding")

# Provider 映射
PROVIDER_MAP = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
}


class SentenceTransformerEmbedding:
    """SentenceTransformer Embedding 后端"""

    def __init__(self, model_dir: str, provider: str = "cpu", file_name: str = None):
        """
        初始化 SentenceTransformer 模型

        Args:
            model_dir: 模型目录
            provider: 推理设备（"cpu", "cuda", "tensorrt"）
            file_name: ONNX 文件名
        """
        self.model_dir = Path(model_dir)
        self.provider = PROVIDER_MAP.get(provider, provider)  # 支持简写和全称

        # 自动检测 ONNX 文件名
        if file_name is None:
            onnx_files = list(self.model_dir.glob("*.onnx"))
            if not onnx_files:
                raise FileNotFoundError(f"未找到 ONNX 文件: {self.model_dir}")
            file_name = onnx_files[0].name
        self.file_name = file_name

        logger.info(f"正在加载模型: {self.model_dir} (Provider={self.provider}, File={self.file_name})")
        self.model = SentenceTransformer(
            str(self.model_dir),
            backend="onnx",
            model_kwargs={
                "file_name": self.file_name,
                "provider": self.provider,
            }
        )

        # 获取实际使用的 Provider
        actual_provider = self.model[0].model.providers[0]
        self.device = "GPU" if "CUDA" in actual_provider or "TensorRT" in actual_provider else "CPU"
        logger.info(f"模型加载完成 (设备={self.device}, Provider={actual_provider})")

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        将文本转换为 embedding 向量

        Args:
            texts: 文本列表
        Returns:
            embeddings: shape (n, dim) 的 numpy 数组
        """
        return self.model.encode(texts)
