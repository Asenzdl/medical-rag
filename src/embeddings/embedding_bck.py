"""
Embedding 模型封装

Usage:
    from src.embeddings.embedding import EmbeddingModel

    # 使用默认配置（ONNX 后端）
    model = EmbeddingModel()
    embeddings = model.encode(["高血压怎么治疗"])

    # 自定义配置
    model = EmbeddingModel(
        model_dir="src/embeddings/onnx/bge-base-zh-v1.5",
        backend="onnx"
    )
"""

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.base.logger import setup_logger
import os
import sys

# 1. 动态获取你虚拟环境里 tensorrt_libs 的绝对路径
venv_base = sys.prefix
trt_libs_dir = os.path.join(venv_base, "Lib", "site-packages", "tensorrt_libs")

# 2. 如果路径存在，强行通知 Windows 系统：在这里面找 DLL！
if os.path.exists(trt_libs_dir):
    os.add_dll_directory(trt_libs_dir)

logger = setup_logger("EmbeddingModel")

# 默认配置
DEFAULT_MODEL_DIR = Path(__file__).parent / "onnx" / "bge-base-zh-v1.5" / "o3"
DEFAULT_BACKEND = "onnx"  # "onnx", "pytorch", "auto"
DEFAULT_FILE_NAME = "model_optimized.onnx"  # ONNX 文件名（由 export_onnx.py 生成）

# 推理提供程序（按优先级排序）：
# 1. "TensorrtExecutionProvider" — NVIDIA TensorRT 加速，最快（3~5ms），需安装 TensorRT SDK
# 2. "CUDAExecutionProvider"    — NVIDIA GPU 推理，快（10~15ms），需安装 onnxruntime-gpu + CUDA
# 3. "CPUExecutionProvider"     — CPU 推理，最慢（16~20ms），无额外依赖
DEFAULT_PROVIDER = "CUDAExecutionProvider"


class EmbeddingModel:
    """Embedding 模型封装（依赖注入）"""

    def __init__(self, model_dir: str = None, backend: str = None):
        """
        初始化 Embedding 模型

        Args:
            model_dir: 模型目录，默认为 src/embeddings/onnx/bge-base-zh-v1.5
            backend: 推理后端（"onnx", "pytorch", "auto"），默认为 "onnx"
        """
        self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
        self.backend = backend or DEFAULT_BACKEND

        logger.info(f"正在加载 Embedding 模型: {self.model_dir} (后端={self.backend})")
        self.model = SentenceTransformer(
            str(self.model_dir),
            backend=self.backend,
            model_kwargs={
                "file_name": DEFAULT_FILE_NAME,
                "provider": DEFAULT_PROVIDER,
            }
        )
        self.provider = self.model[0].model.providers[0]
        self.device = "GPU" if "CUDA" in self.provider or "TensorRT" in self.provider else "CPU"
        logger.info(f"Embedding 模型加载完成 (设备={self.device}, Provider={self.provider})")

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        将文本转换为 embedding 向量

        Args:
            texts: 文本列表
        Returns:
            embeddings: shape (n, dim) 的 numpy 数组
        """
        return self.model.encode(texts)


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("Embedding 模型测试")
    print("=" * 60)

    # 检查模型是否存在
    if not DEFAULT_MODEL_DIR.exists():
        print(f"模型未找到: {DEFAULT_MODEL_DIR}")
        print("请先运行: python src/embeddings/download_model.py")
        print("然后运行: python src/embeddings/export_onnx.py")
        exit(1)

    # 创建模型实例
    print(f"\n模型目录: {DEFAULT_MODEL_DIR}")
    print(f"推理后端: {DEFAULT_BACKEND}")
    model = EmbeddingModel()
    print(f"推理设备: {model.device}")

    # 测试用例
    test_cases = [
        ["高血压怎么治疗"],
        ["高血压怎么治疗", "糖尿病的症状有哪些"],
        ["感冒了怎么办", "发烧吃什么药", "头痛的原因"],
    ]

    for i, texts in enumerate(test_cases):
        print(f"\n测试 {i + 1}: {len(texts)} 条文本")
        print(f"输入: {texts}")

        # 测量推理时间
        start = time.perf_counter()
        embeddings = model.encode(texts)
        elapsed = (time.perf_counter() - start) * 1000

        # 输出结果
        print(f"输出形状: {embeddings.shape}")
        print(f"输出类型: {embeddings.dtype}")
        print(f"推理耗时: {elapsed:.2f} ms")

        # 验证维度
        assert embeddings.shape[0] == len(texts), "批量大小不匹配"
        assert embeddings.ndim == 2, "输出应为二维数组"

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
