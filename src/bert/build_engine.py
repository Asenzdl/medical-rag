"""
将 ONNX 模型转换为 TensorRT Engine

用法:
    python -m src.bert.build_engine
"""

import sys
import os

# 修复 Windows 编码问题
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
import tensorrt as trt

from src.base import setup_logger

logger = setup_logger("BuildEngine")

MODEL_DIR = Path(__file__).parent
ONNX_PATH = MODEL_DIR / "model.onnx"
ENGINE_PATH = MODEL_DIR / "model.trt"


def build_engine(
    onnx_path: Path = ONNX_PATH,
    engine_path: Path = ENGINE_PATH,
    max_batch_size: int = 32,
    max_seq_len: int = 128,
) -> None:
    """将 ONNX 模型转换为 TensorRT Engine

    TensorRT for RTX 使用 strong typing 模式，精度由输入 tensor 类型决定，
    TensorRT 会自动选择最优实现（FP32/FP16/BF16）。
    """

    logger.info(f"TensorRT 版本: {trt.__version__}")
    logger.info(f"ONNX 模型: {onnx_path}")
    logger.info(f"输出路径: {engine_path}")

    # 1. 创建 logger、builder、network、config
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network()  # 显式 batch 是默认行为
    config = builder.create_builder_config()

    # 2. 解析 ONNX 模型（使用 parse_from_file 处理外部数据文件）
    parser = trt.OnnxParser(network, trt_logger)
    logger.info("解析 ONNX 模型...")

    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            logger.error(f"ONNX 解析错误: {parser.get_error(i)}")
        raise RuntimeError("ONNX 解析失败")

    logger.info(f"ONNX 解析成功，层数: {network.num_layers}")

    # 3. 配置（TensorRT for RTX 自动选择最优精度）
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # 4. 设置动态 shapes（seq_len 固定，只让 batch 动态）
    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids", (1, max_seq_len), (8, max_seq_len), (max_batch_size, max_seq_len))
    profile.set_shape("attention_mask", (1, max_seq_len), (8, max_seq_len), (max_batch_size, max_seq_len))
    config.add_optimization_profile(profile)
    logger.info(f"优化配置: batch=[1,{max_batch_size}], seq_len={max_seq_len}（固定）")

    # 5. 构建 Engine（TensorRT for RTX 使用 build_serialized_network）
    logger.info("构建 TensorRT Engine（可能需要几分钟）...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Engine 构建失败")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    logger.info(f"Engine 构建成功: {engine_path}")
    logger.info(f"文件大小: {engine_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    build_engine()
