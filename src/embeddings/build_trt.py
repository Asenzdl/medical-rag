"""
构建 TensorRT 引擎

Usage:
    python src/embeddings/build_trt.py
"""

import subprocess
from pathlib import Path

# 全局默认配置
DEFAULT_ONNX_PATH = Path(__file__).parent / "onnx" / "bge-base-zh-v1.5" / "none" / "model.onnx"
DEFAULT_OUTPUT_PATH = Path(__file__).parent / "onnx" / "bge-base-zh-v1.5" / "none" / "model.trt"

# 动态 Shape 配置（根据实际业务调整）
# 格式：tensor_name:batch_size x sequence_length
MIN_SHAPES = "input_ids:1x1,attention_mask:1x1,token_type_ids:1x1"
OPT_SHAPES = "input_ids:8x128,attention_mask:8x128,token_type_ids:8x128"
MAX_SHAPES = "input_ids:32x512,attention_mask:32x512,token_type_ids:32x512"

# 工作空间大小（MB）
WORKSPACE_SIZE = 1024


def build_trt_engine(
    onnx_path: str = DEFAULT_ONNX_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    min_shapes: str = MIN_SHAPES,
    opt_shapes: str = OPT_SHAPES,
    max_shapes: str = MAX_SHAPES,
    workspace_size: int = WORKSPACE_SIZE,
):
    """
    使用 trtexec 构建 TensorRT 引擎

    Args:
        onnx_path: 输入 ONNX 文件路径
        output_path: 输出 TRT 引擎文件路径
        min_shapes: 最小输入尺寸
        opt_shapes: 最优输入尺寸
        max_shapes: 最大输入尺寸
        workspace_size: 工作空间大小（MB）
    """
    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"ONNX 文件: {onnx_path}")
    print(f"输出路径: {output_path}")
    print(f"最小尺寸: {min_shapes}")
    print(f"最优尺寸: {opt_shapes}")
    print(f"最大尺寸: {max_shapes}")

    # 构建 trtexec 命令
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        f"--minShapes={min_shapes}",
        f"--optShapes={opt_shapes}",
        f"--maxShapes={max_shapes}",
        f"--memPoolSize=workspace:{workspace_size}",
    ]

    print(f"\n执行命令: {' '.join(cmd)}")
    print("=" * 60)

    # 执行命令
    result = subprocess.run(cmd, capture_output=True, text=True)

    # 输出结果
    if result.returncode == 0:
        print("构建成功！")
        print(f"TRT 引擎已保存到: {output_path}")
    else:
        print("构建失败！")
        print(f"错误信息:\n{result.stderr}")

    return result.returncode


if __name__ == "__main__":
    build_trt_engine()
