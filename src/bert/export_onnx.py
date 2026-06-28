"""
导出 HuggingFace 模型为 ONNX 格式

目录结构：
    src/embeddings/onnx/{model_name}/{variant}/
    - none:        无优化（FP32，标准 Attention，兼容 TensorRT）
    - o1:          基本优化（冗余节点消除、常量折叠）
    - o2:          扩展优化（O1 + 算子融合、注意力融合）
    - o3:          GELU 近似（牺牲少量精度换取速度）
    - o4:          FP16 混合精度（需要 GPU，最大加速）
    - dynamic:     动态量化（INT8，不需要校准数据）
    - static:      静态量化（INT8，需要校准数据，精度更高）
    - weights_only: 仅权重量化（INT8，只量化权重）

Usage:
    from src.embeddings.export_onnx import export_onnx

    # 无优化（兼容 TensorRT）
    export_onnx(optimize=None)

    # 优化
    export_onnx(optimize="o1")
    export_onnx(optimize="o3")

    # 量化
    export_onnx(quantize="dynamic")
"""

from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig, AutoOptimizationConfig

# 全局默认配置
DEFAULT_MODEL_DIR = Path(__file__).parent / "model" / "chinese-roberta-wwm-ext-distilled-27745"

# 量化方式（三选一，或 None 不量化）
# 输出精度：INT8
# - "dynamic": 动态量化，不需要校准数据，推理时动态量化激活值
# - "static": 静态量化，需要校准数据，预先计算量化参数，精度更高
# - "weights_only": 仅权重量化，只量化模型权重，不量化激活值
DEFAULT_QUANTIZE = None  # None, "dynamic", "static", "weights_only"

# 优化级别（四选一，或 None 不优化）
# 输出精度：O1-O3 保持 FP32，O4 输出 FP16
# - None: 不优化（标准 Attention，兼容 TensorRT）
# - O1: 基本优化（冗余节点消除、常量折叠）
# - O2: 扩展优化（O1 + 算子融合、注意力融合）
# - O3: O2 + GELU 近似（牺牲少量精度换取速度）
# - O4: O3 + FP16 混合精度（需要 GPU，最大加速）
DEFAULT_OPTIMIZE = None  # None, "o1", "o2", "o3", "o4"


def export_onnx(
    model_dir: str = DEFAULT_MODEL_DIR,
    quantize: str = DEFAULT_QUANTIZE,
    optimize: str = DEFAULT_OPTIMIZE,
):
    """
    导出 HuggingFace 模型为 ONNX 格式

    输出目录：src/embeddings/onnx/{model_name}/{variant}

    Args:
        model_dir: 输入模型目录（HuggingFace 格式）
        quantize: 量化方式（None, "dynamic", "static", "weights_only"）
        optimize: 优化级别（None, "o1", "o2", "o3", "o4"）
    """
    model_dir = Path(model_dir)

    # 确定 variant 名称（优化/量化/无）
    if optimize:
        variant = optimize.lower()
    elif quantize:
        variant = quantize.lower()
    else:
        variant = "none"

    # 输出目录：src/embeddings/onnx/{model_name}/{variant}
    model_name = model_dir.name
    output_dir = Path(__file__).parent / "onnx" / model_name / variant
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"模型目录: {model_dir}")
    print(f"输出目录: {output_dir}")
    print(f"量化方式: {quantize}")
    print(f"优化级别: {optimize}")

    # 导出 ONNX（CPU 导出）
    print("正在导出 ONNX 模型...")
    model = ORTModelForSequenceClassification.from_pretrained(
        model_dir,
        export=True
    )

    # 优化处理（优先级：optimize > quantize）
    if optimize:
        print(f"正在应用 {optimize.upper()} 优化...")
        optimizer = ORTOptimizer.from_pretrained(model)
        optimization_config = getattr(AutoOptimizationConfig, optimize.upper())()
        optimizer.optimize(save_dir=output_dir, optimization_config=optimization_config)
    elif quantize:
        # 量化处理
        print(f"正在应用 {quantize} 量化...")
        quantizer = ORTQuantizer.from_pretrained(model)

        if quantize == "dynamic":
            # 动态量化：不需要校准数据
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)
        elif quantize == "static":
            # 静态量化：需要校准数据（此处使用默认配置）
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=True)
        elif quantize == "weights_only":
            # 仅权重量化
            qconfig = AutoQuantizationConfig.avx512_vnni(
                is_static=False,
                operators_to_quantize=["MatMul"]  # 只量化权重相关的算子
            )
        else:
            raise ValueError(f"未知的量化方式: {quantize}")

        quantizer.quantize(save_dir=output_dir, quantization_config=qconfig)
    else:
        # 无量化
        print("保存 ONNX 模型（无优化）...")
        model.save_pretrained(output_dir)

    print(f"ONNX 模型已保存到: {output_dir}")
    return output_dir


if __name__ == "__main__":
    export_onnx()
