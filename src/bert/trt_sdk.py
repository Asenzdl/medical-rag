import os
import sys
import numpy as np
import torch
import tensorrt as trt

class TRT11InferenceEngine:
    def __init__(self, engine_path: str, profile_index: int = 0):
        """
        TensorRT 11 风格现代推理引擎
        完全废弃 Index-bindings，全量拥抱 Name-based Tensor API 与现代内存规约
        """
        self.logger = trt.Logger(trt.Logger.ERROR)  # 生产环境推荐 ERROR 级别，拒绝日志刷屏
        self.engine_path = engine_path
        self.profile_index = profile_index
        
        # C++ 资源句柄初始化占位
        self.runtime = None
        self.engine = None
        self.context = None
        
        # 张量元数据规格账本
        self.input_specs = {}
        self.output_specs = {}
        
        self._build_engine_lifecycle()

    def _build_engine_lifecycle(self):
        # 1. 现代安全反序列化
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"未找到指定的 Engine 文件: {self.engine_path}")
            
        with open(self.engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        if not self.engine:
            raise RuntimeError(f"TensorRT 11 反序列化 {self.engine_path} 失败！")
            
        self.context = self.engine.create_execution_context()
        
        # 2. TensorRT 11 铁律一：必须完全废弃隐式 Batch 检查
        if hasattr(self.engine, "has_implicit_batch_dimension") and self.engine.has_implicit_batch_dimension:
            raise ValueError("TensorRT 11 不再支持隐式 Batch 模型！请确保导出 ONNX 时开启了 explicit_batch。")
            
        # 3. 激活动态尺寸优化 Profile（RAG 变长文本必备）
        if self.engine.num_optimization_profiles > 0:
            self.context.set_optimization_profile_async(
                self.profile_index, 
                torch.cuda.current_stream().cuda_stream
            )

        # 4. 基于 Name-based 提取 I/O 张量元数据 (num_io_tensors 是全新标准)
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            
            # 类型映射：全面兼容现代医疗大模型常用的 FP16 / BF16 / INT32 等类型
            torch_dtype = self._trt_to_torch_dtype(dtype)
            spec = {"shape": list(shape), "dtype": torch_dtype}
            
            if mode == trt.TensorIOMode.INPUT:
                self.input_specs[name] = spec
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_specs[name] = spec

    def _trt_to_torch_dtype(self, trt_dtype: trt.DataType) -> torch.dtype:
        """TensorRT 11 核心数据类型安全桥接"""
        mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int8: torch.int8,
            trt.bool: torch.bool,
        }
        # 现代支持：如果是 TensorRT 11 的 bfloat16 类型
        if hasattr(trt, 'bfloat16') and trt_dtype == trt.bfloat16:
            return torch.bfloat16
        return mapping.get(trt_dtype, torch.float32)

    def predict(self, input_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        纯 GPU 零拷贝异步推理管道 (Zero-Copy Async Pipeline)
        """
        # 获取 PyTorch 当前工作流的原始 C++ CUDA Stream 句柄
        current_stream = torch.cuda.current_stream()
        stream_handle = current_stream.cuda_stream
        
        output_dict = {}
        
        # 1. 输入张量绑定与动态尺寸更新
        for name, tensor in input_dict.items():
            if name not in self.input_specs:
                continue
            assert tensor.is_cuda, f"TensorRT 11 核心规范：输入张量 {name} 必须驻留在 GPU 显存上！"
            
            # 动态 Shape 核心：如果捕获到 -1（动态轴）或者输入 Shape 发生变化，动态刷新上下文
            if -1 in self.input_specs[name]["shape"] or self.context.get_tensor_shape(name) != list(tensor.shape):
                self.context.set_input_shape(name, tensor.shape)
                
            # TensorRT 11 唯一指定显存指针直接挂载 API
            self.context.set_tensor_address(name, tensor.data_ptr())
            
        # 2. 输出张量显存开辟（直接根据 Context 刷新后的真实尺寸）
        for name, spec in self.output_specs.items():
            # 必须通过 context 获取真实 shape，因为动态输入会导致输出维度联动刷新
            actual_output_shape = tuple(self.context.get_tensor_shape(name))
            
            # 利用 PyTorch 快速在 GPU 侧开辟空白容器
            out_tensor = torch.empty(
                actual_output_shape, 
                dtype=spec["dtype"], 
                device="cuda"
            )
            output_dict[name] = out_tensor
            self.context.set_tensor_address(name, out_tensor.data_ptr())
            
        # 3. 触发 TensorRT 11 异步核心推理 (完全交由 Stream 托管，非阻塞)
        success = self.context.execute_async_v3(stream_handle)
        if not success:
            raise RuntimeError("TensorRT 11 execute_v2 推理流水线执行异常！")
            
        return output_dict

    # =========================================================================
    # 现代 RAII 上下文管理器支持：杜绝智能体频繁调用/热重载导致的显存泄露漏洞
    # =========================================================================
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 强制解构 C++ 层的重量级指针
        if self.context: del self.context
        if self.engine: del self.engine
        if self.runtime: del self.runtime
        # 冲刷 PyTorch 缓存池
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import time
    from transformers import AutoTokenizer

    # 配置
    ENGINE_PATH = "src/embeddings/onnx/bge-base-zh-v1.5/none/model.trt"
    TOKENIZER_PATH = "src/embeddings/model/bge-base-zh-v1.5"

    print("=" * 60)
    print("TensorRT 推理测试")
    print("=" * 60)

    # 加载 tokenizer
    print(f"\n加载 Tokenizer: {TOKENIZER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # 加载 TRT 引擎
    print(f"加载 TRT 引擎: {ENGINE_PATH}")
    engine = TRT11InferenceEngine(ENGINE_PATH)
    print(f"输入张量: {engine.input_specs}")
    print(f"输出张量: {engine.output_specs}")

    # 测试用例
    test_cases = [
        ["高血压怎么治疗"],
        ["高血压怎么治疗", "糖尿病的症状有哪些"],
        ["感冒了怎么办", "发烧吃什么药", "头痛的原因"],
    ]

    for i, texts in enumerate(test_cases):
        print(f"\n测试 {i + 1}: {len(texts)} 条文本")
        print(f"输入: {texts}")

        # tokenize
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        input_dict = {
            "input_ids": inputs.input_ids.cuda(),
            "attention_mask": inputs.attention_mask.cuda(),
            "token_type_ids": inputs.token_type_ids.cuda(),
        }

        # 推理
        start = time.perf_counter()
        outputs = engine.predict(input_dict)
        elapsed = (time.perf_counter() - start) * 1000

        # 输出
        last_hidden_state = outputs["last_hidden_state"]
        print(f"输出形状: {last_hidden_state.shape}")
        print(f"推理耗时: {elapsed:.2f} ms")

        # mean pooling
        attention_mask = inputs.attention_mask.unsqueeze(-1).cuda()
        embeddings = (last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        print(f"Embedding 形状: {embeddings.shape}")

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)