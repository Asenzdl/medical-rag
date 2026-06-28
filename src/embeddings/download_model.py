"""
下载 HuggingFace 模型到本地目录

Usage:
    python src/embeddings/download_model.py --model BAAI/bge-base-zh-v1.5
"""

import argparse
from pathlib import Path
from transformers import AutoModel, AutoTokenizer


def download_model(model_name: str, save_dir: str = None):
    """
    从 HuggingFace 下载模型到本地

    Args:
        model_name: HuggingFace 模型名称（如 "BAAI/bge-base-zh-v1.5"）
        save_dir: 保存目录，默认为 src/embeddings/model/{model_name}
    """
    if save_dir is None:
        # 提取模型简称（去掉组织前缀）
        model_short = model_name.split("/")[-1]
        save_dir = Path(__file__).parent / "model" / model_short
    else:
        save_dir = Path(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model: {model_name}")
    print(f"Save directory: {save_dir}")

    # 下载 tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_dir)

    # 下载模型
    print("Downloading model...")
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(save_dir)

    print(f"Model saved to: {save_dir}")
    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HuggingFace model")
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-base-zh-v1.5",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Save directory (default: src/embeddings/model/{model_short})"
    )

    args = parser.parse_args()
    download_model(args.model, args.save_dir)
