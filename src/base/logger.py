from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

SRC_PATH = Path(__file__).resolve().parent.parent
LOG_PATH = SRC_PATH / "logs" / "app.log"
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
CONSOLE_LEVEL = logging.DEBUG
FILE_LEVEL = logging.WARNING

def setup_logger(
    name: str,
    log_file: Path = LOG_PATH,
) -> logging.Logger:
    """
    配置并返回一个日志记录器。
    如果同名 logger 已存在且带有处理器，则先清除再重新配置，确保一致性。
    """
    logger = logging.getLogger(name)
    # 清除已有处理器，避免重复和残留配置
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)  # 总阀值设为最低，由具体处理器控制

    formatter = logging.Formatter(DEFAULT_FORMAT)

    # 控制台处理器
    console = logging.StreamHandler()
    console.setLevel(CONSOLE_LEVEL)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # 文件处理器
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(FILE_LEVEL)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

if __name__ == "__main__":
    logger = setup_logger("Test")
    logger.debug("这是调试信息，通常用于开发")
    logger.info("程序运行正常")
    logger.warning("注意，可能有小问题")
    logger.error("发生错误")
    logger.critical("严重错误，程序可能崩溃")
    