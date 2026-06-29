"""
Loguru 日志配置

集中管理日志配置，应用启动时调用 configure_logger() 即可。

用法:
    from src.base.log_config import configure_logger
    configure_logger()

    # 任何模块
    from loguru import logger
    logger.info("xxx")
"""

import sys
import time
import inspect
import logging
from pathlib import Path
from functools import wraps, partial
from loguru import logger

import builtins
from loguru import logger

# # 1. 备份原生的 print（万一某些地方非要用原版的话）
# _raw_print = builtins.print

# # 2. 定义一个冒牌的 print
# def elegant_print(*args, **kwargs):
#     # 将 print 传入的多个参数用空格（或指定的 sep）拼接起来
#     sep = kwargs.get("sep", " ")
#     message = sep.join(str(arg) for arg in args)
    
#     # 核心：扔给 loguru，并用 opt(depth=1) 让它精准显示“到底是哪一行代码 print 的”
#     logger.opt(depth=1).info(message)

# # 3. 偷天换日：用冒牌货顶替全局内置的 print
# builtins.print = elegant_print



class InterceptHandler(logging.Handler):
    """拦截标准 logging，转发到 loguru"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 找到调用标准 logging 的源头代码位置
        frame = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def log_latency(func):
    """记录函数执行耗时（支持同步/异步、偏函数、异常时也能记录）"""

    # 1. 剥离 functools.partial，获取真实函数
    actual_func = func
    while isinstance(actual_func, partial):
        actual_func = actual_func.func

    # 2. 精准判断是否为异步
    is_async = inspect.iscoroutinefunction(actual_func) or (
        hasattr(actual_func, "__call__") and inspect.iscoroutinefunction(actual_func.__call__)
    )

    if is_async:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed = (time.perf_counter() - start) * 1000
                logger.opt(depth=1).debug(f"{func.__qualname__} 耗时: {elapsed:.2f}ms")
        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = (time.perf_counter() - start) * 1000
                logger.opt(depth=1).debug(f"{func.__qualname__} 耗时: {elapsed:.2f}ms")
        return sync_wrapper


SRC_PATH = Path(__file__).resolve().parent.parent
LOG_DIR = SRC_PATH / "logs"
_initialized = False


def configure_logger(
    console_level: str = "DEBUG",
    file_level: str = "INFO",
    error_level: str = "ERROR",
) -> None:
    """
    配置 loguru 日志

    Args:
        console_level: 控制台最低日志级别
        file_level: 常规日志文件最低级别
        error_level: 错误日志文件最低级别
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    # Windows 终端 UTF-8 输出
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 拦截标准 logging，统一走 loguru
    # logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # 清除默认 handler
    logger.remove()

    # 控制台输出（开发环境，带颜色）
    logger.add(
        sys.stdout,
        level=console_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        # format="<green>{time:H:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{name:<25}</cyan>:<cyan>{line:<4}</cyan> | <level>{message}</level>",
        colorize=True,
        enqueue=True,
    )

    # 常规日志文件（INFO+）
    logger.add(
        str(LOG_DIR / "app.log"),
        level=file_level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="100 MB",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
        enqueue=True,
    )

    # 错误日志文件（ERROR+，单独收集便于排查）
    logger.add(
        str(LOG_DIR / "error.log"),
        level=error_level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
        rotation="00:00",       # 每天凌晨轮转
        retention="30 days",
        encoding="utf-8",
        enqueue=True,
        backtrace=True,         # 完整堆栈追踪
        diagnose=False,         # 生产环境不泄露变量值
    )


if __name__ == "__main__":
    configure_logger()

    logger.debug("调试信息")
    logger.info("普通信息")
    logger.warning("警告")
    logger.error("错误")
    logger.success("成功")
