# 創建一個獨立的 Python 檔案，命名為 performance_monitor.py
import time
import functools
import logging

# 設置基本的日誌格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("performance_monitor")

# 可以選擇將日誌寫入檔案
file_handler = logging.FileHandler("performance_timing.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(file_handler)

def timing_decorator(func):
    """測量函數執行時間的裝飾器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        # 記錄函數名稱和執行時間
        logger.info(f"Function {func.__name__} took {execution_time:.3f} seconds to run")
        return result
    return wrapper