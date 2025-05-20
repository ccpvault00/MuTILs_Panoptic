# 創建一個測試腳本，命名為 test_mutils_performance.py
from mutils_panoptic.performance_monitor import timing_decorator
import importlib
import os
import sys
import time
import logging

# 確保 MuTILs_Panoptic 在 Python 路徑中
sys.path.append('/home/chang-chia-ping/code_base')

# 導入所需模組
from MuTILs_Panoptic.mutils_panoptic.MuTILsWSIRunner import MuTILsWSIRunner
from MuTILs_Panoptic.configs.MuTILsWSIRunConfigs import RunConfigs
from histolab.slide import SlideSet, Slide

# 設置日誌
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(message)s',
                    filename='mutils_performance_test.log',
                    filemode='w')
logger = logging.getLogger(__name__)

# 裝飾 MuTILsWSIRunner 的關鍵方法
MuTILsWSIRunner.run_slide = timing_decorator(MuTILsWSIRunner.run_slide)
MuTILsWSIRunner.run_single_model = timing_decorator(MuTILsWSIRunner.run_single_model)
MuTILsWSIRunner._load_or_extract_roi_locs = timing_decorator(MuTILsWSIRunner._load_or_extract_roi_locs)
MuTILsWSIRunner.run_parallel_models = timing_decorator(MuTILsWSIRunner.run_parallel_models)

# 測量 ROIPreProcessor 關鍵方法
from MuTILs_Panoptic.mutils_panoptic.MuTILsInference import ROIPreProcessor
ROIPreProcessor.run = timing_decorator(ROIPreProcessor.run)
ROIPreProcessor.run_roi = timing_decorator(ROIPreProcessor.run_roi)
ROIPreProcessor._provide_input = timing_decorator(ROIPreProcessor._provide_input)
ROIPreProcessor._get_tile_ignore = timing_decorator(ROIPreProcessor._get_tile_ignore)
ROIPreProcessor._maybe_color_normalize = timing_decorator(ROIPreProcessor._maybe_color_normalize)

# 測量 ROIInferenceProcessor 關鍵方法
from MuTILs_Panoptic.mutils_panoptic.MuTILsInference import ROIInferenceProcessor
ROIInferenceProcessor.run = timing_decorator(ROIInferenceProcessor.run)
ROIInferenceProcessor.inference = timing_decorator(ROIInferenceProcessor.inference)

# 測量 ROIPostProcessor 關鍵方法
from MuTILs_Panoptic.mutils_panoptic.MuTILsInference import ROIPostProcessor
ROIPostProcessor.run = timing_decorator(ROIPostProcessor.run)
ROIPostProcessor.run_roi = timing_decorator(ROIPostProcessor.run_roi)
ROIPostProcessor.refactor_inference = timing_decorator(ROIPostProcessor.refactor_inference)
ROIPostProcessor._get_nuclei_objects_mask = timing_decorator(ROIPostProcessor._get_nuclei_objects_mask)
ROIPostProcessor._refactor_nuclear_hpf_mask = timing_decorator(ROIPostProcessor._refactor_nuclear_hpf_mask)
ROIPostProcessor._aggregate_pixels_for_nucleus = timing_decorator(ROIPostProcessor._aggregate_pixels_for_nucleus)
ROIPostProcessor._maybe_save_roi_preds = timing_decorator(ROIPostProcessor._maybe_save_roi_preds)
ROIPostProcessor.get_nuclei_props_df = timing_decorator(ROIPostProcessor.get_nuclei_props_df)

# 測量實用工具函數
from MuTILs_Panoptic.utils.MiscRegionUtils import get_objects_from_binmask, get_region_within_x_pixels
get_objects_from_binmask = timing_decorator(get_objects_from_binmask)
get_region_within_x_pixels = timing_decorator(get_region_within_x_pixels)

# 主函數
def main():
    # 記錄開始時間
    logger.info("Starting MuTILs performance test")
    start_time = time.time()
    
    # 獲取設定
    runconfig = RunConfigs()
    config = runconfig.get_config()
    
    # 初始化 MuTILsWSIRunner
    runner = MuTILsWSIRunner(config)
    runner.check_gpus()
    
    # 獲取幻燈片集
    slides = SlideSet(
        slides_path=config.slides_path,
        processed_path=config.base_savedir,
        valid_extensions=config.valid_extensions,
        keep_slides=config.slide_names,
        slide_kwargs={"use_largeimage": True},
    )
    
    # 處理第一張幻燈片以進行性能測試
    for slidx, slide in enumerate(slides):
        if slidx > 0:  # 只處理第一張幻燈片
            break
            
        # 幻燈片監控信息
        slmonitor = f"{config.monitor} slide {slidx + 1} of {len(slides)}: {slide.name}"
        logger.info(f"*** {slmonitor} ***")
        
        # 運行幻燈片
        runner.run_slide(slmonitor, slide)
        
        # 只測試一張幻燈片
        break
    
    # 記錄總運行時間
    end_time = time.time()
    logger.info(f"Total runtime: {end_time - start_time:.3f} seconds")

if __name__ == "__main__":
    main()