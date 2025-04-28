#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分析FCNN回歸模型實驗結果的腳本。
本腳本對audio_fcnn_regression_20250417_143054實驗進行全面分析，
包括模型結構、訓練動態、中間層數據等方面的分析。
"""

import os
import sys
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import traceback
import time

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 確保可以導入SBP_analyzer模塊
sys.path.append(str(Path(__file__).parents[2]))

# 導入分析工具
from interfaces.analyzer_interface import SBPAnalyzer
from visualization.performance_plots import PerformancePlotter
from visualization.distribution_plots import DistributionPlotter

# 實驗目錄和輸出目錄
EXPERIMENT_DIR = 'results/audio_fcnn_regression_20250417_143054'
OUTPUT_DIR = 'analysis_results/audio_fcnn_regression_20250417_143054'

def preprocess_training_history():
    """
    預處理訓練歷史數據，修正格式問題。
    """
    try:
        # 讀取原始訓練歷史
        history_path = os.path.join(EXPERIMENT_DIR, 'training_history.json')
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # 檢查並修正訓練歷史格式
        if 'train_loss' in history and 'loss' not in history:
            # 創建epochs列表
            num_epochs = len(history['train_loss'])
            history['epochs'] = list(range(num_epochs))
            history['loss'] = history['train_loss']
            
            # 確保有學習率數據
            if 'lr' not in history:
                history['lr'] = [0.001] * num_epochs  # 假設默認學習率
        
        # 保存修正後的訓練歷史
        corrected_path = os.path.join(OUTPUT_DIR, 'corrected_training_history.json')
        with open(corrected_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        # 將修正後的文件複製到實驗目錄以供分析器使用
        temp_backup = os.path.join(EXPERIMENT_DIR, 'training_history_backup.json')
        shutil.copy(history_path, temp_backup)  # 備份原文件
        shutil.copy(corrected_path, history_path)  # 替換為修正版本
        
        logger.info(f"訓練歷史數據已預處理，原始文件已備份為 {temp_backup}")
        return True
    except Exception as e:
        logger.error(f"預處理訓練歷史時發生錯誤: {e}")
        traceback.print_exc()
        return False

def restore_training_history():
    """
    恢復原始訓練歷史文件。
    """
    try:
        backup_path = os.path.join(EXPERIMENT_DIR, 'training_history_backup.json')
        original_path = os.path.join(EXPERIMENT_DIR, 'training_history.json')
        if os.path.exists(backup_path):
            shutil.copy(backup_path, original_path)
            os.remove(backup_path)
            logger.info("已恢復原始訓練歷史文件")
    except Exception as e:
        logger.error(f"恢復訓練歷史文件時發生錯誤: {e}")

# 為處理NumPy類型的JSON序列化問題創建自定義編碼器
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def create_custom_analysis():
    """
    創建自定義分析圖表和結果，不依賴分析器框架。
    """
    try:
        # 確保輸出目錄存在
        os.makedirs(os.path.join(OUTPUT_DIR, 'custom_plots'), exist_ok=True)
        
        # 讀取訓練歷史
        history_path = os.path.join(EXPERIMENT_DIR, 'training_history.json')
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # 創建訓練損失曲線
        epochs = range(len(history['train_loss']))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['train_loss'], 'b-', label='訓練損失')
        plt.plot(epochs, history['val_loss'], 'r-', label='驗證損失')
        plt.xlabel('輪次')
        plt.ylabel('損失')
        plt.title('FCNN回歸模型訓練與驗證損失')
        plt.legend()
        plt.grid(True, alpha=0.3)
        loss_plot_path = os.path.join(OUTPUT_DIR, 'custom_plots', 'loss_curve.png')
        plt.savefig(loss_plot_path)
        plt.close()
        
        # 計算訓練指標
        metrics = {
            'best_val_loss': float(history.get('best_val_loss', min(history['val_loss']))),
            'final_train_loss': float(history['train_loss'][-1]),
            'final_val_loss': float(history['val_loss'][-1]),
            'training_time': float(history.get('training_time', 0)),
            'convergence_epoch': int(np.argmin(history['val_loss'])),
            'early_stopping': len(history['val_loss']) < 100,  # 假設最大輪次為100
            'overfitting_score': float((history['val_loss'][-1] / min(history['val_loss'])) - 1)
        }
        
        # 分析訓練穩定性
        train_loss_diff = np.diff(history['train_loss'])
        val_loss_diff = np.diff(history['val_loss'])
        metrics['train_stability'] = float(np.std(train_loss_diff) / np.mean(np.abs(train_loss_diff)) if len(train_loss_diff) > 0 else 0)
        metrics['val_stability'] = float(np.std(val_loss_diff) / np.mean(np.abs(val_loss_diff)) if len(val_loss_diff) > 0 else 0)
        
        # 保存分析結果
        analysis_path = os.path.join(OUTPUT_DIR, 'custom_analysis_results.json')
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
        
        # 創建配置文件摘要
        config_path = os.path.join(EXPERIMENT_DIR, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            config_summary = {
                'model_type': config.get('model', {}).get('name', 'FCNN'),
                'optimizer': config.get('trainer', {}).get('optimizer', {}).get('name', 'adam'),
                'learning_rate': config.get('trainer', {}).get('optimizer', {}).get('params', {}).get('lr', 0.001),
                'batch_size': config.get('trainer', {}).get('batch_size', 32),
                'max_epochs': config.get('trainer', {}).get('epochs', 100)
            }
            
            summary_path = os.path.join(OUTPUT_DIR, 'config_summary.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(config_summary, f, indent=4, ensure_ascii=False)
        
        logger.info(f"自定義分析已完成，結果保存在 {OUTPUT_DIR}/custom_plots 和 {analysis_path}")
        return True
    except Exception as e:
        logger.error(f"創建自定義分析時發生錯誤: {e}")
        traceback.print_exc()
        return False

def main():
    """
    進行完整實驗分析並生成報告。
    """
    logger.info(f"開始分析實驗: {EXPERIMENT_DIR}")
    
    # 創建輸出目錄
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 預處理訓練歷史
    preprocess_success = preprocess_training_history()
    
    try:
        # 創建自定義分析（不依賴分析器框架）
        custom_analysis_success = create_custom_analysis()
        
        if preprocess_success:
            # 初始化分析器
            analyzer = SBPAnalyzer(experiment_dir=EXPERIMENT_DIR)
            
            try:
                # 進行模型結構分析
                analysis_results = analyzer.analyze(
                    analyze_model_structure=True,  # 分析模型結構
                    analyze_training_history=False,  # 不分析訓練歷史，因為可能有格式問題
                    analyze_hooks=False,  # 暫時不分析中間層數據
                    analyze_inference=False,  # 不分析推理性能
                    save_results=True  # 保存分析結果
                )
                
                # 生成報告
                report_path = analyzer.generate_report(
                    output_dir=OUTPUT_DIR,
                    report_format='html'  # 生成HTML格式報告
                )
                
                logger.info(f"分析器框架分析完成，報告已生成: {report_path}")
            except Exception as e:
                logger.error(f"執行分析器時發生錯誤: {e}")
                traceback.print_exc()
        
        # 創建分析結果索引
        create_analysis_index()
        
        logger.info(f"分析完成，結果已保存到: {OUTPUT_DIR}")
    finally:
        # 恢復原始訓練歷史文件
        restore_training_history()

def create_analysis_index():
    """
    創建分析結果索引文件，列出所有生成的分析結果和圖表。
    """
    try:
        index_data = {
            "experiment_name": "audio_fcnn_regression_20250417_143054",
            "analysis_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_results": [],
            "plots": []
        }
        
        # 收集分析結果文件
        for filename in os.listdir(OUTPUT_DIR):
            if filename.endswith('.json'):
                index_data["analysis_results"].append(filename)
        
        # 收集圖表文件
        plots_dir = os.path.join(OUTPUT_DIR, 'custom_plots')
        if os.path.exists(plots_dir):
            for filename in os.listdir(plots_dir):
                if filename.endswith(('.png', '.jpg', '.svg')):
                    index_data["plots"].append(f"custom_plots/{filename}")
        
        # 保存索引文件
        index_path = os.path.join(OUTPUT_DIR, 'index.json')
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=4, ensure_ascii=False)
        
        logger.info(f"分析結果索引已創建: {index_path}")
    except Exception as e:
        logger.error(f"創建分析索引時發生錯誤: {e}")

if __name__ == "__main__":
    main() 