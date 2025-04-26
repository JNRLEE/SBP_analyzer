#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
簡單測試訓練動態分析器功能
"""

import os
import sys
import tempfile
import json
import shutil

# 添加專案根目錄到路徑中
sys.path.insert(0, os.path.abspath('.'))

from analyzer.training_dynamics_analyzer import TrainingDynamicsAnalyzer

def main():
    """執行訓練動態分析器的簡單測試"""
    print("開始測試訓練動態分析器...")
    
    # 創建臨時目錄作為測試實驗目錄
    test_dir = tempfile.mkdtemp()
    print(f"創建臨時測試目錄: {test_dir}")
    
    try:
        # 創建虛擬的訓練歷史數據
        train_loss = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.9, 0.8, 0.7, 0.6]
        val_loss = [12.0, 9.0, 7.0, 5.0, 3.0, 2.5, 2.2, 2.1, 2.0, 1.9]
        accuracy = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8]
        
        # 模擬訓練歷史JSON文件
        training_history = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'training_time': 100.0,
            'best_val_loss': 1.9
        }
        
        # 將訓練歷史保存為JSON文件
        with open(os.path.join(test_dir, 'training_history.json'), 'w') as f:
            json.dump(training_history, f)
        
        print(f"創建模擬訓練歷史文件: {os.path.join(test_dir, 'training_history.json')}")
        
        # 創建分析輸出目錄
        analysis_dir = os.path.join(test_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # 初始化分析器
        analyzer = TrainingDynamicsAnalyzer(test_dir)
        print("初始化訓練動態分析器完成")
        
        # 測試載入訓練歷史
        loaded_history = analyzer.load_training_history()
        print("載入訓練歷史完成")
        
        # 測試預處理指標
        metrics_df = analyzer.preprocess_metrics()
        print(f"預處理指標完成，指標形狀: {metrics_df.shape}")
        
        # 測試完整分析，不保存結果
        results = analyzer.analyze(save_results=False)
        print("執行完整分析完成")
        
        # 檢查結果
        if 'convergence_analysis' in results and 'train_loss' in results['convergence_analysis']:
            conv_results = results['convergence_analysis']['train_loss']
            print(f"訓練損失收斂點: {conv_results.get('convergence_point')}")
            print(f"訓練是否收斂: {conv_results.get('has_converged')}")
            print(f"訓練最小損失值: {conv_results.get('min_value')}")
        
        # 測試可視化，不保存圖像
        fig = analyzer.visualize_metrics(output_path=None)
        print("可視化指標完成")
        
        print("訓練動態分析器測試成功!")
        
    finally:
        # 清理臨時目錄
        shutil.rmtree(test_dir)
        print(f"清理臨時測試目錄: {test_dir}")

if __name__ == "__main__":
    main() 