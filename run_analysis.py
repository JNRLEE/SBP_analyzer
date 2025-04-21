#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# 此腳本用於分析指定的實驗結果並將分析結果儲存到指定位置
"""

# 修改導入方式，使用相對路徑，而非已安裝的套件
import sys
import os

# 添加專案根目錄到路徑中
sys.path.insert(0, os.path.abspath('.'))

from interfaces.analyzer_interface import SBPAnalyzer
import json
import time
import argparse
import shutil
from datetime import datetime

def setup_output_dirs(output_base, experiment_name):
    """建立輸出目錄結構"""
    
    # 創建實驗分析資料夾
    experiment_analysis_dir = os.path.join(output_base, experiment_name)
    os.makedirs(experiment_analysis_dir, exist_ok=True)
    
    # 創建子目錄
    subdirs = [
        'model_analysis',
        'training_analysis',
        'activation_analysis',
        'plots',
        'report'
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_analysis_dir, subdir), exist_ok=True)
    
    return experiment_analysis_dir

def create_metadata(experiment_dir, experiment_analysis_dir):
    """創建分析元數據文件"""
    
    metadata = {
        "original_path": os.path.abspath(experiment_dir),
        "analysis_time": datetime.now().isoformat(),
        "analyzer_version": "0.1.0"  # 假設版本號
    }
    
    # 寫入元數據文件
    metadata_path = os.path.join(experiment_analysis_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return metadata

def update_index(output_base, experiment_name, metadata):
    """更新或創建索引文件"""
    
    index_path = os.path.join(output_base, "index.json")
    index_data = {}
    
    # 如果索引文件已存在，先讀取它
    if os.path.exists(index_path):
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
        except json.JSONDecodeError:
            print(f"警告: 索引文件 {index_path} 格式錯誤，將創建新索引")
    
    # 更新索引
    index_data[experiment_name] = {
        "original_path": metadata["original_path"],
        "analysis_path": os.path.join(output_base, experiment_name),
        "analysis_time": metadata["analysis_time"]
    }
    
    # 寫入索引文件
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

def main():
    # 命令行參數解析
    parser = argparse.ArgumentParser(description='執行SBP實驗分析')
    parser.add_argument('--experiment', type=str, default='audio_swin_regression_20250417_142912', 
                        help='實驗資料夾名稱 (例如: audio_swin_regression_20250417_142912)')
    parser.add_argument('--output', type=str, default='analysis_results', 
                        help='分析結果輸出資料夾')
    parser.add_argument('--epochs', type=str, default=None, 
                        help='要分析的輪次，逗號分隔 (例如: 0,5,10)')
    parser.add_argument('--layers', type=str, default=None, 
                        help='要分析的層名稱，逗號分隔 (例如: patch_embed,layers.0)')
    args = parser.parse_args()
    
    # 構建實驗完整路徑
    experiment_dir = os.path.join('results', args.experiment)
    output_base = args.output
    
    # 確保輸出基礎目錄存在
    os.makedirs(output_base, exist_ok=True)
    
    # 處理epochs和layers參數
    epochs = [int(e) for e in args.epochs.split(',')] if args.epochs else None
    layers = args.layers.split(',') if args.layers else None
    
    print(f"開始分析實驗: {args.experiment}")
    print(f"分析結果將保存至: {output_base}/{args.experiment}")
    
    # 設置輸出目錄結構
    experiment_analysis_dir = setup_output_dirs(output_base, args.experiment)
    
    # 創建元數據
    metadata = create_metadata(experiment_dir, experiment_analysis_dir)
    
    # 更新索引
    update_index(output_base, args.experiment, metadata)
    
    # 創建分析器
    start_time = time.time()
    analyzer = SBPAnalyzer(experiment_dir=experiment_dir)
    
    try:
        # 運行分析
        analysis_results = analyzer.analyze(
            analyze_model_structure=True,
            analyze_training_history=True,
            analyze_hooks=True,
            epochs=epochs,
            layers=layers
        )
        
        # 生成報告
        report_path = analyzer.generate_report(
            output_dir=os.path.join(experiment_analysis_dir, 'report'),
            report_format='html'
        )
        
        # 處理並保存繪圖結果
        plots_dir = os.path.join(experiment_analysis_dir, 'plots')
        
        # 保存訓練損失曲線
        loss_curve = analysis_results.get_plot('loss_curve')
        if loss_curve:
            loss_curve.savefig(os.path.join(plots_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
        
        # 保存其他可能的圖表
        for plot_name in ['metric_curves', 'learning_rate', 'activation_distributions']:
            plot = analysis_results.get_plot(plot_name)
            if plot:
                plot.savefig(os.path.join(plots_dir, f'{plot_name}.png'), dpi=300, bbox_inches='tight')
        
        # 複製或保存模型分析結果
        if analysis_results.model_structure_results:
            model_analysis_path = os.path.join(experiment_analysis_dir, 'model_analysis', 'model_structure_analysis.json')
            with open(model_analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_results.model_structure_results, f, indent=2, ensure_ascii=False)
        
        # 複製或保存訓練分析結果
        if analysis_results.training_dynamics_results:
            training_analysis_path = os.path.join(experiment_analysis_dir, 'training_analysis', 'training_dynamics_analysis.json')
            with open(training_analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_results.training_dynamics_results, f, indent=2, ensure_ascii=False)
        
        # 複製或保存激活分析結果
        if analysis_results.intermediate_data_results:
            activation_analysis_path = os.path.join(experiment_analysis_dir, 'activation_analysis', 'activation_analysis.json')
            with open(activation_analysis_path, 'w', encoding='utf-8') as f:
                json.dump({'summary': analysis_results.intermediate_data_results}, f, indent=2, ensure_ascii=False)
        
        # 計算耗時
        elapsed = time.time() - start_time
        
        print(f"分析完成！耗時: {elapsed:.2f} 秒")
        print(f"HTML報告路徑: {report_path}")
        print(f"圖片輸出路徑: {plots_dir}")
        
        # 將分析完成的標記添加到元數據
        metadata["status"] = "completed"
        metadata["elapsed_time"] = elapsed
        
        # 更新元數據文件
        metadata_path = os.path.join(experiment_analysis_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
    except Exception as e:
        print(f"分析過程中出錯: {e}")
        
        # 將錯誤標記添加到元數據
        metadata["status"] = "failed"
        metadata["error"] = str(e)
        
        # 更新元數據文件
        metadata_path = os.path.join(experiment_analysis_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        raise

if __name__ == "__main__":
    main() 