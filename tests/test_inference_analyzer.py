"""
測試推理分析器 (InferenceAnalyzer) 模組。

此模組提供用於測試深度學習模型推理性能分析器的測試案例。
"""

import os
import json
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import torch
import matplotlib.pyplot as plt

from analyzer.inference_analyzer import InferenceAnalyzer


class TestInferenceAnalyzer(unittest.TestCase):
    """
    測試 InferenceAnalyzer 類的功能。
    """
    
    def setUp(self):
        """
        設置測試環境和樣本數據。
        """
        # 創建臨時目錄
        self.temp_dir = tempfile.mkdtemp()
        
        # 創建樣本推理日誌
        self.inference_logs = {
            'timing': {
                'inference_times': [15.2, 14.8, 15.5, 14.9, 15.0, 16.2, 14.7, 15.3, 14.5, 15.1],
                'initialization_time': 120.5,
                'preprocessing_times': [3.2, 3.1, 3.0, 3.3, 3.1, 3.2, 3.0, 3.1, 3.2, 3.1],
                'postprocessing_times': [1.1, 1.2, 1.0, 1.1, 1.1, 1.3, 1.1, 1.2, 1.0, 1.1]
            },
            'batch_size': 32,
            'memory_usage': {
                'gpu_memory': {
                    'peak': 1024,
                    'allocated': 768,
                    'reserved': 1536
                },
                'cpu_memory': {
                    'peak': 1536,
                    'rss': 1280,
                    'vms': 2048
                }
            },
            'outputs': [
                [0.1, 0.2, 0.7],
                [0.05, 0.15, 0.8],
                [0.2, 0.3, 0.5],
                [0.01, 0.09, 0.9],
                [0.3, 0.2, 0.5]
            ],
            'batch_size_tests': {
                '1': {
                    'inference_times': [5.0, 5.1, 4.9, 5.0, 5.2],
                    'memory_usage': {'peak': 512}
                },
                '8': {
                    'inference_times': [10.0, 10.2, 9.8, 10.1, 10.0],
                    'memory_usage': {'peak': 768}
                },
                '32': {
                    'inference_times': [15.0, 15.2, 14.9, 15.1, 15.0],
                    'memory_usage': {'peak': 1024}
                },
                '64': {
                    'inference_times': [22.0, 22.3, 21.8, 22.1, 22.0],
                    'memory_usage': {'peak': 1536}
                }
            },
            'input_shape_tests': {
                '1_224_224_3': {
                    'inference_times': [5.0, 5.1, 4.9, 5.0, 5.2],
                    'memory_usage': {'peak': 768}
                },
                '1_320_320_3': {
                    'inference_times': [8.0, 8.2, 7.9, 8.1, 8.0],
                    'memory_usage': {'peak': 1024}
                },
                '1_512_512_3': {
                    'inference_times': [15.0, 15.3, 14.8, 15.2, 15.1],
                    'memory_usage': {'peak': 1536}
                }
            },
            'accuracy': {
                'overall_accuracy': 0.85,
                'class_accuracies': {
                    '0': 0.78,
                    '1': 0.92,
                    '2': 0.85
                },
                'confusion_matrix': [
                    [78, 12, 10],
                    [6, 92, 2],
                    [10, 5, 85]
                ]
            },
            'hardware_comparison': {
                'cpu': {
                    'inference_times': [25.0, 25.2, 24.9, 25.1, 25.0],
                    'batch_size': 32,
                    'memory_usage': {'peak': 1024},
                    'power_usage': {'avg_watts': 65}
                },
                'gpu': {
                    'inference_times': [5.0, 5.1, 4.9, 5.0, 5.2],
                    'batch_size': 32,
                    'memory_usage': {'peak': 2048},
                    'power_usage': {'avg_watts': 150}
                }
            }
        }
        
        # 創建樣本模型結構
        self.model_structure = {
            'total_params': 11689512,
            'trainable_params': 11689512,
            'non_trainable_params': 0,
            'model_type': 'cnn',
            'layers': {
                'conv1': {'type': 'Conv2d', 'params': 1792},
                'conv2': {'type': 'Conv2d', 'params': 36928},
                'fc1': {'type': 'Linear', 'params': 1048576},
                'fc2': {'type': 'Linear', 'params': 10240}
            }
        }
        
        # 保存樣本數據到臨時文件
        with open(os.path.join(self.temp_dir, 'inference_logs.json'), 'w') as f:
            json.dump(self.inference_logs, f)
        
        with open(os.path.join(self.temp_dir, 'model_structure.json'), 'w') as f:
            json.dump(self.model_structure, f)
        
        # 創建分析器實例
        self.analyzer = InferenceAnalyzer(self.temp_dir)
    
    def tearDown(self):
        """
        清理測試環境。
        """
        # 刪除臨時目錄
        shutil.rmtree(self.temp_dir)
        
        # 關閉所有打開的圖表
        plt.close('all')
    
    def test_initialization(self):
        """
        測試 InferenceAnalyzer 的初始化。
        """
        self.assertEqual(self.analyzer.experiment_dir, self.temp_dir)
        self.assertEqual(self.analyzer.device, 'cpu')  # 默認值
        self.assertDictEqual(self.analyzer.inference_logs, {})
        self.assertDictEqual(self.analyzer.model_structure, {})
    
    def test_load_inference_logs(self):
        """
        測試載入推理日誌功能。
        """
        logs = self.analyzer.load_inference_logs()
        
        self.assertIsInstance(logs, dict)
        self.assertIn('timing', logs)
        self.assertIn('memory_usage', logs)
        self.assertIn('outputs', logs)
        self.assertIn('batch_size_tests', logs)
        self.assertIn('input_shape_tests', logs)
        self.assertIn('accuracy', logs)
    
    def test_load_model_structure(self):
        """
        測試載入模型結構功能。
        """
        structure = self.analyzer.load_model_structure()
        
        self.assertIsInstance(structure, dict)
        self.assertIn('total_params', structure)
        self.assertIn('model_type', structure)
        self.assertIn('layers', structure)
    
    def test_load_nonexistent_logs(self):
        """
        測試載入不存在的日誌檔案時的錯誤處理。
        """
        # 創建一個新的臨時目錄（無檔案）
        empty_dir = tempfile.mkdtemp()
        analyzer = InferenceAnalyzer(empty_dir)
        
        # 測試載入不存在的檔案
        with self.assertRaises(Exception):
            analyzer.load_inference_logs()
        
        # 清理
        shutil.rmtree(empty_dir)
    
    def test_analyze_time_performance(self):
        """
        測試時間性能分析。
        """
        # 載入日誌並分析
        self.analyzer.load_inference_logs()
        self.analyzer._analyze_time_performance()
        
        # 檢查結果
        self.assertIn('time_analysis', self.analyzer.results)
        time_analysis = self.analyzer.results['time_analysis']
        
        self.assertIn('inference_time', time_analysis)
        self.assertIn('initialization_time', time_analysis)
        self.assertIn('preprocessing_time', time_analysis)
        self.assertIn('postprocessing_time', time_analysis)
        
        # 檢查推理時間統計
        inference_time = time_analysis['inference_time']
        self.assertAlmostEqual(inference_time['average_ms'], 15.12, places=1)
        self.assertAlmostEqual(inference_time['median_ms'], 15.05, places=1)
        self.assertAlmostEqual(inference_time['min_ms'], 14.5, places=1)
        self.assertAlmostEqual(inference_time['max_ms'], 16.2, places=1)
    
    def test_analyze_throughput(self):
        """
        測試吞吐量分析。
        """
        # 載入日誌並分析
        self.analyzer.load_inference_logs()
        self.analyzer._analyze_throughput()
        
        # 檢查結果
        self.assertIn('throughput_analysis', self.analyzer.results)
        throughput = self.analyzer.results['throughput_analysis']
        
        # 批次大小為32，平均時間約15.12毫秒
        expected_samples_per_sec = (32 * 1000) / 15.12
        self.assertAlmostEqual(throughput['samples_per_second'], expected_samples_per_sec, delta=10)
        self.assertEqual(throughput['batch_size'], 32)
    
    def test_analyze_memory_usage(self):
        """
        測試記憶體使用分析。
        """
        # 載入日誌和模型結構並分析
        self.analyzer.load_inference_logs()
        self.analyzer.load_model_structure()
        self.analyzer._analyze_memory_usage()
        
        # 檢查結果
        self.assertIn('memory_usage', self.analyzer.results)
        memory = self.analyzer.results['memory_usage']
        
        # 檢查 GPU 記憶體
        self.assertIn('gpu', memory)
        self.assertEqual(memory['gpu']['peak_mb'], 1024)
        self.assertEqual(memory['gpu']['allocated_mb'], 768)
        
        # 檢查 CPU 記憶體
        self.assertIn('cpu', memory)
        self.assertEqual(memory['cpu']['peak_mb'], 1536)
        self.assertEqual(memory['cpu']['rss_mb'], 1280)
        
        # 檢查記憶體效率
        self.assertIn('efficiency', memory)
        self.assertIn('bytes_per_parameter', memory['efficiency'])
        
        # 檢查計算的正確性：1024MB轉為bytes除以參數數量
        expected_bytes_per_param = (1024 * 1024 * 1024) / 11689512
        self.assertAlmostEqual(memory['efficiency']['bytes_per_parameter'], expected_bytes_per_param, delta=0.1)
    
    def test_analyze_output_statistics(self):
        """
        測試輸出統計分析。
        """
        # 載入日誌並分析
        self.analyzer.load_inference_logs()
        self.analyzer._analyze_output_statistics()
        
        # 檢查結果
        self.assertIn('output_statistics', self.analyzer.results)
        stats = self.analyzer.results['output_statistics']
        
        # 輸出形狀為 (5, 3)
        self.assertEqual(stats['shape'], [5, 3])
        
        # 檢查統計數據
        self.assertAlmostEqual(stats['mean'], 0.35, places=2)
        self.assertAlmostEqual(stats['min'], 0.01, places=2)
        self.assertAlmostEqual(stats['max'], 0.9, places=2)
        
        # 檢查熵計算
        self.assertIn('entropy', stats)
        self.assertIsInstance(stats['entropy']['mean'], float)
    
    def test_analyze_batch_size_impact(self):
        """
        測試批次大小影響分析。
        """
        # 載入日誌並分析
        self.analyzer.load_inference_logs()
        batch_sizes = [1, 8, 32, 64]
        self.analyzer._analyze_batch_size_impact(batch_sizes)
        
        # 檢查結果
        self.assertIn('batch_size_impact', self.analyzer.results)
        impact = self.analyzer.results['batch_size_impact']
        
        # 檢查性能數據
        self.assertIn('performance_by_batch', impact)
        perf = impact['performance_by_batch']
        
        for batch_size in batch_sizes:
            self.assertIn(batch_size, perf)
            self.assertIn('avg_inference_time_ms', perf[batch_size])
            self.assertIn('throughput_samples_per_sec', perf[batch_size])
        
        # 檢查最佳批次大小
        self.assertIn('optimal_batch_size', impact)
        # 基於我們提供的性能數據，32 應該是最佳批次大小（單位時間下最大吞吐量）
        self.assertEqual(impact['optimal_batch_size'], 32)
        
        # 檢查擴展效率
        self.assertIn('scaling_efficiency', impact)
        scaling = impact['scaling_efficiency']
        for batch_size in [8, 32, 64]:
            self.assertIn(batch_size, scaling)
    
    def test_analyze_input_shape_impact(self):
        """
        測試輸入形狀影響分析。
        """
        # 載入日誌並分析
        self.analyzer.load_inference_logs()
        input_shapes = [[1, 224, 224, 3], [1, 320, 320, 3], [1, 512, 512, 3]]
        self.analyzer._analyze_input_shape_impact(input_shapes)
        
        # 檢查結果
        self.assertIn('input_shape_impact', self.analyzer.results)
        impact = self.analyzer.results['input_shape_impact']
        
        # 檢查性能數據
        self.assertIn('performance_by_shape', impact)
        perf = impact['performance_by_shape']
        
        # 檢查每個形狀的數據
        self.assertIn('1_224_224_3', perf)
        self.assertIn('1_320_320_3', perf)
        self.assertIn('1_512_512_3', perf)
        
        # 檢查相關性和最佳形狀
        self.assertIn('size_time_correlation', impact)
        self.assertIn('optimal_input_shape', impact)
    
    def test_analyze_accuracy(self):
        """
        測試準確性分析。
        """
        # 載入日誌並分析
        self.analyzer.load_inference_logs()
        self.analyzer._analyze_accuracy()
        
        # 檢查結果
        self.assertIn('accuracy_analysis', self.analyzer.results)
        accuracy = self.analyzer.results['accuracy_analysis']
        
        # 檢查總體準確率
        self.assertEqual(accuracy['overall'], 0.85)
        
        # 檢查各類別準確率
        self.assertIn('per_class', accuracy)
        self.assertEqual(accuracy['per_class']['0'], 0.78)
        self.assertEqual(accuracy['per_class']['1'], 0.92)
        
        # 檢查最佳和最差類別
        self.assertIn('best_class', accuracy)
        self.assertEqual(accuracy['best_class']['class_id'], '1')
        self.assertEqual(accuracy['best_class']['accuracy'], 0.92)
        
        self.assertIn('worst_class', accuracy)
        self.assertEqual(accuracy['worst_class']['class_id'], '0')
        self.assertEqual(accuracy['worst_class']['accuracy'], 0.78)
        
        # 檢查 F1 分數、精度、召回率
        self.assertIn('precision', accuracy)
        self.assertIn('recall', accuracy)
        self.assertIn('f1_score', accuracy)
        self.assertIn('macro_precision', accuracy)
        self.assertIn('macro_recall', accuracy)
        self.assertIn('macro_f1', accuracy)
    
    def test_analyze_resource_efficiency(self):
        """
        測試資源效率分析。
        """
        # 載入日誌和模型結構並分析
        self.analyzer.load_inference_logs()
        self.analyzer.load_model_structure()
        
        # 需要先計算吞吐量和記憶體使用
        self.analyzer._analyze_throughput()
        self.analyzer._analyze_memory_usage()
        
        # 分析資源效率
        self.analyzer._analyze_resource_efficiency()
        
        # 檢查結果
        self.assertIn('resource_efficiency', self.analyzer.results)
        efficiency = self.analyzer.results['resource_efficiency']
        
        # 檢查記憶體效率指標
        self.assertIn('throughput_per_mb', efficiency)
        self.assertIn('memory_efficiency_class', efficiency)
        
        # 檢查參數效率指標
        self.assertIn('throughput_per_million_params', efficiency)
        self.assertIn('parameter_efficiency_class', efficiency)
    
    def test_analyze_hardware_comparison(self):
        """
        測試硬體比較分析。
        """
        # 載入日誌並分析
        self.analyzer.load_inference_logs()
        self.analyzer._analyze_hardware_comparison()
        
        # 檢查結果
        self.assertIn('hardware_comparison', self.analyzer.results)
        comparison = self.analyzer.results['hardware_comparison']
        
        # 檢查硬體性能數據
        self.assertIn('performance_by_hardware', comparison)
        perf = comparison['performance_by_hardware']
        
        self.assertIn('cpu', perf)
        self.assertIn('gpu', perf)
        
        # 檢查最佳硬體
        self.assertIn('best_throughput_hardware', comparison)
        self.assertEqual(comparison['best_throughput_hardware'], 'gpu')
        
        # 檢查能源效率最高的硬體
        self.assertIn('most_energy_efficient_hardware', comparison)
    
    def test_full_analyze(self):
        """
        測試完整的分析流程。
        """
        # 執行完整分析
        batch_sizes = [1, 8, 32, 64]
        input_shapes = [[1, 224, 224, 3], [1, 320, 320, 3], [1, 512, 512, 3]]
        
        results = self.analyzer.analyze(
            batch_sizes=batch_sizes,
            input_shapes=input_shapes,
            save_results=True
        )
        
        # 檢查結果是否包含所有分析部分
        self.assertIn('time_analysis', results)
        self.assertIn('throughput_analysis', results)
        self.assertIn('memory_usage', results)
        self.assertIn('output_statistics', results)
        self.assertIn('batch_size_impact', results)
        self.assertIn('input_shape_impact', results)
        self.assertIn('accuracy_analysis', results)
        self.assertIn('resource_efficiency', results)
        self.assertIn('hardware_comparison', results)
        
        # 檢查結果文件是否已保存
        result_path = os.path.join(self.temp_dir, 'inference_analysis.json')
        self.assertTrue(os.path.exists(result_path))
    
    def test_getter_methods(self):
        """
        測試獲取分析結果的方法。
        """
        # 先執行完整分析
        self.analyzer.analyze(save_results=False)
        
        # 測試各個獲取方法
        time_perf = self.analyzer.get_time_performance()
        self.assertIsInstance(time_perf, dict)
        self.assertIn('inference_time', time_perf)
        
        memory_usage = self.analyzer.get_memory_usage()
        self.assertIsInstance(memory_usage, dict)
        self.assertIn('gpu', memory_usage)
        
        accuracy = self.analyzer.get_accuracy_analysis()
        self.assertIsInstance(accuracy, dict)
        self.assertIn('overall', accuracy)
        
        efficiency = self.analyzer.get_resource_efficiency()
        self.assertIsInstance(efficiency, dict)
        self.assertIn('throughput_per_mb', efficiency)


if __name__ == '__main__':
    unittest.main() 