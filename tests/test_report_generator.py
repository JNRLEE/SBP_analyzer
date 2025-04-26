"""
此模組包含對報告生成器 (ReportGenerator) 功能的測試。

Tests:
    - ReportGenerator初始化
    - 模板載入和使用
    - HTML、Markdown 和 PDF 報告生成
    - 報告模板自定義
"""

import os
import sys
import tempfile
import pytest
import shutil
from datetime import datetime
from pathlib import Path

# 添加項目根目錄到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reporter.report_generator import ReportGenerator

# 測試常量
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'test_output', 'reports')
TEST_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'test_output', 'templates')

# 測試用的分析結果類
class MockAnalysisResults:
    def __init__(self, data):
        self.data = data
        
    def to_dict(self):
        return self.data

# 測試固件 - 創建輸出目錄
@pytest.fixture(scope="module")
def setup_test_dirs():
    """設置測試目錄"""
    # 確保測試輸出目錄存在
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEST_TEMPLATES_DIR, exist_ok=True)
    
    # 測試完成後清理
    yield
    # 清理臨時文件 (可選，取消註釋以啟用清理)
    # shutil.rmtree(TEST_OUTPUT_DIR)
    # shutil.rmtree(TEST_TEMPLATES_DIR)

# 測試固件 - 創建自定義模板
@pytest.fixture(scope="module")
def setup_custom_templates(setup_test_dirs):
    """創建自定義報告模板"""
    # 創建自定義HTML模板
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{{experiment_name}} - 自定義分析報告</title>
        <style>
            body { font-family: Helvetica, sans-serif; background-color: #eef; }
            .header { background-color: #44a; color: white; padding: 20px; }
            .section { margin: 20px; padding: 15px; background-color: white; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{{experiment_name}} - 深度學習模型分析</h1>
            <p>分析時間: {{analysis_time}}</p>
        </div>
        {{model_structure_section}}
        {{training_dynamics_section}}
        {{intermediate_data_section}}
    </body>
    </html>
    """
    
    # 創建自定義Markdown模板
    md_template = """
    # {{experiment_name}} - 模型分析報告
    
    > 生成時間: {{analysis_time}}
    
    ## 目錄
    
    1. [模型結構](#model-structure)
    2. [訓練動態](#training-dynamics)
    3. [中間層數據](#intermediate-data)
    
    {{model_structure_section}}
    {{training_dynamics_section}}
    {{intermediate_data_section}}
    
    ---
    報告生成由 SBP Analyzer 提供
    """
    
    # 寫入模板文件
    with open(os.path.join(TEST_TEMPLATES_DIR, "html_template.html"), "w", encoding="utf-8") as f:
        f.write(html_template)
        
    with open(os.path.join(TEST_TEMPLATES_DIR, "markdown_template.md"), "w", encoding="utf-8") as f:
        f.write(md_template)
    
    return TEST_TEMPLATES_DIR

# 測試固件 - 創建模擬分析結果
@pytest.fixture
def mock_analysis_results():
    """創建模擬分析結果"""
    return MockAnalysisResults({
        'model_structure': {
            'model_name': 'TestModel',
            'total_params': 1234567,
            'trainable_params': 1200000,
            'non_trainable_params': 34567,
            'layer_count': 42,
            'layer_types': {
                'Conv2D': 24,
                'BatchNorm2D': 24,
                'ReLU': 24,
                'Linear': 3,
                'Dropout': 2
            }
        },
        'training_dynamics': {
            'convergence_analysis': {
                'loss': {
                    'convergence_epoch': 7,
                    'convergence_rate': 0.1,
                    'final_value': 0.48
                },
                'accuracy': {
                    'convergence_epoch': 6,
                    'convergence_rate': 0.05,
                    'final_value': 0.9
                }
            },
            'stability_analysis': {
                'loss': {
                    'variance': 0.25,
                    'max_fluctuation': 0.2
                },
                'accuracy': {
                    'variance': 0.01,
                    'max_fluctuation': 0.05
                }
            }
        },
        'intermediate_data': {
            'layer_results': {
                'layer1': {
                    'epochs_analyzed': [0, 1],
                    'statistics': {
                        'mean': 0.01,
                        'std': 0.5,
                        'min': -1.5,
                        'max': 1.5,
                        'median': 0.02
                    },
                    'sparsity': {
                        'l1_sparsity': 0.2,
                        'zero_ratio': 0.1
                    }
                },
                'layer2': {
                    'epochs_analyzed': [0],
                    'statistics': {
                        'mean': 0.001,
                        'std': 0.3,
                        'min': -0.8,
                        'max': 0.8,
                        'median': 0.001
                    },
                    'sparsity': {
                        'l1_sparsity': 0.5,
                        'zero_ratio': 0.5
                    }
                }
            }
        }
    })

# 測試: ReportGenerator 初始化
def test_report_generator_init(setup_test_dirs):
    """測試報告生成器初始化"""
    # 使用默認模板
    report_gen = ReportGenerator(experiment_name="test_experiment", output_dir=TEST_OUTPUT_DIR)
    assert report_gen.experiment_name == "test_experiment"
    assert report_gen.output_dir == TEST_OUTPUT_DIR
    assert "html" in report_gen.templates
    assert "markdown" in report_gen.templates

# 測試: HTML 報告生成
def test_generate_html_report(setup_test_dirs, mock_analysis_results):
    """測試HTML報告生成"""
    report_gen = ReportGenerator(experiment_name="test_html", output_dir=TEST_OUTPUT_DIR)
    report_path = report_gen.generate(mock_analysis_results, format="html")
    
    # 檢查報告是否生成
    assert os.path.exists(report_path)
    assert report_path.endswith(".html")
    
    # 檢查報告內容
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "test_html" in content
        assert "TestModel" in content
        assert "1,234,567" in content  # 總參數量應該被格式化

# 測試: Markdown 報告生成
def test_generate_markdown_report(setup_test_dirs, mock_analysis_results):
    """測試Markdown報告生成"""
    report_gen = ReportGenerator(experiment_name="test_md", output_dir=TEST_OUTPUT_DIR)
    report_path = report_gen.generate(mock_analysis_results, format="markdown")
    
    # 檢查報告是否生成
    assert os.path.exists(report_path)
    assert report_path.endswith(".md")
    
    # 檢查報告內容
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "test_md" in content
        assert "模型結構分析" in content
        assert "TestModel" in content

# 測試: 自定義模板
def test_custom_templates(setup_custom_templates, mock_analysis_results):
    """測試使用自定義模板"""
    report_gen = ReportGenerator(
        experiment_name="custom_template_test", 
        output_dir=TEST_OUTPUT_DIR,
        template_dir=TEST_TEMPLATES_DIR
    )
    
    # 檢查模板是否成功載入
    assert "自定義分析報告" in report_gen.templates["html"]
    assert "目錄" in report_gen.templates["markdown"]
    
    # 生成報告並檢查內容
    html_path = report_gen.generate(mock_analysis_results, format="html")
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "自定義分析報告" in content
        assert "custom_template_test" in content
    
    md_path = report_gen.generate(mock_analysis_results, format="markdown")
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "目錄" in content
        assert "報告生成由 SBP Analyzer 提供" in content

# 測試: 動態設置模板
def test_set_template(setup_test_dirs, mock_analysis_results):
    """測試動態設置模板"""
    report_gen = ReportGenerator(experiment_name="dynamic_template", output_dir=TEST_OUTPUT_DIR)
    
    # 設置自定義模板
    custom_html = """
    <!DOCTYPE html>
    <html>
    <head><title>簡潔模板 - {{experiment_name}}</title></head>
    <body>
        <h1>{{experiment_name}}</h1>
        <p>{{analysis_time}}</p>
        {{model_structure_section}}
        {{training_dynamics_section}}
        {{intermediate_data_section}}
    </body>
    </html>
    """
    
    report_gen.set_template("html", custom_html)
    
    # 檢查模板是否成功設置
    assert "簡潔模板" in report_gen.templates["html"]
    
    # 生成報告並檢查內容
    html_path = report_gen.generate(mock_analysis_results, format="html")
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "簡潔模板" in content
        assert "dynamic_template" in content

# 測試: PDF 報告生成 (如果安裝了相關庫)
def test_generate_pdf_report(setup_test_dirs, mock_analysis_results):
    """測試PDF報告生成 (如果支持)"""
    report_gen = ReportGenerator(experiment_name="test_pdf", output_dir=TEST_OUTPUT_DIR)
    report_path = report_gen.generate(mock_analysis_results, format="pdf")
    
    # 檢查報告是否生成 (即使回退到HTML也應該有輸出)
    assert os.path.exists(report_path)
    
    # 如果成功生成了PDF，檢查文件類型
    if report_path.endswith(".pdf"):
        # 檢查文件大小 > 0
        assert os.path.getsize(report_path) > 0
    else:
        # 如果回退到HTML，檢查其內容
        assert report_path.endswith(".html")
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "test_pdf" in content

# 測試: 報告生成時處理損毀的分析結果
def test_handle_corrupted_data(setup_test_dirs):
    """測試處理損毀的分析結果數據"""
    # 創建包含部分數據的分析結果
    partial_results = MockAnalysisResults({
        'model_structure': {
            'model_name': 'PartialModel',
            # 缺少其他字段
        },
        # 缺少訓練動態和中間層數據
    })
    
    report_gen = ReportGenerator(experiment_name="partial_data", output_dir=TEST_OUTPUT_DIR)
    
    # 生成報告應該可以處理缺失的數據
    report_path = report_gen.generate(partial_results, format="html")
    assert os.path.exists(report_path)
    
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "PartialModel" in content  # 應該包含可用的數據 