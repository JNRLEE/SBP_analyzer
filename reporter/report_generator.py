"""
報告生成器模組。

此模組提供用於生成分析報告的功能，支持HTML和Markdown格式。

Classes:
    ReportGenerator: 生成分析報告的類別。
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Union

class ReportGenerator:
    """
    報告生成器，用於生成分析報告。
    
    此類別可以基於分析結果生成HTML或Markdown格式的報告。
    
    Attributes:
        experiment_name (str): 實驗名稱。
        output_dir (str): 輸出目錄路徑。
        logger (logging.Logger): 日誌記錄器。
    """
    
    def __init__(self, experiment_name: str, output_dir: str):
        """
        初始化報告生成器。
        
        Args:
            experiment_name (str): 實驗名稱。
            output_dir (str): 輸出目錄路徑。
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 確保輸出目錄存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def generate(self, results: Any, format: str = 'html') -> str:
        """
        生成分析報告。
        
        Args:
            results (Any): 分析結果對象。
            format (str, optional): 報告格式，可選 'html' 或 'markdown'。默認為'html'。
        
        Returns:
            str: 報告文件路徑。
        """
        if format.lower() == 'html':
            return self._generate_html_report(results)
        elif format.lower() == 'markdown':
            return self._generate_markdown_report(results)
        else:
            self.logger.error(f"不支持的報告格式: {format}")
            raise ValueError(f"不支持的報告格式: {format}")
    
    def _generate_html_report(self, results: Any) -> str:
        """
        生成HTML格式的報告。
        
        Args:
            results (Any): 分析結果對象。
        
        Returns:
            str: 報告文件路徑。
        """
        # 將結果轉換為字典
        results_dict = results.to_dict()
        
        # 生成報告時間
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 報告文件名
        report_filename = f"{self.experiment_name}_analysis_report_{timestamp}.html"
        report_path = os.path.join(self.output_dir, report_filename)
        
        # 生成HTML內容
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>分析報告 - {self.experiment_name}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #444;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .section {{
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                .plot-container {{
                    margin: 20px 0;
                    text-align: center;
                }}
                pre {{
                    background-color: #f5f5f5;
                    padding: 15px;
                    overflow: auto;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>深度學習模型分析報告</h1>
                <div class="section">
                    <h2>實驗信息</h2>
                    <table>
                        <tr>
                            <th>實驗名稱</th>
                            <td>{self.experiment_name}</td>
                        </tr>
                        <tr>
                            <th>分析時間</th>
                            <td>{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</td>
                        </tr>
                    </table>
                </div>
        """
        
        # 添加模型結構分析結果
        if results_dict.get('model_structure'):
            html_content += self._generate_model_structure_section(results_dict['model_structure'])
        
        # 添加訓練動態分析結果
        if results_dict.get('training_dynamics'):
            html_content += self._generate_training_dynamics_section(results_dict['training_dynamics'])
        
        # 添加中間層數據分析結果
        if results_dict.get('intermediate_data'):
            html_content += self._generate_intermediate_data_section(results_dict['intermediate_data'])
        
        # 結尾
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # 寫入文件
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path
    
    def _generate_model_structure_section(self, model_structure_results: Dict) -> str:
        """
        生成模型結構分析部分的HTML。
        
        Args:
            model_structure_results (Dict): 模型結構分析結果。
        
        Returns:
            str: HTML內容。
        """
        html = f"""
        <div class="section">
            <h2>模型結構分析</h2>
            <table>
                <tr>
                    <th>模型名稱</th>
                    <td>{model_structure_results.get('model_name', 'Unknown')}</td>
                </tr>
                <tr>
                    <th>總參數量</th>
                    <td>{model_structure_results.get('total_params', 0):,}</td>
                </tr>
                <tr>
                    <th>可訓練參數量</th>
                    <td>{model_structure_results.get('trainable_params', 0):,}</td>
                </tr>
                <tr>
                    <th>非可訓練參數量</th>
                    <td>{model_structure_results.get('non_trainable_params', 0):,}</td>
                </tr>
                <tr>
                    <th>層數量</th>
                    <td>{model_structure_results.get('layer_count', 0)}</td>
                </tr>
            </table>
        """
        
        # 如果有層類型分佈
        if 'layer_types' in model_structure_results and model_structure_results['layer_types']:
            html += """
            <h3>層類型分佈</h3>
            <table>
                <tr>
                    <th>層類型</th>
                    <th>數量</th>
                </tr>
            """
            
            for layer_type, count in model_structure_results['layer_types'].items():
                html += f"""
                <tr>
                    <td>{layer_type}</td>
                    <td>{count}</td>
                </tr>
                """
            
            html += """
            </table>
            """
        
        html += """
        </div>
        """
        
        return html
    
    def _generate_training_dynamics_section(self, training_dynamics_results: Dict) -> str:
        """
        生成訓練動態分析部分的HTML。
        
        Args:
            training_dynamics_results (Dict): 訓練動態分析結果。
        
        Returns:
            str: HTML內容。
        """
        html = """
        <div class="section">
            <h2>訓練動態分析</h2>
        """
        
        # 如果有收斂分析
        if 'convergence_analysis' in training_dynamics_results and training_dynamics_results['convergence_analysis']:
            html += """
            <h3>收斂性分析</h3>
            <table>
                <tr>
                    <th>指標</th>
                    <th>收斂輪次</th>
                    <th>收斂速度</th>
                    <th>最終值</th>
                </tr>
            """
            
            for metric, data in training_dynamics_results['convergence_analysis'].items():
                html += f"""
                <tr>
                    <td>{metric}</td>
                    <td>{data.get('convergence_epoch', 'N/A')}</td>
                    <td>{data.get('convergence_rate', 'N/A')}</td>
                    <td>{data.get('final_value', 'N/A')}</td>
                </tr>
                """
            
            html += """
            </table>
            """
        
        # 如果有穩定性分析
        if 'stability_analysis' in training_dynamics_results and training_dynamics_results['stability_analysis']:
            html += """
            <h3>穩定性分析</h3>
            <table>
                <tr>
                    <th>指標</th>
                    <th>方差</th>
                    <th>最大波動</th>
                </tr>
            """
            
            for metric, data in training_dynamics_results['stability_analysis'].items():
                html += f"""
                <tr>
                    <td>{metric}</td>
                    <td>{data.get('variance', 'N/A')}</td>
                    <td>{data.get('max_fluctuation', 'N/A')}</td>
                </tr>
                """
            
            html += """
            </table>
            """
        
        html += """
        </div>
        """
        
        return html
    
    def _generate_intermediate_data_section(self, intermediate_data_results: Dict) -> str:
        """
        生成中間層數據分析部分的HTML。
        
        Args:
            intermediate_data_results (Dict): 中間層數據分析結果。
        
        Returns:
            str: HTML內容。
        """
        html = """
        <div class="section">
            <h2>中間層數據分析</h2>
        """
        
        layer_results = intermediate_data_results.get('layer_results', {})
        if layer_results:
            for layer_name, layer_data in layer_results.items():
                html += f"""
                <h3>層: {layer_name}</h3>
                <p>分析輪次: {', '.join(map(str, layer_data.get('epochs_analyzed', [])))}</p>
                """
                
                # 如果有統計信息
                if 'statistics' in layer_data and layer_data['statistics']:
                    html += """
                    <h4>基本統計量</h4>
                    <table>
                        <tr>
                            <th>輪次</th>
                            <th>均值</th>
                            <th>標準差</th>
                            <th>最小值</th>
                            <th>最大值</th>
                            <th>中位數</th>
                        </tr>
                    """
                    
                    for epoch, stats in layer_data['statistics'].items():
                        html += f"""
                        <tr>
                            <td>{epoch}</td>
                            <td>{stats.get('mean', 'N/A'):.6f}</td>
                            <td>{stats.get('std', 'N/A'):.6f}</td>
                            <td>{stats.get('min', 'N/A'):.6f}</td>
                            <td>{stats.get('max', 'N/A'):.6f}</td>
                            <td>{stats.get('median', 'N/A'):.6f}</td>
                        </tr>
                        """
                    
                    html += """
                    </table>
                    """
                
                # 如果有稀疏度信息
                if 'sparsity' in layer_data and layer_data['sparsity']:
                    html += """
                    <h4>稀疏度分析</h4>
                    <table>
                        <tr>
                            <th>輪次</th>
                            <th>L1稀疏度</th>
                            <th>零元素比例</th>
                        </tr>
                    """
                    
                    for epoch, sparsity in layer_data['sparsity'].items():
                        html += f"""
                        <tr>
                            <td>{epoch}</td>
                            <td>{sparsity.get('l1_sparsity', 'N/A'):.6f}</td>
                            <td>{sparsity.get('zero_ratio', 'N/A'):.2%}</td>
                        </tr>
                        """
        else:
            html += """
            <p>未找到中間層數據分析結果。</p>
            """
        
        html += """
        </div>
        """
        
        return html
    
    def _generate_markdown_report(self, results: Any) -> str:
        """
        生成Markdown格式的報告。
        
        Args:
            results (Any): 分析結果對象。
        
        Returns:
            str: 報告文件路徑。
        """
        # 將結果轉換為字典
        results_dict = results.to_dict()
        
        # 生成報告時間
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 報告文件名
        report_filename = f"{self.experiment_name}_analysis_report_{timestamp}.md"
        report_path = os.path.join(self.output_dir, report_filename)
        
        # 生成Markdown內容
        md_content = f"""
# 深度學習模型分析報告

## 實驗信息

- **實驗名稱**: {self.experiment_name}
- **分析時間**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

"""
        
        # 添加模型結構分析結果
        if results_dict.get('model_structure'):
            md_content += self._generate_model_structure_section_md(results_dict['model_structure'])
        
        # 添加訓練動態分析結果
        if results_dict.get('training_dynamics'):
            md_content += self._generate_training_dynamics_section_md(results_dict['training_dynamics'])
        
        # 添加中間層數據分析結果
        if results_dict.get('intermediate_data'):
            md_content += self._generate_intermediate_data_section_md(results_dict['intermediate_data'])
        
        # 寫入文件
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return report_path
    
    def _generate_model_structure_section_md(self, model_structure_results: Dict) -> str:
        """
        生成模型結構分析部分的Markdown。
        
        Args:
            model_structure_results (Dict): 模型結構分析結果。
        
        Returns:
            str: Markdown內容。
        """
        md = f"""
## 模型結構分析

- **模型名稱**: {model_structure_results.get('model_name', 'Unknown')}
- **總參數量**: {model_structure_results.get('total_params', 0):,}
- **可訓練參數量**: {model_structure_results.get('trainable_params', 0):,}
- **非可訓練參數量**: {model_structure_results.get('non_trainable_params', 0):,}
- **層數量**: {model_structure_results.get('layer_count', 0)}

"""
        
        # 如果有層類型分佈
        if 'layer_types' in model_structure_results and model_structure_results['layer_types']:
            md += """
### 層類型分佈

| 層類型 | 數量 |
| ------ | ---- |
"""
            
            for layer_type, count in model_structure_results['layer_types'].items():
                md += f"| {layer_type} | {count} |\n"
        
        return md
    
    def _generate_training_dynamics_section_md(self, training_dynamics_results: Dict) -> str:
        """
        生成訓練動態分析部分的Markdown。
        
        Args:
            training_dynamics_results (Dict): 訓練動態分析結果。
        
        Returns:
            str: Markdown內容。
        """
        md = """
## 訓練動態分析

"""
        
        # 如果有收斂分析
        if 'convergence_analysis' in training_dynamics_results and training_dynamics_results['convergence_analysis']:
            md += """
### 收斂性分析

| 指標 | 收斂輪次 | 收斂速度 | 最終值 |
| ---- | -------- | -------- | ------ |
"""
            
            for metric, data in training_dynamics_results['convergence_analysis'].items():
                md += f"| {metric} | {data.get('convergence_epoch', 'N/A')} | {data.get('convergence_rate', 'N/A')} | {data.get('final_value', 'N/A')} |\n"
        
        # 如果有穩定性分析
        if 'stability_analysis' in training_dynamics_results and training_dynamics_results['stability_analysis']:
            md += """
### 穩定性分析

| 指標 | 方差 | 最大波動 |
| ---- | ---- | -------- |
"""
            
            for metric, data in training_dynamics_results['stability_analysis'].items():
                md += f"| {metric} | {data.get('variance', 'N/A')} | {data.get('max_fluctuation', 'N/A')} |\n"
        
        return md
    
    def _generate_intermediate_data_section_md(self, intermediate_data_results: Dict) -> str:
        """
        生成中間層數據分析部分的Markdown。
        
        Args:
            intermediate_data_results (Dict): 中間層數據分析結果。
        
        Returns:
            str: Markdown內容。
        """
        md = """
## 中間層數據分析

"""
        
        layer_results = intermediate_data_results.get('layer_results', {})
        if layer_results:
            for layer_name, layer_data in layer_results.items():
                md += f"""
### 層: {layer_name}

分析輪次: {', '.join(map(str, layer_data.get('epochs_analyzed', [])))}

"""
                
                # 如果有統計信息
                if 'statistics' in layer_data and layer_data['statistics']:
                    md += """
#### 基本統計量

| 輪次 | 均值 | 標準差 | 最小值 | 最大值 | 中位數 |
| ---- | ---- | ------ | ------ | ------ | ------ |
"""
                    
                    for epoch, stats in layer_data['statistics'].items():
                        md += f"| {epoch} | {stats.get('mean', 'N/A'):.6f} | {stats.get('std', 'N/A'):.6f} | {stats.get('min', 'N/A'):.6f} | {stats.get('max', 'N/A'):.6f} | {stats.get('median', 'N/A'):.6f} |\n"
                
                # 如果有稀疏度信息
                if 'sparsity' in layer_data and layer_data['sparsity']:
                    md += """
#### 稀疏度分析

| 輪次 | L1稀疏度 | 零元素比例 |
| ---- | -------- | ---------- |
"""
                    
                    for epoch, sparsity in layer_data['sparsity'].items():
                        zero_ratio = sparsity.get('zero_ratio', 0) * 100
                        md += f"| {epoch} | {sparsity.get('l1_sparsity', 'N/A'):.6f} | {zero_ratio:.2f}% |\n"
        else:
            md += """
未找到中間層數據分析結果。
"""
        
        return md 