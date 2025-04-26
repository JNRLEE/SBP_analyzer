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
import tempfile
from typing import Dict, List, Any, Optional, Union

class ReportGenerator:
    """
    報告生成器，用於生成分析報告。
    
    此類別可以基於分析結果生成HTML、Markdown或PDF格式的報告。
    
    Attributes:
        experiment_name (str): 實驗名稱。
        output_dir (str): 輸出目錄路徑。
        logger (logging.Logger): 日誌記錄器。
        templates (Dict): 報告模板字典。
    """
    
    def __init__(self, experiment_name: str, output_dir: str, template_dir: Optional[str] = None):
        """
        初始化報告生成器。
        
        Args:
            experiment_name (str): 實驗名稱。
            output_dir (str): 輸出目錄路徑。
            template_dir (Optional[str], optional): 模板目錄路徑。若為None，則使用默認模板。
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        self.templates = {}
        
        # 載入模板（如果提供了模板目錄）
        if template_dir and os.path.exists(template_dir):
            self._load_templates(template_dir)
        else:
            # 使用默認模板
            self._init_default_templates()
        
        # 確保輸出目錄存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def _load_templates(self, template_dir: str) -> None:
        """
        從指定目錄載入模板。
        
        Args:
            template_dir (str): 模板目錄路徑。
        """
        self.logger.info(f"從 {template_dir} 載入報告模板")
        
        # 載入HTML模板
        html_template_path = os.path.join(template_dir, "html_template.html")
        if os.path.exists(html_template_path):
            with open(html_template_path, 'r', encoding='utf-8') as f:
                self.templates['html'] = f.read()
        
        # 載入Markdown模板
        md_template_path = os.path.join(template_dir, "markdown_template.md")
        if os.path.exists(md_template_path):
            with open(md_template_path, 'r', encoding='utf-8') as f:
                self.templates['markdown'] = f.read()
    
    def _init_default_templates(self) -> None:
        """
        初始化默認模板。
        """
        # HTML默認模板
        self.templates['html'] = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>分析報告 - {{experiment_name}}</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }
                h1, h2, h3 {
                    color: #444;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .section {
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }
                th, td {
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f2f2f2;
                }
                .plot-container {
                    margin: 20px 0;
                    text-align: center;
                }
                pre {
                    background-color: #f5f5f5;
                    padding: 15px;
                    overflow: auto;
                    border-radius: 5px;
                }
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
                            <td>{{experiment_name}}</td>
                        </tr>
                        <tr>
                            <th>分析時間</th>
                            <td>{{analysis_time}}</td>
                        </tr>
                    </table>
                </div>
                {{model_structure_section}}
                {{training_dynamics_section}}
                {{intermediate_data_section}}
            </div>
        </body>
        </html>
        """
        
        # Markdown默認模板
        self.templates['markdown'] = """
# 深度學習模型分析報告

## 實驗信息

- **實驗名稱**: {{experiment_name}}
- **分析時間**: {{analysis_time}}

{{model_structure_section}}
{{training_dynamics_section}}
{{intermediate_data_section}}
        """
    
    def generate(self, results: Any, format: str = 'html') -> str:
        """
        生成分析報告。
        
        Args:
            results (Any): 分析結果對象。
            format (str, optional): 報告格式，可選 'html'、'markdown' 或 'pdf'。默認為'html'。
        
        Returns:
            str: 報告文件路徑。
        """
        if format.lower() == 'html':
            return self._generate_html_report(results)
        elif format.lower() == 'markdown':
            return self._generate_markdown_report(results)
        elif format.lower() == 'pdf':
            return self._generate_pdf_report(results)
        else:
            self.logger.error(f"不支持的報告格式: {format}")
            raise ValueError(f"不支持的報告格式: {format}")
    
    def generate_report(self, analysis_results: Any, experiment_name: Optional[str] = None, 
                      output_dir: Optional[str] = None, format: str = 'html') -> str:
        """
        生成分析報告（兼容性方法，用於測試）。
        
        Args:
            analysis_results (Any): 分析結果對象。
            experiment_name (Optional[str], optional): 實驗名稱。如果提供，則會覆蓋初始化時設置的實驗名稱。
            output_dir (Optional[str], optional): 輸出目錄。如果提供，則會覆蓋初始化時設置的輸出目錄。
            format (str, optional): 報告格式，可選 'html'、'markdown' 或 'pdf'。默認為'html'。
        
        Returns:
            str: 報告文件路徑。
        """
        # 臨時保存原始值
        original_experiment_name = self.experiment_name
        original_output_dir = self.output_dir
        
        # 如果提供了新值，則臨時覆蓋
        if experiment_name:
            self.experiment_name = experiment_name
        if output_dir:
            self.output_dir = output_dir
            # 確保輸出目錄存在
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
        try:
            # 調用原始的generate方法
            return self.generate(analysis_results, format)
        finally:
            # 恢復原始值
            self.experiment_name = original_experiment_name
            self.output_dir = original_output_dir
    
    def _generate_html_report(self, results: Any) -> str:
        """
        生成HTML格式的報告。
        
        Args:
            results (Any): 分析結果對象。
        
        Returns:
            str: 報告文件路徑。
        """
        # 將結果轉換為字典
        if hasattr(results, 'to_dict'):
            results_dict = results.to_dict()
        elif isinstance(results, dict):
            results_dict = results
        else:
            # 對於 AnalysisResults 對象，特殊處理
            results_dict = {
                'model_structure': results.get_model_structure_results() if hasattr(results, 'get_model_structure_results') else {},
                'training_dynamics': results.get_training_dynamics_results() if hasattr(results, 'get_training_dynamics_results') else {},
                'intermediate_data': results.get_intermediate_data_results() if hasattr(results, 'get_intermediate_data_results') else {},
                'plots': results.get_all_plots() if hasattr(results, 'get_all_plots') else {}
            }
        
        # 生成報告時間
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 報告文件名
        report_filename = f"{self.experiment_name}_analysis_report_{timestamp}.html"
        report_path = os.path.join(self.output_dir, report_filename)
        
        # 使用模板生成HTML內容
        html_content = self.templates['html']
        
        # 替換模板變量
        html_content = html_content.replace("{{experiment_name}}", self.experiment_name)
        html_content = html_content.replace("{{analysis_time}}", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # 生成各部分內容
        model_structure_section = ""
        if 'model_structure' in results_dict and results_dict['model_structure']:
            model_structure_section = self._generate_model_structure_section(results_dict['model_structure'])
        html_content = html_content.replace("{{model_structure_section}}", model_structure_section)
        
        training_dynamics_section = ""
        if 'training_dynamics' in results_dict and results_dict['training_dynamics']:
            training_dynamics_section = self._generate_training_dynamics_section(results_dict['training_dynamics'])
        html_content = html_content.replace("{{training_dynamics_section}}", training_dynamics_section)
        
        intermediate_data_section = ""
        if 'intermediate_data' in results_dict and results_dict['intermediate_data']:
            intermediate_data_section = self._generate_intermediate_data_section(results_dict['intermediate_data'])
        html_content = html_content.replace("{{intermediate_data_section}}", intermediate_data_section)
        
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
        # 確保結果是字典
        results_dict = model_structure_results
        
        # 生成HTML內容
        html_content = f"""
        <div class="section">
            <h2>模型結構分析</h2>
            <table>
                <tr>
                    <th>模型名稱</th>
                    <td>{results_dict.get('model_name', 'Unknown')}</td>
                </tr>
                <tr>
                    <th>總參數量</th>
                    <td>{results_dict.get('total_params', 0):,}</td>
                </tr>
                <tr>
                    <th>可訓練參數量</th>
                    <td>{results_dict.get('trainable_params', 0):,}</td>
                </tr>
                <tr>
                    <th>非可訓練參數量</th>
                    <td>{results_dict.get('non_trainable_params', 0):,}</td>
                </tr>
                <tr>
                    <th>層數量</th>
                    <td>{results_dict.get('layer_count', 0)}</td>
                </tr>
            </table>
        """
        
        # 如果有層類型分佈
        if 'layer_types' in results_dict and results_dict['layer_types']:
            html_content += """
            <h3>層類型分佈</h3>
            <table>
                <tr>
                    <th>層類型</th>
                    <th>數量</th>
                </tr>
            """
            
            for layer_type, count in results_dict['layer_types'].items():
                html_content += f"""
                <tr>
                    <td>{layer_type}</td>
                    <td>{count}</td>
                </tr>
                """
            
            html_content += """
            </table>
            """
        
        html_content += """
        </div>
        """
        
        return html_content
    
    def _generate_training_dynamics_section(self, training_dynamics_results: Dict) -> str:
        """
        生成訓練動態分析部分的HTML。
        
        Args:
            training_dynamics_results (Dict): 訓練動態分析結果。
        
        Returns:
            str: HTML內容。
        """
        # 確保結果是字典
        results_dict = training_dynamics_results
        
        # 生成HTML內容
        html_content = f"""
        <div class="section">
            <h2>訓練動態分析</h2>
        """
        
        # 如果有收斂分析
        if 'convergence_analysis' in results_dict and results_dict['convergence_analysis']:
            html_content += """
            <h3>收斂性分析</h3>
            <table>
                <tr>
                    <th>指標</th>
                    <th>收斂輪次</th>
                    <th>收斂速度</th>
                    <th>最終值</th>
                </tr>
            """
            
            for metric, data in results_dict['convergence_analysis'].items():
                if isinstance(data, dict):
                    html_content += f"""
                    <tr>
                        <td>{metric}</td>
                        <td>{data.get('convergence_epoch', 'N/A')}</td>
                        <td>{data.get('convergence_rate', 'N/A')}</td>
                        <td>{data.get('final_value', 'N/A')}</td>
                    </tr>
                    """
            
            html_content += """
            </table>
            """
        
        # 如果有穩定性分析
        if 'stability_analysis' in results_dict and results_dict['stability_analysis']:
            html_content += """
            <h3>穩定性分析</h3>
            <table>
                <tr>
                    <th>指標</th>
                    <th>方差</th>
                    <th>最大波動</th>
                </tr>
            """
            
            for metric, data in results_dict['stability_analysis'].items():
                if isinstance(data, dict):
                    html_content += f"""
                    <tr>
                        <td>{metric}</td>
                        <td>{data.get('variance', 'N/A')}</td>
                        <td>{data.get('max_fluctuation', 'N/A')}</td>
                    </tr>
                    """
            
            html_content += """
            </table>
            """
        
        # 如果有學習率分析
        if 'learning_rate_analysis' in results_dict and results_dict['learning_rate_analysis']:
            html_content += """
            <h3>學習率分析</h3>
            <table>
                <tr>
                    <th>初始學習率</th>
                    <th>最終學習率</th>
                    <th>衰減模式</th>
                </tr>
                <tr>
                    <td>{}</td>
                    <td>{}</td>
                    <td>{}</td>
                </tr>
            </table>
            """.format(
                results_dict['learning_rate_analysis'].get('initial_lr', 'N/A'),
                results_dict['learning_rate_analysis'].get('final_lr', 'N/A'),
                results_dict['learning_rate_analysis'].get('decay_pattern', 'N/A')
            )
        
        html_content += """
        </div>
        """
        
        return html_content
    
    def _generate_intermediate_data_section(self, intermediate_data_results: Dict) -> str:
        """
        生成中間層數據分析部分的HTML。
        
        Args:
            intermediate_data_results (Dict): 中間層數據分析結果。
        
        Returns:
            str: HTML內容。
        """
        # 確保結果是字典
        results_dict = intermediate_data_results
        
        # 生成HTML內容
        html_content = f"""
        <div class="section">
            <h2>中間層數據分析</h2>
        """
        
        layer_results = results_dict.get('layer_results', {})
        
        if layer_results:
            for layer_name, layer_data in layer_results.items():
                html_content += f"""
                <div class="sub-section">
                    <h3>層: {layer_name}</h3>
                    <p>分析輪次: {', '.join(map(str, layer_data.get('epochs_analyzed', [])))}</p>
                """
                
                # 如果有統計信息
                if 'statistics' in layer_data and layer_data['statistics']:
                    html_content += """
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
                    
                    stats = layer_data['statistics']
                    if isinstance(stats, dict):
                        # 處理不同的統計數據結構
                        if any(isinstance(k, int) for k in stats.keys()):
                            # 按epoch組織的統計數據
                            for epoch, epoch_stats in stats.items():
                                if isinstance(epoch_stats, dict):
                                    html_content += f"""
                                    <tr>
                                        <td>{epoch}</td>
                                        <td>{epoch_stats.get('mean', 'N/A')}</td>
                                        <td>{epoch_stats.get('std', 'N/A')}</td>
                                        <td>{epoch_stats.get('min', 'N/A')}</td>
                                        <td>{epoch_stats.get('max', 'N/A')}</td>
                                        <td>{epoch_stats.get('median', 'N/A')}</td>
                                    </tr>
                                    """
                        else:
                            # 直接包含統計值的字典
                            html_content += f"""
                            <tr>
                                <td>全部</td>
                                <td>{stats.get('mean', 'N/A')}</td>
                                <td>{stats.get('std', 'N/A')}</td>
                                <td>{stats.get('min', 'N/A')}</td>
                                <td>{stats.get('max', 'N/A')}</td>
                                <td>{stats.get('median', 'N/A')}</td>
                            </tr>
                            """
                    
                    html_content += """
                    </table>
                    """
                
                # 如果有稀疏度信息
                if 'sparsity' in layer_data and layer_data['sparsity']:
                    html_content += """
                    <h4>稀疏度分析</h4>
                    <table>
                        <tr>
                            <th>輪次</th>
                            <th>L1稀疏度</th>
                            <th>零元素比例</th>
                        </tr>
                    """
                    
                    sparsity_data = layer_data['sparsity']
                    if isinstance(sparsity_data, dict):
                        # 處理不同的稀疏度數據結構
                        if any(isinstance(k, int) for k in sparsity_data.keys()):
                            # 按epoch組織的稀疏度數據
                            for epoch, epoch_sparsity in sparsity_data.items():
                                if isinstance(epoch_sparsity, dict):
                                    l1_sparsity = epoch_sparsity.get('l1_sparsity', 'N/A')
                                    zero_ratio = epoch_sparsity.get('zero_ratio', 'N/A')
                                    if isinstance(zero_ratio, float):
                                        zero_ratio = f"{zero_ratio*100:.2f}%"
                                    html_content += f"""
                                    <tr>
                                        <td>{epoch}</td>
                                        <td>{l1_sparsity}</td>
                                        <td>{zero_ratio}</td>
                                    </tr>
                                    """
                        else:
                            # 直接包含稀疏度值的字典
                            l1_sparsity = sparsity_data.get('l1_sparsity', 'N/A')
                            zero_ratio = sparsity_data.get('zero_ratio', 'N/A')
                            if isinstance(zero_ratio, float):
                                zero_ratio = f"{zero_ratio*100:.2f}%"
                            html_content += f"""
                            <tr>
                                <td>全部</td>
                                <td>{l1_sparsity}</td>
                                <td>{zero_ratio}</td>
                            </tr>
                            """
                    
                    html_content += """
                    </table>
                    """
                
                html_content += """
                </div>
                """
        else:
            html_content += """
            <p>未找到中間層數據分析結果。</p>
            """
        
        html_content += """
        </div>
        """
        
        return html_content
    
    def _generate_markdown_report(self, results: Any) -> str:
        """
        生成Markdown格式的報告。
        
        Args:
            results (Any): 分析結果對象。
        
        Returns:
            str: 報告文件路徑。
        """
        # 將結果轉換為字典
        if hasattr(results, 'to_dict'):
            results_dict = results.to_dict()
        elif isinstance(results, dict):
            results_dict = results
        else:
            # 對於 AnalysisResults 對象，特殊處理
            results_dict = {
                'model_structure': results.get_model_structure_results() if hasattr(results, 'get_model_structure_results') else {},
                'training_dynamics': results.get_training_dynamics_results() if hasattr(results, 'get_training_dynamics_results') else {},
                'intermediate_data': results.get_intermediate_data_results() if hasattr(results, 'get_intermediate_data_results') else {},
                'plots': results.get_all_plots() if hasattr(results, 'get_all_plots') else {}
            }
        
        # 生成報告時間
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 報告文件名
        report_filename = f"{self.experiment_name}_analysis_report_{timestamp}.md"
        report_path = os.path.join(self.output_dir, report_filename)
        
        # 使用模板生成Markdown內容
        md_content = self.templates['markdown']
        
        # 替換模板變量
        md_content = md_content.replace("{{experiment_name}}", self.experiment_name)
        md_content = md_content.replace("{{analysis_time}}", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # 生成各部分內容
        model_structure_section = ""
        if 'model_structure' in results_dict and results_dict['model_structure']:
            model_structure_section = self._generate_model_structure_section_md(results_dict['model_structure'])
        md_content = md_content.replace("{{model_structure_section}}", model_structure_section)
        
        training_dynamics_section = ""
        if 'training_dynamics' in results_dict and results_dict['training_dynamics']:
            training_dynamics_section = self._generate_training_dynamics_section_md(results_dict['training_dynamics'])
        md_content = md_content.replace("{{training_dynamics_section}}", training_dynamics_section)
        
        intermediate_data_section = ""
        if 'intermediate_data' in results_dict and results_dict['intermediate_data']:
            intermediate_data_section = self._generate_intermediate_data_section_md(results_dict['intermediate_data'])
        md_content = md_content.replace("{{intermediate_data_section}}", intermediate_data_section)
        
        # 寫入文件
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return report_path
    
    def _generate_model_structure_section_md(self, model_structure_results: Dict) -> str:
        """
        生成模型結構分析部分的Markdown內容。
        
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
        生成訓練動態分析部分的Markdown內容。
        
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

| 指標 | 值 |
| ---- | ---- |
"""
            
            conv_analysis = training_dynamics_results['convergence_analysis']
            if isinstance(conv_analysis, dict):
                for metric, value in conv_analysis.items():
                    # 檢查值是否是字典還是直接值
                    if isinstance(value, dict):
                        # 如果是字典，使用原來的 get 方法獲取特定指標
                        epoch = value.get('convergence_epoch', 'N/A')
                        rate = value.get('convergence_rate', 'N/A')
                        final = value.get('final_value', 'N/A')
                        md += f"| {metric} | Epoch: {epoch}, Rate: {rate}, Final: {final} |\n"
                    else:
                        # 如果是直接值，直接顯示
                        md += f"| {metric} | {value} |\n"
        
        # 如果有穩定性分析
        if 'stability_analysis' in training_dynamics_results and training_dynamics_results['stability_analysis']:
            md += """
### 穩定性分析

| 指標 | 值 |
| ---- | ---- |
"""
            
            stab_analysis = training_dynamics_results['stability_analysis']
            if isinstance(stab_analysis, dict):
                for metric, value in stab_analysis.items():
                    # 檢查值是否是字典還是直接值
                    if isinstance(value, dict):
                        # 如果是字典，使用原來的 get 方法獲取特定指標
                        variance = value.get('variance', 'N/A')
                        max_fluct = value.get('max_fluctuation', 'N/A')
                        md += f"| {metric} | Variance: {variance}, Max Fluctuation: {max_fluct} |\n"
                    else:
                        # 如果是直接值，直接顯示
                        md += f"| {metric} | {value} |\n"
        
        return md
    
    def _generate_intermediate_data_section_md(self, intermediate_data_results: Dict) -> str:
        """
        生成中間層數據分析部分的Markdown內容。
        
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
                    
                    stats = layer_data['statistics']
                    if isinstance(stats, dict):
                        # 處理不同的統計數據結構
                        if any(isinstance(k, int) for k in stats.keys()):
                            # 按epoch組織的統計數據
                            for epoch, epoch_stats in stats.items():
                                if isinstance(epoch_stats, dict):
                                    mean = epoch_stats.get('mean', 'N/A')
                                    std = epoch_stats.get('std', 'N/A')
                                    min_val = epoch_stats.get('min', 'N/A')
                                    max_val = epoch_stats.get('max', 'N/A')
                                    median = epoch_stats.get('median', 'N/A')
                                    md += f"| {epoch} | {mean} | {std} | {min_val} | {max_val} | {median} |\n"
                        else:
                            # 直接包含統計值的字典
                            mean = stats.get('mean', 'N/A')
                            std = stats.get('std', 'N/A')
                            min_val = stats.get('min', 'N/A')
                            max_val = stats.get('max', 'N/A')
                            median = stats.get('median', 'N/A')
                            md += f"| 全部 | {mean} | {std} | {min_val} | {max_val} | {median} |\n"
                
                # 如果有稀疏度信息
                if 'sparsity' in layer_data and layer_data['sparsity']:
                    md += """
#### 稀疏度分析

| 輪次 | L1稀疏度 | 零元素比例 |
| ---- | -------- | ---------- |
"""
                    
                    sparsity_data = layer_data['sparsity']
                    if isinstance(sparsity_data, dict):
                        # 處理不同的稀疏度數據結構
                        if any(isinstance(k, int) for k in sparsity_data.keys()):
                            # 按epoch組織的稀疏度數據
                            for epoch, epoch_sparsity in sparsity_data.items():
                                if isinstance(epoch_sparsity, dict):
                                    l1_sparsity = epoch_sparsity.get('l1_sparsity', 'N/A')
                                    zero_ratio = epoch_sparsity.get('zero_ratio', 'N/A')
                                    if isinstance(zero_ratio, float):
                                        zero_ratio = f"{zero_ratio*100:.2f}%"
                                    md += f"| {epoch} | {l1_sparsity} | {zero_ratio} |\n"
                        else:
                            # 直接包含稀疏度值的字典
                            l1_sparsity = sparsity_data.get('l1_sparsity', 'N/A')
                            zero_ratio = sparsity_data.get('zero_ratio', 'N/A')
                            if isinstance(zero_ratio, float):
                                zero_ratio = f"{zero_ratio*100:.2f}%"
                            md += f"| 全部 | {l1_sparsity} | {zero_ratio} |\n"
        else:
            md += """
未找到中間層數據分析結果。
"""
        
        return md
    
    def _generate_pdf_report(self, results: Any) -> str:
        """
        生成PDF格式的報告。
        
        首先生成HTML報告，然後使用weasyprint或pdfkit將其轉換為PDF。
        
        Args:
            results (Any): 分析結果對象。
        
        Returns:
            str: 報告文件路徑。
        """
        # 先生成HTML報告
        html_path = self._generate_html_report(results)
        
        try:
            # 嘗試使用weasyprint
            return self._generate_pdf_with_weasyprint(html_path)
        except ImportError:
            self.logger.warning("未安裝weasyprint，嘗試使用pdfkit...")
            try:
                # 嘗試使用pdfkit作為備選
                return self._generate_pdf_with_pdfkit(html_path)
            except ImportError:
                self.logger.warning("未安裝pdfkit，返回HTML報告...")
                return html_path
    
    def _generate_pdf_with_weasyprint(self, html_path: str) -> str:
        """
        使用weasyprint生成PDF報告。
        
        Args:
            html_path (str): HTML報告文件路徑。
        
        Returns:
            str: PDF報告文件路徑。
        """
        import weasyprint
        
        # 生成PDF文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"{self.experiment_name}_analysis_report_{timestamp}.pdf"
        pdf_path = os.path.join(self.output_dir, pdf_filename)
        
        # 轉換HTML為PDF
        html = weasyprint.HTML(filename=html_path)
        html.write_pdf(pdf_path)
        
        self.logger.info(f"已生成PDF報告: {pdf_path}")
        return pdf_path
    
    def _generate_pdf_with_pdfkit(self, html_path: str) -> str:
        """
        使用pdfkit生成PDF報告。
        
        Args:
            html_path (str): HTML報告文件路徑。
        
        Returns:
            str: PDF報告文件路徑。
        """
        import pdfkit
        
        # 生成PDF文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"{self.experiment_name}_analysis_report_{timestamp}.pdf"
        pdf_path = os.path.join(self.output_dir, pdf_filename)
        
        # 設置轉換選項
        options = {
            'encoding': 'UTF-8',
            'quiet': ''
        }
        
        # 轉換HTML為PDF
        pdfkit.from_file(html_path, pdf_path, options=options)
        
        self.logger.info(f"已生成PDF報告: {pdf_path}")
        return pdf_path
    
    def set_template(self, format: str, template_content: str) -> None:
        """
        設置指定格式的報告模板。
        
        Args:
            format (str): 模板格式，可選 'html' 或 'markdown'。
            template_content (str): 模板內容。
        """
        if format.lower() not in ['html', 'markdown']:
            self.logger.error(f"不支持的模板格式: {format}")
            raise ValueError(f"不支持的模板格式: {format}")
        
        self.templates[format.lower()] = template_content
        self.logger.info(f"已設置{format}模板") 