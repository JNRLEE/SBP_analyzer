"""
# 分析器接口
# 提供分析器的主接口類別
"""

import logging
import os
from typing import Dict, List, Any, Optional, Union

from data_loader import ExperimentLoader, HookDataLoader
from utils.file_utils import ensure_dir, get_experiment_id


class SBPAnalyzer:
    """
    SBPAnalyzer 的主接口類別，用於啟動分析流程。
    
    Args:
        experiment_dir: 實驗結果目錄的路徑
        output_dir: 輸出目錄，預設為 "analysis_results"
        log_level: 日誌級別，預設為 INFO
        
    Returns:
        None
        
    Description:
        此類別協調不同的分析器組件，載入數據，執行分析，並產生報告。
        
    References:
        None
    """
    
    def __init__(
        self,
        experiment_dir: str,
        output_dir: str = "analysis_results",
        log_level: int = logging.INFO
    ):
        self.experiment_dir = experiment_dir
        self.output_dir = output_dir
        self.logger = self._setup_logger(log_level)
        self.results = {}
        self.experiment_id = get_experiment_id(experiment_dir)
        
        # 初始化數據載入器
        self.experiment_loader = ExperimentLoader(experiment_dir, log_level)
        self.hook_loader = HookDataLoader(experiment_dir, log_level)
        
        # 確保輸出目錄存在
        ensure_dir(output_dir)
        
        self.logger.info(f"初始化 SBPAnalyzer，實驗目錄: {experiment_dir}，輸出目錄: {output_dir}")
    
    def _setup_logger(self, log_level: int) -> logging.Logger:
        """
        設置日誌記錄器。
        
        Args:
            log_level: 日誌級別
            
        Returns:
            logging.Logger: 設置好的日誌記錄器
            
        Description:
            創建並配置一個日誌記錄器實例，用於記錄分析過程中的信息。
            
        References:
            None
        """
        logger = logging.getLogger("SBPAnalyzer")
        logger.setLevel(log_level)
        
        # 避免重複處理器
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def analyze(
        self,
        analyze_model_structure: bool = True,
        analyze_training_history: bool = True,
        analyze_hooks: bool = False,
        epochs: Optional[List[int]] = None,
        layers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        執行分析流程。
        
        Args:
            analyze_model_structure: 是否分析模型結構
            analyze_training_history: 是否分析訓練歷史
            analyze_hooks: 是否分析模型鉤子數據
            epochs: 要分析的特定輪次列表，如果為 None 則分析所有可用輪次
            layers: 要分析的特定層名稱列表，如果為 None 則分析所有可用層
            
        Returns:
            Dict[str, Any]: 分析結果字典
            
        Description:
            協調不同的分析器組件，根據指定的參數執行相應的分析任務。
            
        References:
            None
        """
        self.logger.info("開始分析流程")
        
        # 根據指定的分析選項執行分析
        if analyze_model_structure:
            self._analyze_model_structure()
        
        if analyze_training_history:
            self._analyze_training_history()
        
        if analyze_hooks:
            self._analyze_hooks(epochs, layers)
        
        self.logger.info("分析流程完成")
        
        return self.results
    
    def _analyze_model_structure(self) -> None:
        """
        分析模型結構。
        
        Args:
            None
            
        Returns:
            None
            
        Description:
            載入模型結構數據，並執行結構分析。
            
        References:
            None
        """
        self.logger.info("分析模型結構")
        
        # 載入模型結構
        model_structure = self.experiment_loader.load(data_type='model_structure')
        if model_structure:
            # 簡單的結構摘要
            self.results['model_structure'] = model_structure
            self.logger.info(f"載入了模型結構數據: {len(model_structure) if model_structure else 0} 個鍵")
        else:
            self.logger.warning("未能載入模型結構數據")
    
    def _analyze_training_history(self) -> None:
        """
        分析訓練歷史。
        
        Args:
            None
            
        Returns:
            None
            
        Description:
            載入並分析訓練歷史數據，包括損失曲線和指標趨勢。
            
        References:
            None
        """
        self.logger.info("分析訓練歷史")
        
        # 載入訓練歷史
        training_history = self.experiment_loader.load(data_type='training_history')
        if training_history:
            # 存儲原始數據
            self.results['training_history'] = training_history
            
            # 計算一些基本的指標
            metrics_summary = {}
            for metric_name, values in training_history.items():
                if isinstance(values, list) and values:
                    # 對於數值型指標，計算均值和範圍
                    try:
                        metrics_summary[metric_name] = {
                            'mean': sum(values) / len(values),
                            'min': min(values),
                            'max': max(values),
                            'start': values[0],
                            'end': values[-1],
                            'improvement': values[0] - values[-1] if 'loss' in metric_name.lower() else values[-1] - values[0]
                        }
                    except (TypeError, ValueError):
                        # 跳過非數值型數據
                        pass
            
            self.results['metrics_summary'] = metrics_summary
            self.logger.info(f"載入並分析了訓練歷史數據: {len(training_history) if training_history else 0} 個指標")
        else:
            self.logger.warning("未能載入訓練歷史數據")
    
    def _analyze_hooks(
        self,
        epochs: Optional[List[int]] = None,
        layers: Optional[List[str]] = None
    ) -> None:
        """
        分析模型鉤子數據。
        
        Args:
            epochs: 要分析的特定輪次列表
            layers: 要分析的特定層名稱列表
            
        Returns:
            None
            
        Description:
            載入並分析模型鉤子數據，包括激活值和中間層輸出。
            
        References:
            None
        """
        self.logger.info("分析模型鉤子數據")
        
        # 確定要分析的輪次
        available_epochs = self.hook_loader.list_available_epochs()
        if not available_epochs:
            self.logger.warning("未找到可用的鉤子數據輪次")
            return
        
        if epochs is None:
            # 如果未指定，使用所有可用輪次
            epochs_to_analyze = available_epochs
        else:
            # 篩選出可用的輪次
            epochs_to_analyze = [e for e in epochs if e in available_epochs]
            if not epochs_to_analyze:
                self.logger.warning(f"指定的輪次 {epochs} 不可用，可用的輪次為: {available_epochs}")
                return
        
        # 分析每個輪次的鉤子數據
        hooks_results = {}
        for epoch in epochs_to_analyze:
            epoch_summary = self.hook_loader.load(data_type='epoch_summary', epoch=epoch)
            if epoch_summary:
                hooks_results[f'epoch_{epoch}'] = {'summary': epoch_summary}
                
                # 確定要分析的層
                available_layers = self.hook_loader.list_available_layer_activations(epoch)
                if not available_layers:
                    self.logger.warning(f"輪次 {epoch} 未找到可用的層激活值數據")
                    continue
                
                if layers is None:
                    # 如果未指定，使用所有可用層
                    layers_to_analyze = available_layers
                else:
                    # 篩選出可用的層
                    layers_to_analyze = [l for l in layers if l in available_layers]
                    if not layers_to_analyze:
                        self.logger.warning(f"指定的層 {layers} 不可用，可用的層為: {available_layers}")
                        continue
                
                # 分析每個層的激活值
                for layer in layers_to_analyze:
                    layer_data = self.hook_loader.load(
                        data_type='layer_activation',
                        layer_name=layer,
                        epoch=epoch,
                        batch=0  # 僅分析第 0 批次
                    )
                    if layer_data is not None:
                        # 使用 tensor_utils 計算張量統計量
                        # 這裡只是簡單地存儲層數據，實際應用中可能會有更複雜的分析
                        hooks_results[f'epoch_{epoch}'][layer] = {
                            'shape': layer_data.shape,
                            'type': str(layer_data.dtype),
                            # 這裡可以添加其他分析結果
                        }
        
        self.results['hooks_analysis'] = hooks_results
        self.logger.info(f"完成對 {len(epochs_to_analyze)} 個輪次的鉤子數據分析")
    
    def generate_report(
        self,
        output_dir: Optional[str] = None,
        report_format: str = 'markdown'
    ) -> str:
        """
        生成分析報告。
        
        Args:
            output_dir: 報告輸出目錄，如果為 None 則使用默認的輸出目錄
            report_format: 報告格式，可選值為 'markdown', 'html'
            
        Returns:
            str: 報告文件的路徑
            
        Description:
            根據分析結果生成結構化的報告，可以選擇不同的輸出格式。
            
        References:
            None
        """
        self.logger.info(f"生成 {report_format} 格式的報告")
        
        # 使用指定的輸出目錄或默認目錄
        output_dir = output_dir or self.output_dir
        ensure_dir(output_dir)
        
        # 根據報告格式選擇檔案副檔名
        extension = '.md' if report_format == 'markdown' else '.html'
        report_filename = f"analysis_report{extension}"
        
        # 如果有實驗 ID，將其加入檔名
        if self.experiment_id:
            report_filename = f"{self.experiment_id}_analysis_report{extension}"
        
        report_path = os.path.join(output_dir, report_filename)
        
        # 生成報告內容
        content = self._generate_report_content(report_format)
        
        # 寫入檔案
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.logger.info(f"報告已保存至: {report_path}")
        return report_path
    
    def _generate_report_content(self, report_format: str) -> str:
        """
        生成報告內容。
        
        Args:
            report_format: 報告格式，可選值為 'markdown', 'html'
            
        Returns:
            str: 報告內容
            
        Description:
            根據分析結果生成報告內容，可以選擇不同的輸出格式。
            
        References:
            None
        """
        # 報告標題
        title = f"SBP Analyzer 分析報告" + (f" - {self.experiment_id}" if self.experiment_id else "")
        
        # 根據不同格式生成內容
        if report_format == 'markdown':
            content = f"# {title}\n\n"
            
            # 基本信息部分
            content += "## 基本信息\n\n"
            content += f"- 實驗目錄: `{self.experiment_dir}`\n"
            content += f"- 分析時間: {logging._startTime}\n\n"
            
            # 配置信息
            config = self.experiment_loader.load(data_type='config')
            if config:
                content += "## 實驗配置\n\n"
                content += "```json\n"
                import json
                content += json.dumps(config, indent=2, ensure_ascii=False)
                content += "\n```\n\n"
            
            # 訓練歷史摘要
            if 'metrics_summary' in self.results:
                content += "## 訓練指標摘要\n\n"
                content += "| 指標 | 起始值 | 結束值 | 改善量 | 均值 | 最小值 | 最大值 |\n"
                content += "|------|--------|--------|--------|------|--------|--------|\n"
                
                for metric, stats in self.results['metrics_summary'].items():
                    content += f"| {metric} | {stats['start']:.4f} | {stats['end']:.4f} | "
                    content += f"{stats['improvement']:.4f} | {stats['mean']:.4f} | "
                    content += f"{stats['min']:.4f} | {stats['max']:.4f} |\n"
                
                content += "\n"
            
            # 模型結構摘要
            if 'model_structure' in self.results:
                content += "## 模型結構摘要\n\n"
                # 在此添加模型結構的摘要信息，根據實際結構格式調整
                content += "模型結構信息可用，但為了簡潔未在報告中詳細列出。\n\n"
            
            # 鉤子數據分析
            if 'hooks_analysis' in self.results:
                content += "## 鉤子數據分析\n\n"
                
                for epoch_key, epoch_data in self.results['hooks_analysis'].items():
                    content += f"### {epoch_key}\n\n"
                    
                    # 排除 summary 以列出層信息
                    layers = [k for k in epoch_data.keys() if k != 'summary']
                    
                    if layers:
                        content += "| 層名稱 | 形狀 | 類型 |\n"
                        content += "|--------|------|------|\n"
                        
                        for layer in layers:
                            layer_info = epoch_data[layer]
                            content += f"| {layer} | {layer_info.get('shape', 'N/A')} | {layer_info.get('type', 'N/A')} |\n"
                    
                    content += "\n"
        
        else:  # HTML 格式
            content = f"<html><head><title>{title}</title></head><body>\n"
            content += f"<h1>{title}</h1>\n"
            
            # 基本信息部分
            content += "<h2>基本信息</h2>\n"
            content += f"<p>實驗目錄: <code>{self.experiment_dir}</code></p>\n"
            content += f"<p>分析時間: {logging._startTime}</p>\n"
            
            # 配置信息
            config = self.experiment_loader.load(data_type='config')
            if config:
                content += "<h2>實驗配置</h2>\n"
                content += "<pre><code>"
                import json
                content += json.dumps(config, indent=2, ensure_ascii=False)
                content += "</code></pre>\n"
            
            # 訓練歷史摘要
            if 'metrics_summary' in self.results:
                content += "<h2>訓練指標摘要</h2>\n"
                content += "<table border='1'>\n"
                content += "<tr><th>指標</th><th>起始值</th><th>結束值</th><th>改善量</th>"
                content += "<th>均值</th><th>最小值</th><th>最大值</th></tr>\n"
                
                for metric, stats in self.results['metrics_summary'].items():
                    content += f"<tr><td>{metric}</td><td>{stats['start']:.4f}</td><td>{stats['end']:.4f}</td>"
                    content += f"<td>{stats['improvement']:.4f}</td><td>{stats['mean']:.4f}</td>"
                    content += f"<td>{stats['min']:.4f}</td><td>{stats['max']:.4f}</td></tr>\n"
                
                content += "</table>\n"
            
            # 模型結構摘要
            if 'model_structure' in self.results:
                content += "<h2>模型結構摘要</h2>\n"
                content += "<p>模型結構信息可用，但為了簡潔未在報告中詳細列出。</p>\n"
            
            # 鉤子數據分析
            if 'hooks_analysis' in self.results:
                content += "<h2>鉤子數據分析</h2>\n"
                
                for epoch_key, epoch_data in self.results['hooks_analysis'].items():
                    content += f"<h3>{epoch_key}</h3>\n"
                    
                    # 排除 summary 以列出層信息
                    layers = [k for k in epoch_data.keys() if k != 'summary']
                    
                    if layers:
                        content += "<table border='1'>\n"
                        content += "<tr><th>層名稱</th><th>形狀</th><th>類型</th></tr>\n"
                        
                        for layer in layers:
                            layer_info = epoch_data[layer]
                            content += f"<tr><td>{layer}</td><td>{layer_info.get('shape', 'N/A')}</td>"
                            content += f"<td>{layer_info.get('type', 'N/A')}</td></tr>\n"
                        
                        content += "</table>\n"
            
            content += "</body></html>"
        
        return content 