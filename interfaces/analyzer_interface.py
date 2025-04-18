"""
分析器接口模組。

此模組提供用於整合各種分析器並提供統一接口的功能。

Classes:
    SBPAnalyzer: SBP分析器的主接口類別。
    AnalysisResults: 封裝分析結果的類別。
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union, Set

from sbp_analyzer.analyzer import (
    ModelStructureAnalyzer,
    TrainingDynamicsAnalyzer,
    IntermediateDataAnalyzer
)
from sbp_analyzer.data_loader import ExperimentLoader, HookDataLoader
from sbp_analyzer.reporter import ReportGenerator

class AnalysisResults:
    """
    封裝分析結果的類別。
    
    此類別用於整合和存儲來自不同分析器的分析結果，並提供方便的訪問方法。
    
    Attributes:
        model_structure_results (Dict): 模型結構分析結果。
        training_dynamics_results (Dict): 訓練動態分析結果。
        intermediate_data_results (Dict): 中間層數據分析結果。
        plots (Dict[str, Any]): 生成的圖表。
    """
    
    def __init__(self):
        """
        初始化分析結果對象。
        """
        self.model_structure_results = {}
        self.training_dynamics_results = {}
        self.intermediate_data_results = {}
        self.plots = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_model_structure_results(self, results: Dict) -> None:
        """
        添加模型結構分析結果。
        
        Args:
            results (Dict): 模型結構分析結果。
        """
        self.model_structure_results = results
    
    def add_training_dynamics_results(self, results: Dict) -> None:
        """
        添加訓練動態分析結果。
        
        Args:
            results (Dict): 訓練動態分析結果。
        """
        self.training_dynamics_results = results
    
    def add_intermediate_data_results(self, results: Dict) -> None:
        """
        添加中間層數據分析結果。
        
        Args:
            results (Dict): 中間層數據分析結果。
        """
        self.intermediate_data_results = results
    
    def add_plot(self, name: str, plot: Any) -> None:
        """
        添加圖表。
        
        Args:
            name (str): 圖表名稱。
            plot (Any): 圖表對象。
        """
        self.plots[name] = plot
    
    def get_model_summary(self) -> Dict:
        """
        獲取模型摘要。
        
        Returns:
            Dict: 模型摘要字典。
        """
        if not self.model_structure_results:
            self.logger.warning("模型結構分析結果不可用")
            return {}
        
        return {
            'model_name': self.model_structure_results.get('model_name', 'Unknown'),
            'total_params': self.model_structure_results.get('total_params', 0),
            'trainable_params': self.model_structure_results.get('trainable_params', 0),
            'layer_count': self.model_structure_results.get('layer_count', 0)
        }
    
    def get_plot(self, name: str) -> Optional[Any]:
        """
        獲取指定名稱的圖表。
        
        Args:
            name (str): 圖表名稱。
        
        Returns:
            Optional[Any]: 圖表對象，如果不存在則返回None。
        """
        return self.plots.get(name)
    
    def get_activation_distribution(self, layer_name: str, epoch: int) -> Optional[Dict]:
        """
        獲取特定層在特定輪次的激活值分布。
        
        Args:
            layer_name (str): 層的名稱。
            epoch (int): 輪次。
        
        Returns:
            Optional[Dict]: 分布數據字典，如果不存在則返回None。
        """
        if not self.intermediate_data_results or 'layer_results' not in self.intermediate_data_results:
            self.logger.warning("中間層數據分析結果不可用")
            return None
        
        layer_results = self.intermediate_data_results['layer_results'].get(layer_name, {})
        if not layer_results or 'distribution' not in layer_results:
            self.logger.warning(f"層 {layer_name} 的結果不可用")
            return None
        
        return layer_results['distribution'].get(epoch)
    
    def to_dict(self) -> Dict:
        """
        將分析結果轉換為字典。
        
        Returns:
            Dict: 包含所有分析結果的字典。
        """
        return {
            'model_structure': self.model_structure_results,
            'training_dynamics': self.training_dynamics_results,
            'intermediate_data': self.intermediate_data_results
        }

class SBPAnalyzer:
    """
    SBP分析器的主接口類別。
    
    此類別整合了各種分析器，提供了統一的接口來執行分析和生成報告。
    
    Attributes:
        experiment_dir (str): 實驗結果的目錄路徑。
        experiment_loader (ExperimentLoader): 實驗數據載入器。
        hook_loader (HookDataLoader): Hook數據載入器。
        model_structure_analyzer (ModelStructureAnalyzer): 模型結構分析器。
        training_dynamics_analyzer (TrainingDynamicsAnalyzer): 訓練動態分析器。
        intermediate_data_analyzer (IntermediateDataAnalyzer): 中間層數據分析器。
        report_generator (ReportGenerator): 報告生成器。
        results (AnalysisResults): 分析結果。
    """
    
    def __init__(self, experiment_dir: str):
        """
        初始化SBP分析器。
        
        Args:
            experiment_dir (str): 實驗結果的目錄路徑。
        """
        self.experiment_dir = experiment_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 確認實驗目錄是否存在
        if not os.path.exists(experiment_dir):
            self.logger.error(f"實驗目錄不存在: {experiment_dir}")
            raise FileNotFoundError(f"實驗目錄不存在: {experiment_dir}")
        
        # 初始化數據載入器
        self.experiment_loader = ExperimentLoader(experiment_dir)
        self.hook_loader = HookDataLoader(experiment_dir)
        
        # 初始化分析器
        self.model_structure_analyzer = ModelStructureAnalyzer(experiment_dir)
        self.training_dynamics_analyzer = TrainingDynamicsAnalyzer(experiment_dir)
        self.intermediate_data_analyzer = IntermediateDataAnalyzer(experiment_dir)
        
        # 初始化報告生成器
        self.report_generator = None  # 將在generate_report方法中初始化
        
        # 初始化結果對象
        self.results = AnalysisResults()
    
    def analyze(self, analyze_model_structure: bool = True, 
                analyze_training_history: bool = True, 
                analyze_hooks: bool = True,
                epochs: Optional[List[int]] = None,
                layers: Optional[List[str]] = None,
                batch: int = 0,
                save_results: bool = True) -> AnalysisResults:
        """
        執行分析。
        
        Args:
            analyze_model_structure (bool, optional): 是否分析模型結構。默認為True。
            analyze_training_history (bool, optional): 是否分析訓練歷史。默認為True。
            analyze_hooks (bool, optional): 是否分析hook數據。默認為True。
            epochs (List[int], optional): 要分析的輪次列表。如果為None，則分析所有可用輪次。
            layers (List[str], optional): 要分析的層列表。如果為None，則分析所有可用層。
            batch (int, optional): 要分析的批次索引。默認為0。
            save_results (bool, optional): 是否保存分析結果。默認為True。
        
        Returns:
            AnalysisResults: 分析結果對象。
        """
        self.logger.info(f"開始分析實驗: {os.path.basename(self.experiment_dir)}")
        
        # 分析模型結構
        if analyze_model_structure:
            self.logger.info("分析模型結構...")
            model_structure_results = self.model_structure_analyzer.analyze(save_results=save_results)
            self.results.add_model_structure_results(model_structure_results)
        
        # 分析訓練歷史
        if analyze_training_history:
            self.logger.info("分析訓練動態...")
            training_dynamics_results = self.training_dynamics_analyzer.analyze(save_results=save_results)
            self.results.add_training_dynamics_results(training_dynamics_results)
        
        # 分析Hook數據
        if analyze_hooks:
            self.logger.info("分析中間層數據...")
            intermediate_data_results = self.intermediate_data_analyzer.analyze(
                layers=layers, 
                epochs=epochs, 
                batch=batch, 
                save_results=save_results
            )
            self.results.add_intermediate_data_results(intermediate_data_results)
        
        self.logger.info("分析完成")
        return self.results
    
    def generate_report(self, output_dir: str, report_format: str = 'html') -> str:
        """
        生成分析報告。
        
        Args:
            output_dir (str): 輸出目錄路徑。
            report_format (str, optional): 報告格式，可選 'html' 或 'markdown'。默認為'html'。
        
        Returns:
            str: 報告文件路徑。
        """
        from sbp_analyzer.reporter import ReportGenerator
        
        self.logger.info(f"生成 {report_format} 格式的分析報告...")
        
        # 如果還沒有執行分析，則先執行
        if not (self.results.model_structure_results or 
                self.results.training_dynamics_results or 
                self.results.intermediate_data_results):
            self.logger.warning("還沒有執行分析，先執行分析...")
            self.analyze()
        
        # 初始化報告生成器
        self.report_generator = ReportGenerator(
            experiment_name=os.path.basename(self.experiment_dir),
            output_dir=output_dir
        )
        
        # 生成報告
        report_path = self.report_generator.generate(
            results=self.results,
            format=report_format
        )
        
        self.logger.info(f"報告已生成: {report_path}")
        return report_path 