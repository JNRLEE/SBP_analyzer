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

from analyzer import (
    ModelStructureAnalyzer,
    TrainingDynamicsAnalyzer,
    IntermediateDataAnalyzer,
    InferenceAnalyzer
)
from data_loader import ExperimentLoader, HookDataLoader
from reporter import ReportGenerator

class AnalysisResults:
    """
    封裝分析結果的類別。
    
    此類別用於整合和存儲來自不同分析器的分析結果，並提供方便的訪問方法。
    
    Attributes:
        model_structure_results (Dict): 模型結構分析結果。
        training_dynamics_results (Dict): 訓練動態分析結果。
        intermediate_data_results (Dict): 中間層數據分析結果。
        inference_results (Dict): 推理性能分析結果。
        plots (Dict[str, Any]): 生成的圖表。
    """
    
    def __init__(self):
        """
        初始化分析結果對象。
        """
        self.model_structure_results = {}
        self.training_dynamics_results = {}
        self.intermediate_data_results = {}
        self.inference_results = {}
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
    
    def add_inference_results(self, results: Dict) -> None:
        """
        添加推理性能分析結果。
        
        Args:
            results (Dict): 推理性能分析結果。
        """
        self.inference_results = results
    
    def add_plot(self, plot_name: str, plot_data: Any) -> None:
        """
        添加圖表。
        
        Args:
            plot_name (str): 圖表名稱。
            plot_data (Any): 圖表數據。
        """
        self.plots[plot_name] = plot_data
    
    def get_model_structure_results(self) -> Dict:
        """
        獲取模型結構分析結果。
        
        Returns:
            Dict: 模型結構分析結果。
        """
        return self.model_structure_results
    
    def get_training_dynamics_results(self) -> Dict:
        """
        獲取訓練動態分析結果。
        
        Returns:
            Dict: 訓練動態分析結果。
        """
        return self.training_dynamics_results
    
    def get_intermediate_data_results(self) -> Dict:
        """
        獲取中間層數據分析結果。
        
        Returns:
            Dict: 中間層數據分析結果。
        """
        return self.intermediate_data_results
    
    def get_inference_results(self) -> Dict:
        """
        獲取推理性能分析結果。
        
        Returns:
            Dict: 推理性能分析結果。
        """
        return self.inference_results
    
    def get_plot(self, plot_name: str) -> Any:
        """
        獲取指定名稱的圖表。
        
        Args:
            plot_name (str): 圖表名稱。
        
        Returns:
            Any: 圖表數據，如果不存在則返回None。
        """
        return self.plots.get(plot_name)
    
    def get_all_plots(self) -> Dict[str, Any]:
        """
        獲取所有圖表。
        
        Returns:
            Dict[str, Any]: 包含所有圖表的字典。
        """
        return self.plots


class SBPAnalyzer:
    """
    SBP分析器的主接口類。
    
    此類整合了各種不同類型的分析器，並提供了統一的接口。
    
    Attributes:
        experiment_dir (str): 實驗結果的目錄路徑。
        experiment_loader (ExperimentLoader): 實驗數據加載器。
        model_structure_analyzer (ModelStructureAnalyzer): 模型結構分析器。
        training_dynamics_analyzer (TrainingDynamicsAnalyzer): 訓練動態分析器。
        intermediate_data_analyzer (IntermediateDataAnalyzer): 中間層數據分析器。
        inference_analyzer (InferenceAnalyzer): 推理性能分析器。
        results (AnalysisResults): 分析結果。
    """
    
    def __init__(self, experiment_dir: str):
        """
        初始化SBP分析器。
        
        Args:
            experiment_dir (str): 實驗結果的目錄路徑。
        """
        self.experiment_dir = experiment_dir
        self.experiment_loader = ExperimentLoader(experiment_dir)
        self.model_structure_analyzer = ModelStructureAnalyzer(experiment_dir)
        self.training_dynamics_analyzer = TrainingDynamicsAnalyzer(experiment_dir)
        self.intermediate_data_analyzer = IntermediateDataAnalyzer(experiment_dir)
        self.inference_analyzer = InferenceAnalyzer(experiment_dir)
        self.results = AnalysisResults()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze(self, analyze_model_structure: bool = True, 
                analyze_training_history: bool = True, 
                analyze_hooks: bool = True,
                analyze_inference: bool = False,
                epochs: Optional[List[int]] = None,
                layers: Optional[List[str]] = None,
                batch: int = 0,
                batch_sizes: Optional[List[int]] = None,
                input_shapes: Optional[List[List[int]]] = None,
                save_results: bool = True) -> str:
        """
        執行分析。
        
        Args:
            analyze_model_structure (bool, optional): 是否分析模型結構。默認為True。
            analyze_training_history (bool, optional): 是否分析訓練歷史。默認為True。
            analyze_hooks (bool, optional): 是否分析hook數據。默認為True。
            analyze_inference (bool, optional): 是否分析推理性能。默認為False。
            epochs (List[int], optional): 要分析的輪次列表。如果為None，則分析所有可用輪次。
            layers (List[str], optional): 要分析的層列表。如果為None，則分析所有可用層。
            batch (int, optional): 要分析的批次索引。默認為0。
            batch_sizes (List[int], optional): 要分析的批次大小列表。用於推理分析。
            input_shapes (List[List[int]], optional): 要分析的輸入形狀列表。用於推理分析。
            save_results (bool, optional): 是否保存分析結果。默認為True。
        
        Returns:
            str: 生成的報告文件路徑
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
        
        # 分析中間層數據
        if analyze_hooks:
            self.logger.info("分析中間層數據...")
            # 載入Hook數據，不再需要提供layers參數
            hook_data_loader = HookDataLoader(self.experiment_dir)
            try:
                hook_data = hook_data_loader.load_hook_data(
                    epochs=epochs, 
                    layers=layers, 
                    batch_idx=batch
                )
                
                intermediate_data_results = self.intermediate_data_analyzer.analyze(
                    hook_data=hook_data,  # 明確傳入hook_data
                    save_results=save_results
                )
                
                self.results.add_intermediate_data_results(intermediate_data_results)
            except Exception as e:
                self.logger.error(f"載入Hook數據或分析中間層數據時發生錯誤: {e}")
                # 繼續分析其他部分，不因為Hook數據的問題而中斷整個分析流程
            
        # 分析推理性能
        if analyze_inference:
            self.logger.info("分析推理性能...")
            inference_results = self.inference_analyzer.analyze(
                batch_sizes=batch_sizes, 
                input_shapes=input_shapes, 
                save_results=save_results
            )
            self.results.add_inference_results(inference_results)
        
        # 生成報告
        self.logger.info("生成分析報告...")
        report_path = self.generate_report()
        
        return report_path
    
    def generate_report(self, output_dir: Optional[str] = None, 
                      report_format: str = 'html',
                      template_dir: Optional[str] = None) -> str:
        """
        生成分析報告。
        
        Args:
            output_dir (Optional[str], optional): 報告輸出目錄。如果為None，則使用默認路徑。
            report_format (str, optional): 報告格式，可選值為 'html'、'markdown' 或 'pdf'。默認為 'html'。
            template_dir (Optional[str], optional): 模板目錄路徑。若為None，則使用默認模板。
        
        Returns:
            str: 報告路徑。
        """
        if output_dir is None:
            output_dir = os.path.join(self.experiment_dir, 'analysis_report')
        
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 從config.json獲取實驗名稱
        try:
            config = self.experiment_loader.load_config()
            experiment_name = config.get('experiment_name', os.path.basename(self.experiment_dir))
        except:
            # 如果無法獲取，使用目錄名稱
            experiment_name = os.path.basename(self.experiment_dir)
        
        # 創建報告生成器
        report_generator = ReportGenerator(
            experiment_name=experiment_name,
            output_dir=output_dir,
            template_dir=template_dir
        )
        
        # 生成報告
        report_path = report_generator.generate(self.results, format=report_format)
        self.logger.info(f"分析報告已生成: {report_path}")
        
        return report_path
    
    def get_results(self) -> AnalysisResults:
        """
        獲取分析結果。
        
        Returns:
            AnalysisResults: 分析結果對象。
        """
        return self.results 