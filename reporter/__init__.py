"""
reporter 模組

此模組包含以下主要組件：

Classes:
    ReportGenerator: 報告生成器，用於生成分析報告。

此類別可以基於分析結果生成HTML、Markdown或PDF格式的報告。

Attributes:
    experiment_name (str): 實驗名稱。
    output_dir (str): 輸出目錄路徑。
    logger (logging (report_generator.py)


完整的模組索引請參見：FUNCTION_CLASS_INDEX.md
"""

from .report_generator import ReportGenerator

__all__ = [
    'ReportGenerator'
]