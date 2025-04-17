# Docstring 標準格式指南

本文檔提供標準化 docstring 格式要求以及如何使用自動化工具檢查和修復 docstring 問題。

## Docstring 標準格式

所有模組、類、函數和方法都必須遵循以下 docstring 格式標準：

### 函數與方法 Docstring 格式

```python
def example_function(param1, param2):
    """
    函數的簡要描述（一句話）。

    函數的詳細描述，可以包含多行文字，解釋函數的用途、功能、
    實現方式、使用場景等相關資訊。

    Args:
        param1 (type): 參數1的描述
        param2 (type): 參數2的描述

    Returns:
        type: 返回值的描述

    Raises:
        ExceptionType: 異常情況的描述（可選）

    Examples:
        >>> example_function(1, 2)
        3
    """
    pass
```

### 類 Docstring 格式

```python
class ExampleClass:
    """
    類的簡要描述（一句話）。

    類的詳細描述，可以包含多行文字，解釋類的用途、設計理念等。

    Attributes:
        attr1 (type): 屬性1的描述
        attr2 (type): 屬性2的描述
    """
    
    def __init__(self, param1):
        """
        初始化方法。

        Args:
            param1 (type): 參數1的描述

        Returns:
            None
        """
        self.attr1 = param1
```

### 模組 Docstring 格式

```python
"""
模組名稱

模組的簡要描述，說明模組的功能與用途。

模組的詳細描述，可以包含更多資訊，如：
- 主要功能
- 依賴關係
- 使用範例
- 注意事項等
"""

import numpy as np
```

## 使用自動化工具管理 Docstring

我們提供了兩個工具來協助管理和標準化 docstring：

### 1. Docstring 檢查工具

`scripts/docstring_checker.py` 用於檢查專案中不符合標準的 docstring。

使用方法：

```bash
# 檢查整個專案
python scripts/docstring_checker.py .

# 檢查特定目錄，並生成報告
python scripts/docstring_checker.py data_loader --output docstring_report.md

# 顯示標準 docstring 模板
python scripts/docstring_checker.py --template

# 排除特定目錄
python scripts/docstring_checker.py . --exclude venv tests
```

### 2. Docstring 修復工具

`scripts/docstring_fixer.py` 用於自動修復缺少標準 docstring 的檔案。

使用方法：

```bash
# 根據檢查報告修復 docstring
python scripts/docstring_fixer.py docstring_report.md

# 預覽修改而不實際執行
python scripts/docstring_fixer.py docstring_report.md --dry-run
```

## 常見修復流程

1. 執行檢查工具生成報告
   ```bash
   python scripts/docstring_checker.py . --output docstring_report.md
   ```

2. 檢視報告，確認需要修復的問題
   ```bash
   cat docstring_report.md
   ```

3. 執行修復工具自動添加缺少的 docstring
   ```bash
   python scripts/docstring_fixer.py docstring_report.md
   ```

4. 手動審查自動生成的 docstring，填寫 TODO 項目

5. 再次執行檢查工具，確認修復效果
   ```bash
   python scripts/docstring_checker.py . --output docstring_report_after.md
   ```

## 最佳實踐

1. 在開發新功能前先查看此指南，遵循標準格式
2. 在提交 PR 前運行檢查工具確認 docstring 符合標準
3. 審查人員應檢查 docstring 的完整性和準確性
4. 定期執行自動化工具保持專案文檔的一致性 