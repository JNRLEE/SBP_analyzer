# SBP Analyzer 貢獻指南

## 函數與類別索引維護

為了幫助開發人員有效地理解和導覽專案，SBP Analyzer 提供了一個完整的函數與類別索引系統，包括：

1. **主索引文件**：`FUNCTION_CLASS_INDEX.md` 提供整個專案的完整函數與類別索引，按照資料夾樹狀結構排列。
2. **模塊級索引**：每個主要模塊的 `__init__.py` 文件中都包含該模塊的簡要索引。

### 更新索引

當您進行以下操作時，應該更新索引：

- 添加新的類別或函數
- 移動或重命名現有類別或函數
- 修改類別或函數的功能或簡介文檔

更新索引可以通過以下命令完成：

```bash
# 更新主索引文件
python function_class_index_generator.py

# 更新模塊級索引
python update_module_index.py
```

### 索引文件內容

#### 主索引文件 (FUNCTION_CLASS_INDEX.md)

主索引文件包含專案中所有 Python 文件中定義的類別和函數，按照資料夾結構組織。每個類別和函數都包含：

- 名稱
- 繼承關係（對於類別）
- 簡短的文檔說明
- 所在文件

#### 模塊級索引 (各模塊的 __init__.py)

每個主要模塊的 `__init__.py` 文件包含該模塊內定義的所有類別和函數的摘要，以及它們的簡短說明和所在文件。此外，還有一個指向主索引文件的連結。

### 索引維護工具

專案包含兩個索引生成工具：

1. **function_class_index_generator.py**
   - 生成完整的函數與類別索引
   - 輸出為 Markdown 格式
   - 按照資料夾結構組織
   - 忽略 .gitignore 中指定的目錄和文件

2. **update_module_index.py**
   - 更新每個主要模塊的 `__init__.py` 文件的索引部分
   - 保留原有的導入語句和 `__all__` 定義
   - 添加模塊內所有類別和函數的摘要

## 程式碼風格與文檔標準

維護良好的文檔有助於其他開發者理解您的代碼。請遵循以下文檔標準：

1. **類別文檔**：
   ```python
   class MyClass:
       """
       簡短的類別描述。

       更詳細的類別描述，可以包含多行。

       Attributes:
           attribute1 (type): 屬性1的描述。
           attribute2 (type): 屬性2的描述。
       """
   ```

2. **函數文檔**：
   ```python
   def my_function(param1, param2):
       """
       簡短的函數描述。

       更詳細的函數描述，可以包含多行。

       Args:
           param1 (type): 參數1的描述。
           param2 (type): 參數2的描述。

       Returns:
           type: 返回值的描述。

       Raises:
           ExceptionType: 可能引發的異常及條件。
       """
   ```

遵循這些標準有助於自動化工具正確生成文檔和索引。 