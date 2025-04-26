#!/usr/bin/env python
"""
此程式用於修復 SBP_analyzer 專案的導入問題。
"""

import os
import sys
import re
import traceback
import importlib
from pathlib import Path

def create_init_files():
    """
    在項目目錄和子目錄中創建 __init__.py 文件。
    
    Args:
        無
    
    Returns:
        無
    """
    root_dir = Path(__file__).parent.resolve()
    
    # 要創建 __init__.py 文件的目錄列表
    directories = [
        root_dir,  # 項目根目錄
        root_dir / "analyzer",
        root_dir / "utils",
        root_dir / "metrics",
        root_dir / "tests",
        root_dir / "visualization",
        # 添加其他需要的目錄
    ]
    
    # 確保所有目錄存在並包含 __init__.py 文件
    for directory in directories:
        if directory.exists() and directory.is_dir():
            init_file = directory / "__init__.py"
            if not init_file.exists():
                with open(init_file, 'w') as f:
                    f.write(f"# {directory.name} 包的初始化文件\n")
                print(f"已創建: {init_file}")
            else:
                print(f"已存在: {init_file}")
        else:
            print(f"目錄不存在: {directory}, 跳過創建 __init__.py")

def fix_python_path():
    """
    將項目根目錄添加到 Python 路徑。
    
    Args:
        無
    
    Returns:
        無
    """
    root_dir = str(Path(__file__).parent.resolve())
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
        print(f"已將項目根目錄添加到 Python 路徑: {root_dir}")
    else:
        print(f"項目根目錄已在 Python 路徑中: {root_dir}")

def verify_imports():
    """
    驗證常用模塊的導入。
    
    Args:
        無
    
    Returns:
        bool: 所有模塊導入是否成功
    """
    modules_to_check = [
        "analyzer", 
        "utils", 
        "metrics", 
        "visualization"
    ]
    
    all_success = True
    
    for module_name in modules_to_check:
        try:
            importlib.import_module(module_name)
            print(f"✓ 成功導入 {module_name}")
        except ImportError as e:
            print(f"✗ 無法導入 {module_name}: {e}")
            all_success = False
            traceback.print_exc()
    
    return all_success

def fix_import_statements_in_file(file_path):
    """
    修復文件中的導入語句。
    
    Args:
        file_path: 要修復的文件路徑
        
    Returns:
        bool: 是否進行了更改
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 原始內容
    original_content = content
    
    # 修復從 sbp_analyzer 開頭的導入
    pattern = r'from\s+sbp_analyzer\.([a-zA-Z0-9_\.]+)\s+import'
    replacement = r'from \1 import'
    content = re.sub(pattern, replacement, content)
    
    # 修復 import X 形式的導入
    pattern = r'import\s+sbp_analyzer\.([a-zA-Z0-9_\.]+)'
    replacement = r'import \1'
    content = re.sub(pattern, replacement, content)
    
    # 將更改寫回文件
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        return True
    
    return False

def fix_import_statements():
    """
    遍歷項目中的所有 Python 文件並修復導入語句。
    
    Args:
        無
        
    Returns:
        無
    """
    root_dir = Path(__file__).parent.resolve()
    python_files = list(root_dir.glob('**/*.py'))
    
    modified_files = 0
    
    for file_path in python_files:
        # 跳過虛擬環境中的文件
        if any(venv_dir in str(file_path) for venv_dir in ['venv', 'env', '.venv', '.env']):
            continue
            
        try:
            if fix_import_statements_in_file(file_path):
                modified_files += 1
                print(f"已修復導入: {file_path}")
        except Exception as e:
            print(f"處理文件時出錯 {file_path}: {e}")
    
    print(f"\n總共修復了 {modified_files} 個文件中的導入語句")

def ensure_flat_structure():
    """
    確保項目使用扁平結構，沒有嵌套的 sbp_analyzer 目錄。
    如果發現嵌套目錄，會自動將其內容移動到項目根目錄。
    
    Args:
        無
        
    Returns:
        無
    """
    root_dir = Path(__file__).parent.resolve()
    nested_dir = root_dir / "sbp_analyzer"
    
    if nested_dir.exists() and nested_dir.is_dir():
        print(f"⚠️ 警告: 檢測到嵌套的 sbp_analyzer 目錄: {nested_dir}")
        print("正在將文件移動到項目根目錄...")
        
        # 移動所有文件到根目錄
        for item in nested_dir.iterdir():
            if item.is_file():
                target = root_dir / item.name
                if target.exists():
                    print(f"跳過已存在的文件: {target}")
                else:
                    item.rename(target)
                    print(f"已移動: {item} -> {target}")
            elif item.is_dir():
                target = root_dir / item.name
                if target.exists():
                    print(f"跳過已存在的目錄: {target}")
                else:
                    item.rename(target)
                    print(f"已移動目錄: {item} -> {target}")
        
        # 刪除空的 sbp_analyzer 目錄
        try:
            nested_dir.rmdir()
            print(f"已刪除空的 sbp_analyzer 目錄")
        except OSError as e:
            print(f"無法刪除 sbp_analyzer 目錄: {e}")
    else:
        print("✓ 項目結構正確，沒有嵌套的 sbp_analyzer 目錄")

def main():
    """
    執行所有修復步驟。
    
    Args:
        無
        
    Returns:
        無
    """
    print("=== SBP_analyzer 導入修復工具 ===\n")
    
    # 步驟 1: 創建必要的 __init__.py 文件
    print("\n步驟 1: 創建 __init__.py 文件")
    create_init_files()
    
    # 步驟 2: 修復 Python 路徑
    print("\n步驟 2: 修復 Python 路徑")
    fix_python_path()
    
    # 步驟 3: 確保扁平結構
    print("\n步驟 3: 確保扁平結構")
    ensure_flat_structure()
    
    # 步驟 4: 修復導入語句
    print("\n步驟 4: 修復導入語句")
    fix_import_statements()
    
    # 步驟 5: 驗證導入
    print("\n步驟 5: 驗證模塊導入")
    success = verify_imports()
    
    if success:
        print("\n✓ 所有修復完成，導入問題已解決!")
    else:
        print("\n⚠️ 修復過程完成，但仍有一些導入問題。請查看上面的錯誤信息。")

if __name__ == "__main__":
    main() 