#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
為專案中的 Python 文件自動生成或修復 NumPy 風格的 Docstring。
此腳本將遍歷指定目錄中的所有 Python 文件，找出缺少 Docstring 的函數和類，
並自動添加符合 NumPy 風格的 Docstring 模板。
"""

import os
import ast
import sys
import re
import argparse
from typing import List, Dict, Optional, Any, Set, Tuple


def get_python_files(directory: str) -> List[str]:
    """獲取指定目錄下所有的 Python 文件。
    
    參數
    ----------
    directory : str
        要搜索的目錄路徑
    
    返回值
    -------
    List[str]
        Python 文件路徑列表
    """
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files


def extract_parameters(func_node: ast.FunctionDef) -> List[Dict[str, str]]:
    """從函數定義節點中提取參數信息。
    
    參數
    ----------
    func_node : ast.FunctionDef
        AST 函數定義節點
    
    返回值
    -------
    List[Dict[str, str]]
        參數列表，每個參數都是一個字典，包含名稱和類型
    """
    parameters = []
    
    # 處理參數
    for arg in func_node.args.args:
        param_name = arg.arg
        
        # 跳過 self 參數
        if param_name == 'self' and parameters == []:
            continue
            
        param_type = 'Any'  # 默認類型
        
        # 從註釋中獲取類型
        if arg.annotation:
            if isinstance(arg.annotation, ast.Name):
                param_type = arg.annotation.id
            elif isinstance(arg.annotation, ast.Subscript):
                # 處理泛型類型，如 List[str]
                value_id = ""
                if isinstance(arg.annotation.value, ast.Name):
                    value_id = arg.annotation.value.id
                elif isinstance(arg.annotation.value, ast.Attribute):
                    value_id = arg.annotation.value.attr
                
                # 提取索引
                index_id = ""
                slice_node = arg.annotation.slice
                
                if hasattr(slice_node, 'value') and isinstance(slice_node.value, ast.Name):
                    index_id = slice_node.value.id
                elif isinstance(slice_node, ast.Name):
                    index_id = slice_node.id
                elif isinstance(slice_node, ast.Tuple):
                    index_parts = []
                    for elt in slice_node.elts:
                        if isinstance(elt, ast.Name):
                            index_parts.append(elt.id)
                    index_id = ", ".join(index_parts)
                
                param_type = f"{value_id}[{index_id}]"
            elif isinstance(arg.annotation, ast.Attribute):
                # 處理 module.Type 類型
                param_type = arg.annotation.attr
        
        parameters.append({
            'name': param_name,
            'type': param_type
        })
    
    return parameters


def extract_return_type(func_node: ast.FunctionDef) -> Optional[str]:
    """從函數定義節點中提取返回類型。
    
    參數
    ----------
    func_node : ast.FunctionDef
        AST 函數定義節點
    
    返回值
    -------
    Optional[str]
        返回類型，如果沒有註釋則為 None
    """
    if func_node.returns:
        if isinstance(func_node.returns, ast.Name):
            return func_node.returns.id
        elif isinstance(func_node.returns, ast.Subscript):
            # 處理泛型類型，如 List[str]
            value_id = ""
            if isinstance(func_node.returns.value, ast.Name):
                value_id = func_node.returns.value.id
            elif isinstance(func_node.returns.value, ast.Attribute):
                value_id = func_node.returns.value.attr
            
            # 提取索引
            index_id = ""
            slice_node = func_node.returns.slice
            
            if hasattr(slice_node, 'value') and isinstance(slice_node.value, ast.Name):
                index_id = slice_node.value.id
            elif isinstance(slice_node, ast.Name):
                index_id = slice_node.id
            elif isinstance(slice_node, ast.Tuple):
                index_parts = []
                for elt in slice_node.elts:
                    if isinstance(elt, ast.Name):
                        index_parts.append(elt.id)
                index_id = ", ".join(index_parts)
            
            return f"{value_id}[{index_id}]"
        elif isinstance(func_node.returns, ast.Attribute):
            # 處理 module.Type 類型
            return func_node.returns.attr
        elif isinstance(func_node.returns, ast.Constant) and func_node.returns.value is None:
            return "None"
    
    return None


def generate_docstring_template(node: ast.AST) -> str:
    """為給定的 AST 節點生成 Docstring 模板。
    
    參數
    ----------
    node : ast.AST
        要生成 Docstring 的 AST 節點
    
    返回值
    -------
    str
        生成的 Docstring 模板
    """
    if isinstance(node, ast.FunctionDef):
        # 提取函數名以生成描述
        func_name = node.name
        description = f"{''.join(' ' + word if i > 0 else word for i, word in enumerate(re.findall(r'[A-Z][a-z0-9]*|[a-z0-9]+', func_name)))}"
        
        # 獲取參數
        parameters = extract_parameters(node)
        
        # 獲取返回類型
        return_type = extract_return_type(node)
        
        # 構建 Docstring
        docstring = f"{description}.\n\n"
        
        # 添加參數部分
        if parameters:
            docstring += "參數\n----------\n"
            for param in parameters:
                docstring += f"{param['name']} : {param['type']}\n    描述\n"
            docstring += "\n"
        
        # 添加返回值部分
        if return_type and return_type != "None":
            docstring += "返回值\n-------\n"
            docstring += f"{return_type}\n    描述\n\n"
        
        # 添加範例部分
        docstring += "範例\n--------\n>>> # 示例代碼\n"
        
        return docstring
    
    elif isinstance(node, ast.ClassDef):
        # 提取類名以生成描述
        class_name = node.name
        description = f"{''.join(' ' + word if i > 0 else word for i, word in enumerate(re.findall(r'[A-Z][a-z0-9]*|[a-z0-9]+', class_name)))}"
        
        # 構建 Docstring
        docstring = f"{description}.\n\n"
        
        # 添加屬性部分
        docstring += "屬性\n----------\n"
        docstring += "attribute_name : type\n    描述\n\n"
        
        # 添加方法部分
        docstring += "方法\n-------\n"
        docstring += "method_name(param1, param2, ...)\n    描述\n\n"
        
        # 添加範例部分
        docstring += "範例\n--------\n>>> # 示例代碼\n"
        
        return docstring
    
    else:
        # 模塊級 Docstring
        return "模塊描述.\n\n"


def fix_file_docstrings(file_path: str, dry_run: bool = False) -> Tuple[int, List[str]]:
    """修復文件中缺少的 Docstring。
    
    參數
    ----------
    file_path : str
        Python 文件路徑
    dry_run : bool, optional
        如果為 True，僅顯示將進行的更改而不實際修改文件，默認為 False
    
    返回值
    -------
    Tuple[int, List[str]]
        (修復的 Docstring 數量, 修復詳細列表)
    """
    fixes = []
    total_fixes = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        tree = ast.parse(file_content)
        
        # 檢查模塊級文檔字符串
        module_docstring = ast.get_docstring(tree)
        if not module_docstring:
            # 生成模塊級 Docstring
            new_docstring = generate_docstring_template(tree)
            fixes.append(f"添加模塊級 Docstring")
            total_fixes += 1
            
            if not dry_run:
                # 在文件開頭添加 Docstring
                lines = file_content.split('\n')
                # 跳過可能的 shebang 和編碼聲明
                insert_line = 0
                while insert_line < len(lines) and (lines[insert_line].startswith('#!') or lines[insert_line].startswith('# -*-')):
                    insert_line += 1
                
                # 插入 Docstring
                docstring_lines = [f'"""', *new_docstring.split('\n'), f'"""', '']
                lines[insert_line:insert_line] = docstring_lines
                
                # 更新文件內容
                file_content = '\n'.join(lines)
        
        # 對函數和類進行遍歷
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                # 獲取現有的 Docstring
                existing_docstring = ast.get_docstring(node)
                
                if not existing_docstring:
                    # 生成 Docstring 模板
                    new_docstring = generate_docstring_template(node)
                    fixes.append(f"為 {node.name} 添加 Docstring")
                    total_fixes += 1
                    
                    if not dry_run:
                        # 找出函數/類的起始位置
                        start_line = node.lineno
                        col_offset = node.col_offset
                        
                        # 獲取函數/類的第一行
                        lines = file_content.split('\n')
                        indent = ' ' * col_offset
                        
                        # 找到函數/類體的第一行
                        body_start = start_line
                        for lineno, line in enumerate(lines[start_line-1:], start_line):
                            if ':' in line:
                                body_start = lineno + 1
                                break
                        
                        # 插入 Docstring
                        docstring_lines = [f'{indent}"""', *[f'{indent}{line}' for line in new_docstring.split('\n')], f'{indent}"""']
                        lines[body_start-1:body_start-1] = docstring_lines
                        
                        # 更新文件內容
                        file_content = '\n'.join(lines)
        
        # 寫回文件
        if not dry_run and total_fixes > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
        
        return total_fixes, fixes
    
    except SyntaxError:
        return 0, [f"無法解析文件（語法錯誤）"]
    except Exception as e:
        return 0, [f"修復時發生錯誤: {str(e)}"]


def main():
    """主函數，解析命令列參數並執行 Docstring 修復。"""
    parser = argparse.ArgumentParser(description='修復 Python 文件的 Docstring 格式')
    parser.add_argument('--dir', type=str, default='.', help='要修復的目錄')
    parser.add_argument('--exclude', type=str, nargs='*', default=['venv', '.git', '__pycache__', 'build', 'dist'],
                        help='要排除的目錄')
    parser.add_argument('--dry-run', action='store_true', help='僅顯示將進行的更改，不實際修改文件')
    args = parser.parse_args()
    
    # 獲取所有 Python 文件
    python_files = get_python_files(args.dir)
    
    # 過濾排除的目錄
    filtered_files = []
    for file_path in python_files:
        exclude = False
        for excluded_dir in args.exclude:
            if excluded_dir in file_path.split(os.sep):
                exclude = True
                break
        if not exclude:
            filtered_files.append(file_path)
    
    total_fixes = 0
    files_fixed = 0
    
    # 修復每個文件
    for file_path in filtered_files:
        fixes_count, fixes = fix_file_docstrings(file_path, args.dry_run)
        
        if fixes_count > 0:
            files_fixed += 1
            total_fixes += fixes_count
            print(f"\n修復文件: {file_path}")
            for fix in fixes:
                print(f"  - {fix}")
    
    # 輸出總結
    print(f"\n總結:")
    print(f"檢查了 {len(filtered_files)} 個文件")
    
    if args.dry_run:
        print(f"將修復 {files_fixed} 個文件中的 {total_fixes} 個 Docstring（乾跑模式）")
    else:
        print(f"修復了 {files_fixed} 個文件中的 {total_fixes} 個 Docstring")
    
    sys.exit(0)


if __name__ == '__main__':
    main() 