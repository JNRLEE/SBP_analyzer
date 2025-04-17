#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
檢查專案中所有 Python 文件的 Docstring 格式是否符合 NumPy 風格。
此腳本將遍歷指定目錄中的所有 Python 文件，檢查其 Docstring 格式，
並報告發現的問題。
"""

import os
import re
import ast
import sys
import argparse
from typing import List, Dict, Tuple, Optional


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


def parse_docstring(docstring: Optional[str]) -> Dict:
    """解析文檔字符串並檢查其格式。
    
    參數
    ----------
    docstring : Optional[str]
        要解析的文檔字符串
    
    返回值
    -------
    Dict
        文檔字符串各部分的解析結果和問題
    """
    result = {
        'has_docstring': docstring is not None,
        'has_summary': False,
        'has_parameters': False,
        'has_returns': False,
        'has_examples': False,
        'issues': []
    }
    
    if not docstring:
        result['issues'].append("缺少文檔字符串")
        return result
    
    # 清理文檔字符串
    docstring = docstring.strip()
    
    # 檢查摘要行
    lines = docstring.split('\n')
    if not lines:
        result['issues'].append("文檔字符串為空")
        return result
    
    # 檢查摘要行
    summary = lines[0].strip()
    if not summary or len(summary) < 10:
        result['issues'].append("摘要行過短或不存在")
    else:
        result['has_summary'] = True
    
    # 檢查參數部分
    if "參數" in docstring or "Parameters" in docstring:
        result['has_parameters'] = True
        
        # 檢查參數格式
        param_pattern = re.compile(r'(\w+)\s*:\s*(\w+|\w+\[.+\])')
        param_matches = param_pattern.findall(docstring)
        if not param_matches:
            result['issues'].append("參數部分格式不正確")
    
    # 檢查返回值部分
    if "返回值" in docstring or "Returns" in docstring:
        result['has_returns'] = True
    
    # 檢查範例部分
    if "範例" in docstring or "Examples" in docstring:
        result['has_examples'] = True
    
    return result


def check_file_docstrings(file_path: str) -> Tuple[int, List[str]]:
    """檢查單個文件中的所有 Docstring。
    
    參數
    ----------
    file_path : str
        Python 文件路徑
    
    返回值
    -------
    Tuple[int, List[str]]
        (問題數量, 問題詳細列表)
    """
    issues = []
    total_issues = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        tree = ast.parse(file_content)
        
        # 檢查模塊級文檔字符串
        if not ast.get_docstring(tree):
            issues.append(f"模塊缺少文檔字符串")
            total_issues += 1
        
        # 檢查函數和類的文檔字符串
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node)
                result = parse_docstring(docstring)
                
                if not result['has_docstring']:
                    issues.append(f"{node.name} 缺少文檔字符串")
                    total_issues += 1
                elif result['issues']:
                    for issue in result['issues']:
                        issues.append(f"{node.name}: {issue}")
                        total_issues += 1
                
                # 檢查公共方法是否有合適的文檔字符串
                if isinstance(node, ast.ClassDef):
                    for method in node.body:
                        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if not method.name.startswith('_') or method.name in ('__init__', '__call__'):
                                method_docstring = ast.get_docstring(method)
                                method_result = parse_docstring(method_docstring)
                                
                                if not method_result['has_docstring']:
                                    issues.append(f"{node.name}.{method.name} 缺少文檔字符串")
                                    total_issues += 1
                                elif method_result['issues']:
                                    for issue in method_result['issues']:
                                        issues.append(f"{node.name}.{method.name}: {issue}")
                                        total_issues += 1
        
        return total_issues, issues
    
    except SyntaxError:
        return 1, [f"無法解析文件（語法錯誤）"]
    except Exception as e:
        return 1, [f"檢查時發生錯誤: {str(e)}"]


def main():
    """主函數，解析命令列參數並執行檢查。"""
    parser = argparse.ArgumentParser(description='檢查 Python 文件的 Docstring 格式')
    parser.add_argument('--dir', type=str, default='.', help='要檢查的目錄')
    parser.add_argument('--exclude', type=str, nargs='*', default=['venv', '.git', '__pycache__', 'build', 'dist'],
                        help='要排除的目錄')
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
    
    total_issues = 0
    files_with_issues = 0
    
    # 檢查每個文件
    for file_path in filtered_files:
        issues_count, issues = check_file_docstrings(file_path)
        
        if issues_count > 0:
            files_with_issues += 1
            total_issues += issues_count
            print(f"\n檢查文件: {file_path}")
            for issue in issues:
                print(f"  - {issue}")
    
    # 輸出總結
    print(f"\n總結:")
    print(f"檢查了 {len(filtered_files)} 個文件")
    print(f"發現 {files_with_issues} 個文件中有 {total_issues} 個問題")
    
    if total_issues > 0:
        sys.exit(1)
    else:
        print("所有文件都符合 Docstring 風格指南！")
        sys.exit(0)


if __name__ == '__main__':
    main() 