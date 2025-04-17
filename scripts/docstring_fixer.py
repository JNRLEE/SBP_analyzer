"""
# 修復 docstring 格式的腳本
# 自動為缺少標準格式 docstring 的模組和函數添加模板

此腳本根據 docstring_checker.py 的檢查結果，自動為缺少標準格式 docstring 的文件添加模板
"""

import os
import re
import sys
import json
import argparse
import ast
import astunparse
from pathlib import Path


def get_function_signature(file_path, line_number):
    """
    獲取函數簽名信息。
    
    Args:
        file_path (str): 文件路徑
        line_number (int): 函數定義所在的行號
        
    Returns:
        tuple: (函數名, 參數列表, 返回類型註解)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    try:
        module = ast.parse(source)
        
        # 尋找目標函數
        for node in ast.walk(module):
            if isinstance(node, ast.FunctionDef) and node.lineno == line_number:
                # 獲取函數名
                func_name = node.name
                
                # 獲取參數列表
                args = []
                for arg in node.args.args:
                    arg_name = arg.arg
                    arg_type = ""
                    if arg.annotation:
                        arg_type = " (" + astunparse.unparse(arg.annotation).strip() + ")"
                    args.append(arg_name + arg_type)
                
                # 獲取返回類型
                returns = ""
                if node.returns:
                    returns = astunparse.unparse(node.returns).strip()
                
                return func_name, args, returns
    except SyntaxError:
        pass
    
    # 如果 AST 解析失敗，嘗試直接解析文本
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if line_number <= len(lines):
        line = lines[line_number - 1].strip()
        match = re.match(r'def\s+(\w+)\s*\((.*?)\)\s*(?:->\s*([^:]+))?:', line)
        if match:
            func_name = match.group(1)
            args_str = match.group(2)
            returns = match.group(3) or ""
            
            # 解析參數
            args = []
            if args_str.strip():
                args = [arg.strip() for arg in args_str.split(',')]
            
            return func_name, args, returns
    
    return None, [], ""


def generate_docstring_for_function(func_name, args, returns):
    """
    為函數生成 docstring 模板。
    
    Args:
        func_name (str): 函數名
        args (list): 參數列表
        returns (str): 返回類型註解
        
    Returns:
        str: 生成的 docstring 模板
    """
    docstring = '"""\n'
    docstring += f'TODO: 描述 {func_name} 的功能。\n\n'
    
    if args:
        docstring += 'Args:\n'
        for arg in args:
            # 從參數中提取名稱和類型
            parts = arg.split(':')
            arg_name = parts[0].strip()
            docstring += f'    {arg_name}: TODO: 描述參數的用途\n'
        docstring += '\n'
    
    # 添加返回值部分
    docstring += 'Returns:\n'
    if returns:
        docstring += f'    {returns}: TODO: 描述返回值\n\n'
    else:
        docstring += '    TODO: 描述返回值類型: TODO: 描述返回值\n\n'
    
    # 添加可選的異常部分
    docstring += 'Raises:\n'
    docstring += '    TODO: 列出可能的異常\n'
    
    docstring += '"""\n'
    return docstring


def get_class_info(file_path, line_number):
    """
    獲取類信息。
    
    Args:
        file_path (str): 文件路徑
        line_number (int): 類定義所在的行號
        
    Returns:
        str: 類名
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if line_number <= len(lines):
        line = lines[line_number - 1].strip()
        match = re.match(r'class\s+(\w+)', line)
        if match:
            return match.group(1)
    
    return None


def generate_docstring_for_class(class_name):
    """
    為類生成 docstring 模板。
    
    Args:
        class_name (str): 類名
        
    Returns:
        str: 生成的 docstring 模板
    """
    docstring = '"""\n'
    docstring += f'{class_name} 類。\n\n'
    docstring += f'TODO: 描述 {class_name} 類的功能和用途。\n\n'
    docstring += 'Attributes:\n'
    docstring += '    TODO: 列出類的主要屬性\n'
    docstring += '"""\n'
    return docstring


def fix_docstring(file_path, missing_items, incomplete_items):
    """
    修復文件中的 docstring 問題。
    
    Args:
        file_path (str): 文件路徑
        missing_items (list): 缺少 docstring 的項目列表，每項為 (名稱, 行號)
        incomplete_items (list): 不完整 docstring 的項目列表，每項為 (名稱, 行號, 缺少的部分)
        
    Returns:
        bool: 是否成功修復
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 修復缺少的 docstring
    for item_name, line_no in missing_items:
        if " (class)" in item_name:
            # 處理類
            class_name = item_name.replace(" (class)", "")
            class_name = get_class_info(file_path, line_no)
            if class_name:
                # 獲取類定義行的縮進
                indent = re.match(r'^(\s*)', lines[line_no - 1]).group(1)
                # 找到類定義行後的位置
                insert_pos = line_no
                docstring = generate_docstring_for_class(class_name)
                # 加上縮進
                docstring = "\n".join(indent + "    " + line for line in docstring.split("\n"))
                # 插入 docstring
                lines.insert(insert_pos, docstring)
        else:
            # 處理函數
            func_name, args, returns = get_function_signature(file_path, line_no)
            if func_name:
                # 獲取函數定義行的縮進
                indent = re.match(r'^(\s*)', lines[line_no - 1]).group(1)
                # 找到函數定義行後的位置
                insert_pos = line_no
                docstring = generate_docstring_for_function(func_name, args, returns)
                # 加上縮進
                docstring = "\n".join(indent + "    " + line for line in docstring.split("\n"))
                # 插入 docstring
                lines.insert(insert_pos, docstring)
    
    # TODO: 修復不完整的 docstring（這部分比較複雜，需要解析現有 docstring 結構）
    
    # 寫回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    return True


def process_report(report_file):
    """
    處理 docstring_checker.py 生成的報告文件。
    
    Args:
        report_file (str): 報告文件路徑
        
    Returns:
        dict: 處理結果
    """
    with open(report_file, 'r', encoding='utf-8') as f:
        report_text = f.read()
    
    # 解析報告中的文件和問題
    files_with_issues = {}
    current_file = None
    
    for line in report_text.split('\n'):
        if line.startswith('### '):
            current_file = line[4:].strip()
            files_with_issues[current_file] = {'missing': [], 'incomplete': []}
        elif current_file and line.startswith('- ') and ' (行 ' in line:
            if '缺少 ' in line:
                # 處理不完整的 docstring
                match = re.match(r'- (.*?) \(行 (\d+)\): 缺少 (.*)', line)
                if match:
                    item_name = match.group(1)
                    line_no = int(match.group(2))
                    missing_sections = match.group(3).split(', ')
                    files_with_issues[current_file]['incomplete'].append((item_name, line_no, missing_sections))
            else:
                # 處理缺少的 docstring
                match = re.match(r'- (.*?) \(行 (\d+)\)', line)
                if match:
                    item_name = match.group(1)
                    line_no = int(match.group(2))
                    files_with_issues[current_file]['missing'].append((item_name, line_no))
    
    # 處理每個文件
    results = {}
    for file_path, issues in files_with_issues.items():
        if os.path.exists(file_path):
            try:
                success = fix_docstring(file_path, issues['missing'], issues['incomplete'])
                results[file_path] = {
                    'success': success,
                    'fixed_missing': len(issues['missing']),
                    'fixed_incomplete': 0  # 暫時不修復不完整的 docstring
                }
            except Exception as e:
                results[file_path] = {
                    'success': False,
                    'error': str(e)
                }
        else:
            results[file_path] = {
                'success': False,
                'error': 'File not found'
            }
    
    return results


def main():
    """
    主函數，解析命令行參數並執行修復。
    """
    parser = argparse.ArgumentParser(description='修復 Python 文件的 docstring 問題')
    parser.add_argument('report', help='docstring_checker.py 生成的報告文件路徑')
    parser.add_argument('--dry-run', '-d', action='store_true', help='僅顯示將要進行的修改，不實際修改文件')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("Dry run mode: 不會實際修改文件")
    
    results = process_report(args.report)
    
    # 輸出結果摘要
    total_files = len(results)
    successful_files = sum(1 for info in results.values() if info.get('success', False))
    total_fixed_missing = sum(info.get('fixed_missing', 0) for info in results.values())
    total_fixed_incomplete = sum(info.get('fixed_incomplete', 0) for info in results.values())
    
    print(f"處理了 {total_files} 個文件，成功修復了 {successful_files} 個文件")
    print(f"- 添加了 {total_fixed_missing} 個缺少的 docstring")
    print(f"- 修復了 {total_fixed_incomplete} 個不完整的 docstring")
    
    # 輸出詳細信息
    print("\n詳細結果:")
    for file_path, info in results.items():
        status = "成功" if info.get('success', False) else "失敗"
        details = ""
        if info.get('success', False):
            fixed_missing = info.get('fixed_missing', 0)
            fixed_incomplete = info.get('fixed_incomplete', 0)
            details = f"(添加了 {fixed_missing} 個 docstring, 修復了 {fixed_incomplete} 個不完整的 docstring)"
        else:
            details = f"錯誤: {info.get('error', '未知錯誤')}"
        
        print(f"{file_path}: {status} {details}")


if __name__ == '__main__':
    main() 