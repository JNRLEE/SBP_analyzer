"""
# 檢查 docstring 格式的腳本
# 搜尋缺少標準格式 docstring 的模組和函數

此腳本會掃描專案中的所有 Python 文件，檢查是否包含標準格式的 docstring
(需包含 Args, Returns, Description 等標準部分)
"""

import os
import re
import sys
import argparse
from pathlib import Path

def check_docstring(file_path):
    """
    檢查文件中的函數是否都有標準格式的 docstring。
    
    Args:
        file_path (str): 要檢查的 Python 文件路徑
        
    Returns:
        dict: 包含文件檢查結果的字典，格式為:
            {'file': file_path,
             'missing_docstrings': [(函數名, 行號)],
             'incomplete_docstrings': [(函數名, 行號, 缺少的部分)]}
    """
    result = {
        'file': file_path,
        'missing_docstrings': [],
        'incomplete_docstrings': []
    }
    
    # 讀取文件內容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    # 查找所有函數和類定義
    func_pattern = re.compile(r'^\s*def\s+(\w+)\s*\(', re.MULTILINE)
    class_pattern = re.compile(r'^\s*class\s+(\w+)', re.MULTILINE)
    
    # 查找函數定義並檢查其 docstring
    for match in func_pattern.finditer(content):
        func_name = match.group(1)
        if func_name.startswith('_') and not func_name.startswith('__'):
            continue  # 忽略私有方法
        
        # 獲取函數定義的行號
        line_no = content[:match.start()].count('\n') + 1
        
        # 檢查 docstring 是否存在
        docstring_start = content[match.end():].strip().startswith('"""') or content[match.end():].strip().startswith("'''")
        if not docstring_start:
            result['missing_docstrings'].append((func_name, line_no))
            continue
        
        # 獲取函數的 docstring 內容
        docstring_content = extract_docstring(content[match.end():])
        
        # 檢查 docstring 是否包含必需的部分
        required_sections = ['Args', 'Returns']
        missing_sections = []
        
        for section in required_sections:
            pattern = re.compile(rf'{section}:', re.MULTILINE)
            if not pattern.search(docstring_content):
                missing_sections.append(section)
        
        if missing_sections:
            result['incomplete_docstrings'].append((func_name, line_no, missing_sections))
    
    # 檢查類定義及其 docstring
    for match in class_pattern.finditer(content):
        class_name = match.group(1)
        if class_name.startswith('_'):
            continue  # 忽略私有類
        
        # 獲取類定義的行號
        line_no = content[:match.start()].count('\n') + 1
        
        # 檢查 docstring 是否存在
        docstring_start = content[match.end():].strip().startswith('"""') or content[match.end():].strip().startswith("'''")
        if not docstring_start:
            result['missing_docstrings'].append((class_name + " (class)", line_no))
            continue
        
        # 獲取類的 docstring 內容
        docstring_content = extract_docstring(content[match.end():])
        
        # 類的 docstring 通常不需要 Args 和 Returns，但應該有描述
        if not docstring_content.strip():
            result['incomplete_docstrings'].append((class_name + " (class)", line_no, ['Description']))
    
    return result


def extract_docstring(text):
    """
    提取文本中的 docstring 內容。
    
    Args:
        text (str): 要提取 docstring 的文本
        
    Returns:
        str: 提取的 docstring 內容，如果沒有找到則返回空字符串
    """
    # 查找 triple-quoted 字符串 (''' 或 """)
    triple_quotes = ["'''", '"""']
    start_idx = -1
    end_idx = -1
    triple_quote_type = None
    
    for quote in triple_quotes:
        if text.strip().startswith(quote):
            start_idx = text.strip().find(quote)
            triple_quote_type = quote
            break
    
    if start_idx == -1:
        return ""
    
    # 計算真實的起始位置（考慮開頭的空白）
    real_start_idx = text.find(text.strip())
    start_idx = real_start_idx + start_idx + len(triple_quote_type)
    
    # 查找結束的 triple quote
    end_idx = text.find(triple_quote_type, start_idx)
    if end_idx == -1:
        return ""
    
    return text[start_idx:end_idx].strip()


def scan_directory(directory, exclude_dirs=None):
    """
    遞歸掃描目錄，檢查所有 Python 文件。
    
    Args:
        directory (str): 要掃描的目錄路徑
        exclude_dirs (list): 要排除的目錄列表
        
    Returns:
        list: 包含所有 Python 文件檢查結果的列表
    """
    if exclude_dirs is None:
        exclude_dirs = []
    
    results = []
    
    for root, dirs, files in os.walk(directory):
        # 排除指定的目錄
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.') and not d == '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    result = check_docstring(file_path)
                    if result['missing_docstrings'] or result['incomplete_docstrings']:
                        results.append(result)
                except Exception as e:
                    print(f"錯誤：處理文件 {file_path} 時出現問題: {e}")
    
    return results


def generate_report(results, output_file=None):
    """
    生成檢查結果報告。
    
    Args:
        results (list): 檢查結果列表
        output_file (str, optional): 輸出文件路徑，如果提供則將報告寫入文件
        
    Returns:
        str: 報告內容
    """
    report = "# Docstring 格式檢查報告\n\n"
    
    total_missing = sum(len(r['missing_docstrings']) for r in results)
    total_incomplete = sum(len(r['incomplete_docstrings']) for r in results)
    
    report += f"共檢查了 {len(results)} 個文件，發現 {total_missing} 處缺少 docstring，{total_incomplete} 處 docstring 格式不完整。\n\n"
    
    if results:
        report += "## 詳細結果\n\n"
        
        for result in results:
            file_path = result['file']
            missing = result['missing_docstrings']
            incomplete = result['incomplete_docstrings']
            
            report += f"### {file_path}\n\n"
            
            if missing:
                report += "#### 缺少 Docstring\n\n"
                for func_name, line_no in missing:
                    report += f"- {func_name} (行 {line_no})\n"
                report += "\n"
            
            if incomplete:
                report += "#### Docstring 格式不完整\n\n"
                for func_name, line_no, missing_sections in incomplete:
                    report += f"- {func_name} (行 {line_no}): 缺少 {', '.join(missing_sections)}\n"
                report += "\n"
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
    
    return report


def generate_docstring_template():
    """
    生成標準 docstring 模板。
    
    Returns:
        str: docstring 模板
    """
    template = '"""\n'
    template += '函數或方法的簡要描述。\n\n'
    template += '詳細描述函數或方法的功能、算法、使用場景等。\n\n'
    template += 'Args:\n'
    template += '    param1 (type): 參數1的描述\n'
    template += '    param2 (type): 參數2的描述\n\n'
    template += 'Returns:\n'
    template += '    type: 返回值的描述\n\n'
    template += 'Raises:\n'
    template += '    ExceptionType: 異常情況的描述\n\n'
    template += 'Examples:\n'
    template += '    >>> example_function(1, 2)\n'
    template += '    3\n'
    template += '"""\n'
    
    return template


def main():
    """
    主函數，解析命令行參數並執行檢查。
    """
    parser = argparse.ArgumentParser(description='檢查 Python 文件的 docstring 格式')
    parser.add_argument('directory', help='要掃描的目錄路徑')
    parser.add_argument('--exclude', '-e', nargs='+', default=['venv', 'env', '.git', 'build', 'dist'],
                        help='要排除的目錄')
    parser.add_argument('--output', '-o', help='輸出報告文件的路徑')
    parser.add_argument('--template', '-t', action='store_true', help='顯示標準 docstring 模板')
    
    args = parser.parse_args()
    
    if args.template:
        print(generate_docstring_template())
        return
    
    results = scan_directory(args.directory, args.exclude)
    report = generate_report(results, args.output)
    
    print(report)


if __name__ == '__main__':
    main() 