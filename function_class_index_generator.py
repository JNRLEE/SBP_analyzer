#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成整個專案的函數與類別索引表，按照資料夾樹狀結構排列
"""
import os
import ast
import json
from typing import Dict, List, Optional, Set, Tuple, Union

# 需要忽略的目錄，與.gitignore相符
IGNORED_DIRS = [
    "__pycache__", 
    "venv", 
    "env", 
    "ENV", 
    "dist", 
    "build", 
    ".idea", 
    ".vscode", 
    ".ipynb_checkpoints", 
    "tensorboard_logs",
    "sbp_analyzer.egg-info",
    ".git",
    ".pytest_cache",
    ".cursor"
]

# 需要忽略的檔案類型
IGNORED_FILES = [
    ".pyc", 
    ".pyo", 
    ".pyd", 
    ".DS_Store", 
    "Thumbs.db", 
    ".coverage"
]

class FunctionClassVisitor(ast.NodeVisitor):
    """
    訪問 Python 檔案的抽象語法樹，收集所有函數和類別的定義
    """
    def __init__(self):
        self.functions = []
        self.classes = []
        self.current_class = None
        self.class_methods = {}
        
    def visit_FunctionDef(self, node):
        """訪問函數定義節點"""
        # 如果在類別內部，則這是一個方法
        if self.current_class:
            if node.name not in ["__init__", "__str__", "__repr__"]:  # 可以根據需要調整要忽略的方法
                self.class_methods[self.current_class].append({
                    "name": node.name,
                    "line": node.lineno,
                    "doc": ast.get_docstring(node)
                })
        else:
            # 這是一個普通函數
            self.functions.append({
                "name": node.name,
                "line": node.lineno,
                "doc": ast.get_docstring(node)
            })
        
        # 繼續訪問子節點
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """訪問類別定義節點"""
        self.classes.append({
            "name": node.name,
            "line": node.lineno,
            "doc": ast.get_docstring(node),
            "bases": [getattr(base, 'id', str(base)) for base in node.bases if hasattr(base, 'id')]
        })
        
        # 暫存當前類別名稱
        old_class = self.current_class
        self.current_class = node.name
        if self.current_class not in self.class_methods:
            self.class_methods[self.current_class] = []
        
        # 訪問類別內的所有節點
        self.generic_visit(node)
        
        # 復原當前類別名稱
        self.current_class = old_class


def analyze_file(file_path: str) -> Dict:
    """
    分析單個 Python 檔案，提取所有函數和類別
    
    Args:
        file_path: Python 檔案的路徑
        
    Returns:
        包含檔案中所有函數和類別的字典
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        try:
            tree = ast.parse(source)
            visitor = FunctionClassVisitor()
            visitor.visit(tree)
            
            return {
                "functions": visitor.functions,
                "classes": visitor.classes,
                "class_methods": visitor.class_methods
            }
        except SyntaxError as e:
            return {
                "error": f"SyntaxError: {str(e)}",
                "functions": [],
                "classes": []
            }
    except Exception as e:
        return {
            "error": f"Error reading file: {str(e)}",
            "functions": [],
            "classes": []
        }


def analyze_directory(root_dir: str) -> Dict:
    """
    遞迴分析目錄中的所有 Python 檔案
    
    Args:
        root_dir: 根目錄路徑
        
    Returns:
        包含所有檔案分析結果的嵌套字典
    """
    result = {}
    
    for root, dirs, files in os.walk(root_dir):
        # 移除需要忽略的目錄
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        
        rel_path = os.path.relpath(root, root_dir)
        if rel_path == ".":
            current_dict = result
        else:
            # 創建嵌套字典路徑
            current_dict = result
            for part in rel_path.split(os.sep):
                if part not in current_dict:
                    current_dict[part] = {}
                current_dict = current_dict[part]
        
        # 分析所有 Python 檔案
        for file in files:
            if file.endswith('.py') and not any(file.endswith(ext) for ext in IGNORED_FILES):
                file_path = os.path.join(root, file)
                current_dict[file] = analyze_file(file_path)
    
    return result


def generate_markdown_index(analysis_result: Dict, base_path: str = "") -> str:
    """
    從分析結果生成 Markdown 格式的索引表
    
    Args:
        analysis_result: 分析結果字典
        base_path: 當前在索引表中的基礎路徑
        
    Returns:
        Markdown 格式的索引表
    """
    markdown = []
    
    # 處理檔案
    for name, content in sorted(analysis_result.items()):
        if name.endswith('.py'):
            if "error" in content:
                markdown.append(f"- **{base_path + name}**: {content['error']}")
                continue
                
            # 添加檔案標題
            file_path = base_path + name
            markdown.append(f"### {file_path}")
            
            # 添加類別
            if content["classes"]:
                markdown.append("\n**Classes:**\n")
                for cls in content["classes"]:
                    bases = f" (extends {', '.join(cls['bases'])})" if cls['bases'] else ""
                    doc_summary = f": {cls['doc'].split('.')[0]}..." if cls['doc'] else ""
                    markdown.append(f"- `{cls['name']}`{bases}{doc_summary}")
                    
                    # 添加類別方法
                    if cls["name"] in content.get("class_methods", {}):
                        methods = content["class_methods"][cls["name"]]
                        if methods:
                            for method in methods:
                                doc_summary = f": {method['doc'].split('.')[0]}..." if method['doc'] else ""
                                markdown.append(f"  - `{method['name']}()`{doc_summary}")
            
            # 添加函數
            if content["functions"]:
                markdown.append("\n**Functions:**\n")
                for func in content["functions"]:
                    doc_summary = f": {func['doc'].split('.')[0]}..." if func['doc'] else ""
                    markdown.append(f"- `{func['name']}()`{doc_summary}")
            
            markdown.append("\n")
    
    # 處理子目錄
    for name, subcontent in sorted(analysis_result.items()):
        if not name.endswith('.py'):
            markdown.append(f"## {base_path + name}/\n")
            subindex = generate_markdown_index(subcontent, base_path + name + "/")
            markdown.append(subindex)
    
    return "\n".join(markdown)


def create_index_file(root_dir: str, output_file: str = "FUNCTION_CLASS_INDEX.md") -> None:
    """
    創建索引文件
    
    Args:
        root_dir: 根目錄路徑
        output_file: 輸出文件的路徑
    """
    analysis = analyze_directory(root_dir)
    markdown = "# SBP Analyzer 函數與類別索引\n\n"
    markdown += "此文件自動生成，包含專案中所有函數和類別的索引。\n\n"
    markdown += "## 目錄結構\n\n"
    markdown += generate_markdown_index(analysis)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    print(f"索引已寫入 {output_file}")


if __name__ == "__main__":
    create_index_file(".") 