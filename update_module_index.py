#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
更新每個主要模塊的 __init__.py 文件，添加模塊內容索引
"""
import os
import ast
import glob
from typing import Dict, List, Tuple

# 主要模塊目錄
MAIN_MODULES = [
    "analyzer",
    "data_loader",
    "metrics", 
    "visualization",
    "reporter",
    "utils",
    "interfaces"
]

class ModuleIndexer:
    def __init__(self):
        self.module_contents = {}
        
    def analyze_module(self, module_path: str) -> Dict:
        """
        分析模塊目錄中的所有 .py 文件，提取類別和函數信息
        
        Args:
            module_path: 模塊路徑
            
        Returns:
            包含模塊內所有類別和函數的字典
        """
        module_name = os.path.basename(module_path)
        self.module_contents[module_name] = {"classes": [], "functions": []}
        
        # 獲取模塊中的所有 .py 文件
        py_files = glob.glob(os.path.join(module_path, "*.py"))
        
        for py_file in py_files:
            # 跳過 __init__.py
            if os.path.basename(py_file) == "__init__.py":
                continue
                
            file_name = os.path.basename(py_file)
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                try:
                    tree = ast.parse(source)
                    
                    # 直接獲取頂層定義
                    for node in tree.body:
                        if isinstance(node, ast.ClassDef):
                            doc = ast.get_docstring(node)
                            doc_summary = doc.split('.')[0] if doc else "無文檔字符串"
                            
                            self.module_contents[module_name]["classes"].append({
                                "name": node.name,
                                "file": file_name,
                                "doc": doc_summary
                            })
                        
                        # 獲取頂層函數
                        elif isinstance(node, ast.FunctionDef):
                            doc = ast.get_docstring(node)
                            doc_summary = doc.split('.')[0] if doc else "無文檔字符串"
                            
                            self.module_contents[module_name]["functions"].append({
                                "name": node.name,
                                "file": file_name,
                                "doc": doc_summary
                            })
                            
                except SyntaxError as e:
                    print(f"語法錯誤 {py_file}: {e}")
            except Exception as e:
                print(f"讀取文件錯誤 {py_file}: {e}")
        
        return self.module_contents[module_name]
        
    def generate_module_index(self, module_name: str) -> str:
        """
        為模塊生成索引文檔字符串
        
        Args:
            module_name: 模塊名稱
            
        Returns:
            模塊索引的文檔字符串
        """
        if module_name not in self.module_contents:
            return ""
            
        content = self.module_contents[module_name]
        doc_lines = [
            f"{module_name} 模組",
            "",
            f"此模組包含以下主要組件：",
            ""
        ]
        
        # 添加類別信息
        if content["classes"]:
            doc_lines.append("Classes:")
            for cls in sorted(content["classes"], key=lambda x: x["name"]):
                doc_lines.append(f"    {cls['name']}: {cls['doc']} ({cls['file']})")
            doc_lines.append("")
        
        # 添加函數信息
        if content["functions"]:
            doc_lines.append("Functions:")
            for func in sorted(content["functions"], key=lambda x: x["name"]):
                doc_lines.append(f"    {func['name']}: {func['doc']} ({func['file']})")
            doc_lines.append("")
        
        # 添加索引表連結
        doc_lines.extend([
            "",
            "完整的模組索引請參見：FUNCTION_CLASS_INDEX.md"
        ])
        
        return "\n".join(doc_lines)
        
    def update_init_file(self, module_path: str) -> None:
        """
        更新模塊的 __init__.py 文件，添加索引信息
        
        Args:
            module_path: 模塊路徑
        """
        module_name = os.path.basename(module_path)
        init_file = os.path.join(module_path, "__init__.py")
        
        if not os.path.exists(init_file):
            print(f"創建新的 __init__.py 文件: {init_file}")
            with open(init_file, 'w', encoding='utf-8') as f:
                f.write('"""\n')
                f.write(self.generate_module_index(module_name))
                f.write('\n"""\n\n')
                f.write(f'# 自動生成的 {module_name} 模組 __init__.py\n')
            return
            
        # 讀取現有內容
        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 解析 AST 以獲取導入語句和 __all__ 定義
        try:
            tree = ast.parse(content)
            imports = []
            all_list = None
            
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) or isinstance(node, ast.Import):
                    # 嘗試獲取原始源代碼，如果失敗，則使用手動構建
                    try:
                        imports.append(ast.get_source_segment(content, node))
                    except:
                        if isinstance(node, ast.ImportFrom):
                            imports.append(f"from {node.module} import {', '.join(n.name for n in node.names)}")
                        else:
                            imports.append(f"import {', '.join(n.name for n in node.names)}")
                            
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "__all__":
                            try:
                                all_list = ast.get_source_segment(content, node)
                            except:
                                # 如果無法獲取原始代碼，則跳過
                                pass
        except SyntaxError:
            imports = []
            all_list = None
            
        # 創建新的 __init__.py 內容
        new_content = ['"""']
        new_content.append(self.generate_module_index(module_name))
        new_content.append('"""')
        new_content.append('')
        
        # 添加導入語句
        if imports:
            new_content.extend(imports)
            new_content.append('')
            
        # 添加 __all__ 定義
        if all_list:
            new_content.append(all_list)
            
        # 寫入新文件
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_content))
            
        print(f"已更新 {init_file}")
        
def main():
    indexer = ModuleIndexer()
    
    for module in MAIN_MODULES:
        if os.path.exists(module) and os.path.isdir(module):
            print(f"處理模塊: {module}")
            indexer.analyze_module(module)
            indexer.update_init_file(module)
        else:
            print(f"找不到模塊目錄: {module}")
            
if __name__ == "__main__":
    main() 