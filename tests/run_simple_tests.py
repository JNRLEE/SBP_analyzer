#!/usr/bin/env python
"""
一個簡單的測試運行腳本，測試基本功能。
"""

import os
import sys

# 將當前目錄添加到Python搜索路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print(f"Python 搜索路徑: {sys.path}")

# 運行一個簡單的測試
def test_basic():
    # 測試目錄創建和刪除
    test_dir = 'test_dir'
    
    # 確保目錄存在
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    print(f"創建目錄: {test_dir}")
    assert os.path.exists(test_dir), "目錄應該被創建"
    
    # 創建一些測試文件
    test_files = ['test1.txt', 'test2.txt', 'test.json']
    for file in test_files:
        path = os.path.join(test_dir, file)
        with open(path, 'w') as f:
            f.write('test')
        print(f"創建文件: {path}")
    
    # 列出目錄中的文件
    files = os.listdir(test_dir)
    print(f"目錄中的文件: {files}")
    
    # 清理測試文件和目錄
    for file in test_files:
        path = os.path.join(test_dir, file)
        os.remove(path)
        print(f"刪除文件: {path}")
    
    os.rmdir(test_dir)
    print(f"刪除目錄: {test_dir}")
    
    print("基本功能測試成功!")
    return True

if __name__ == '__main__':
    test_basic()
    print("測試完成!") 