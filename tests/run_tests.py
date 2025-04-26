#!/usr/bin/env python
"""
運行所有單元測試的腳本。
"""

import unittest
import sys
import os
from pathlib import Path

# 將項目根目錄添加到Python路徑
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# 創建測試目錄
test_data_dir = os.path.join(current_dir, 'test_data')
if not os.path.exists(test_data_dir):
    os.makedirs(test_data_dir)

if __name__ == '__main__':
    # 創建測試套件，僅包含已修復的核心模塊
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加已修復的測試模塊
    suite.addTest(loader.loadTestsFromName('test_utils'))
    suite.addTest(loader.loadTestsFromName('test_metrics'))
    suite.addTest(loader.loadTestsFromName('test_data_loader'))
    
    # 運行測試套件
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(suite)
    
    # 輸出結果摘要
    print("\n=== 測試結果摘要 ===")
    print(f"運行測試數: {result.testsRun}")
    print(f"失敗數: {len(result.failures)}")
    print(f"錯誤數: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n所有核心功能測試通過！")
    else:
        print("\n有測試未通過，請查看詳細輸出。")

    # 設置退出碼
    sys.exit(not result.wasSuccessful()) 