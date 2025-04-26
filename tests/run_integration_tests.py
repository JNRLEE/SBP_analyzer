#!/usr/bin/env python
"""
專門用於運行完整的端到端測試的腳本。
這個腳本解決了 SBP_analyzer 的 imports 相關技術問題，
特別是針對中間層數據分析與報告生成的整合測試。
"""

import os
import sys
import unittest
import pytest
from pathlib import Path

# 將項目根目錄添加到Python路徑
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# 確保測試數據目錄存在
test_data_dir = os.path.join(current_dir, 'test_data')
if not os.path.exists(test_data_dir):
    os.makedirs(test_data_dir)

test_output_dir = os.path.join(current_dir, 'test_output')
if not os.path.exists(test_output_dir):
    os.makedirs(test_output_dir)

def run_pytest():
    """運行pytest測試"""
    # 針對已修復的端到端測試
    test_modules = [
        "test_end_to_end.py::test_end_to_end_workflow",
        "test_integration.py::test_full_analysis_flow"
    ]
    
    args = ["-v"]  # 增加詳細輸出
    args.extend(test_modules)
    
    exit_code = pytest.main(args)
    return exit_code

def run_unittest():
    """運行unittest框架的測試"""
    # 創建測試套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加整合測試模塊
    suite.addTest(loader.loadTestsFromName('tests.test_end_to_end'))
    suite.addTest(loader.loadTestsFromName('tests.test_integration'))
    
    # 運行測試套件
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(suite)
    
    # 輸出結果摘要
    print("\n=== 端到端整合測試結果摘要 ===")
    print(f"運行測試數: {result.testsRun}")
    print(f"失敗數: {len(result.failures)}")
    print(f"錯誤數: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n所有端到端整合測試通過！")
    else:
        print("\n有測試未通過，請查看詳細輸出。")
    
    return not result.wasSuccessful()

if __name__ == '__main__':
    print("===== 運行 SBP_analyzer 端到端整合測試 =====")
    print(f"項目根目錄: {project_root}")
    
    # 根據命令行參數決定使用哪種測試框架
    if len(sys.argv) > 1 and sys.argv[1] == '--pytest':
        print("使用 pytest 運行測試...")
        exit_code = run_pytest()
    else:
        print("使用 unittest 運行測試...")
        exit_code = run_unittest()
    
    sys.exit(exit_code) 