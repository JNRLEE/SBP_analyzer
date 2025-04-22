#!/usr/bin/env python
"""
專門運行學習率調度影響分析測試的腳本。
這個腳本會運行 tests/test_lr_schedule_impact.py 中的測試，並將測試結果輸出到控制台。
"""

import unittest
import sys
from pathlib import Path

# 將項目根目錄添加到Python路徑
current_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(current_dir))

# 導入測試模塊
from tests.test_lr_schedule_impact import TestLRScheduleImpact

if __name__ == '__main__':
    # 創建測試套件
    suite = unittest.TestSuite()
    
    # 添加所有學習率調度影響相關的測試方法
    for test_method in [
        'test_analyze_lr_schedule_detection',
        'test_analyze_lr_correlation',
        'test_compare_performance',
        'test_plot_lr_schedule_impact',
        'test_with_missing_data',
        'test_recommendations_generation',
        'test_plot_with_english_labels'
    ]:
        suite.addTest(TestLRScheduleImpact(test_method))
    
    # 運行測試並收集結果
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(suite)
    
    # 輸出結果摘要
    print("\n=== 學習率調度影響分析測試結果摘要 ===")
    print(f"運行測試數: {result.testsRun}")
    print(f"失敗數: {len(result.failures)}")
    print(f"錯誤數: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n所有測試通過！")
        print("學習率調度影響分析功能正常工作。")
    else:
        print("\n有測試未通過:")
        for failure in result.failures:
            print(f"\n失敗: {failure[0]}")
            print(f"{failure[1]}")
        for error in result.errors:
            print(f"\n錯誤: {error[0]}")
            print(f"{error[1]}")
    
    # 設置退出碼
    sys.exit(not result.wasSuccessful()) 