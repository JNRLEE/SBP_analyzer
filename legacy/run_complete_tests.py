#!/usr/bin/env python
"""
運行完整端到端整合測試的腳本。
特別測試中間層數據分析與報告生成功能。
"""

import os
import sys
import pytest
from pathlib import Path

# 設置項目根目錄
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# 測試配置
TEST_MODULE = "test_complete_integration.py"

if __name__ == "__main__":
    print("===== 運行 SBP_analyzer 完整端到端整合測試 =====")
    print(f"測試模組: {TEST_MODULE}")
    
    # 運行測試
    exitcode = pytest.main(["-xvs", os.path.join(current_dir, TEST_MODULE)])
    
    # 輸出結果摘要
    if exitcode == 0:
        print("\n✅ 所有測試通過！中間層數據分析與報告生成功能已完成")
    else:
        print(f"\n❌ 測試失敗 (exit code: {exitcode})")
    
    sys.exit(exitcode) 