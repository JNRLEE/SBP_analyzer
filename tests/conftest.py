"""
此配置檔案用於設置 pytest 環境，解決導入測試模組的問題。
"""

import os
import sys
from pathlib import Path

def pytest_sessionstart(session):
    """
    在測試會話開始前設置 Python 路徑。
    
    Args:
        session: pytest 會話物件
    
    Returns:
        無
    """
    # 獲取項目根目錄（測試目錄的父目錄）
    root_dir = Path(__file__).parent.parent.resolve()
    
    # 將項目根目錄添加到 Python 路徑
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
        print(f"\n已將項目根目錄添加到 Python 路徑: {root_dir}")
    
    # 驗證導入是否正常
    try:
        import analyzer
        print("✓ 成功導入 analyzer 模組")
    except ImportError as e:
        print(f"⚠️ 警告: 導入 analyzer 模組失敗: {e}")
        print(f"Python 路徑: {sys.path}") 