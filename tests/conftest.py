"""
測試配置文件
"""

import os
import pytest
import tempfile
import sys

# 將專案根目錄添加到 Python 路徑
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 共用測試夾具
@pytest.fixture
def temp_dir():
    """提供臨時目錄用於測試"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir 