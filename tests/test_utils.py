"""
# 工具模組測試
# 測試文件和張量工具模組的功能

測試日期: 2024-05-28
測試目的: 驗證文件和張量工具模組的基本功能
"""

import os
import pytest
import tempfile
import torch
import numpy as np
import json
from pathlib import Path

from utils import file_utils, tensor_utils


@pytest.fixture
def temp_test_dir():
    """
    創建臨時測試目錄結構和文件。
    
    Returns:
        str: 臨時測試目錄的路徑
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # 創建必要的子目錄
        sub_dir = os.path.join(temp_dir, "sub_dir")
        os.makedirs(sub_dir, exist_ok=True)
        
        # 創建測試文件
        test_json = {"key": "value", "numbers": [1, 2, 3]}
        with open(os.path.join(temp_dir, "test.json"), "w") as f:
            json.dump(test_json, f)
            
        # 創建測試張量文件
        test_tensor = torch.randn(5, 10)
        torch.save(test_tensor, os.path.join(temp_dir, "tensor.pt"))
        
        yield temp_dir


def test_ensure_dir(temp_test_dir):
    """
    測試 ensure_dir 功能是否能正確創建目錄。
    """
    test_path = os.path.join(temp_test_dir, "new_dir", "sub_dir")
    file_utils.ensure_dir(test_path)
    assert os.path.exists(test_path)
    assert os.path.isdir(test_path)


def test_list_files(temp_test_dir):
    """
    測試 list_files 功能是否能正確列出目錄中的文件。
    """
    # 創建多個測試文件
    for i in range(3):
        with open(os.path.join(temp_test_dir, f"file_{i}.txt"), "w") as f:
            f.write(f"Content of file {i}")
    
    files = file_utils.list_files(temp_test_dir, pattern="*.txt")
    assert len(files) == 3
    assert any("file_0.txt" in f for f in files)
    assert any("file_1.txt" in f for f in files)
    assert any("file_2.txt" in f for f in files)


def test_load_json(temp_test_dir):
    """
    測試 load_json 功能是否能正確載入 JSON 文件。
    """
    json_path = os.path.join(temp_test_dir, "test.json")
    data = file_utils.load_json(json_path)
    assert data is not None
    assert data["key"] == "value"
    assert data["numbers"] == [1, 2, 3]


def test_save_json(temp_test_dir):
    """
    測試 save_json 功能是否能正確保存 JSON 文件。
    """
    json_path = os.path.join(temp_test_dir, "new_test.json")
    test_data = {
        "name": "test",
        "values": [4, 5, 6],
        "nested": {"a": 1, "b": 2}
    }
    
    file_utils.save_json(json_path, test_data)
    
    # 驗證文件是否被創建
    assert os.path.exists(json_path)
    
    # 讀取並驗證內容
    with open(json_path, 'r') as f:
        loaded_data = json.load(f)
    
    assert loaded_data == test_data


def test_tensor_to_numpy():
    """
    測試 tensor_to_numpy 功能是否能正確將張量轉換為 numpy 數組。
    """
    # 創建測試張量
    test_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    
    # 轉換為 numpy
    numpy_array = tensor_utils.tensor_to_numpy(test_tensor)
    
    # 驗證結果
    assert isinstance(numpy_array, np.ndarray)
    assert numpy_array.shape == (2, 2)
    assert np.allclose(numpy_array, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_numpy_to_tensor():
    """
    測試 numpy_to_tensor 功能是否能正確將 numpy 數組轉換為張量。
    """
    # 創建測試 numpy 數組
    test_array = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    # 轉換為張量
    tensor = tensor_utils.numpy_to_tensor(test_array)
    
    # 驗證結果
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (2, 2)
    assert torch.allclose(tensor, torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float))


def test_load_tensor(temp_test_dir):
    """
    測試 load_tensor 功能是否能正確加載張量文件。
    """
    tensor_path = os.path.join(temp_test_dir, "tensor.pt")
    tensor = tensor_utils.load_tensor(tensor_path)
    
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (5, 10)


def test_save_tensor(temp_test_dir):
    """
    測試 save_tensor 功能是否能正確保存張量文件。
    """
    tensor_path = os.path.join(temp_test_dir, "new_tensor.pt")
    test_tensor = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    
    tensor_utils.save_tensor(tensor_path, test_tensor)
    
    # 驗證文件是否被創建
    assert os.path.exists(tensor_path)
    
    # 加載並驗證內容
    loaded_tensor = torch.load(tensor_path)
    assert torch.allclose(loaded_tensor, test_tensor) 