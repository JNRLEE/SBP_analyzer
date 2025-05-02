"""
# 檔案操作工具
"""
import os
import glob
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

def ensure_dir(directory: str) -> str:
    """
    確保目錄存在，若不存在則創建
    
    Args:
        directory: 目錄路徑
        
    Returns:
        str: 創建或確認的目錄路徑
    """
    os.makedirs(directory, exist_ok=True)
    return directory

# 保留原名稱作為別名，這樣兩種名稱都可以使用
ensure_dir_exists = ensure_dir

def get_files_by_extension(directory: str, extension: str) -> List[str]:
    """
    獲取指定目錄下特定副檔名的文件
    
    Args:
        directory: 目錄路徑
        extension: 文件副檔名，例如 '.json'
        
    Returns:
        List[str]: 符合條件的文件路徑列表
    """
    if not extension.startswith('.'):
        extension = '.' + extension
    
    pattern = os.path.join(directory, f'*{extension}')
    return glob.glob(pattern)

def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    載入 JSON 檔案
    
    Args:
        file_path: JSON 檔案路徑
        
    Returns:
        Dict[str, Any]: 解析後的 JSON 內容
    
    Raises:
        FileNotFoundError: 如果檔案不存在
        json.JSONDecodeError: 如果 JSON 解析失敗
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"檔案不存在: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"JSON 解析失敗: {e.msg}", e.doc, e.pos)

def save_json_file(data: Dict[str, Any], file_path: str, indent: int = 4) -> None:
    """
    將數據保存為 JSON 檔案
    
    Args:
        data: 要保存的數據
        file_path: 目標檔案路徑
        indent: JSON 縮排空格數
    """
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=indent, ensure_ascii=False)

def load_pickle_file(file_path: str) -> Any:
    """
    載入 pickle 檔案
    
    Args:
        file_path: pickle 檔案路徑
        
    Returns:
        Any: 解析後的 pickle 內容
    
    Raises:
        FileNotFoundError: 如果檔案不存在
    """
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"檔案不存在: {file_path}")
    except Exception as e:
        raise Exception(f"載入 pickle 檔案失敗: {str(e)}")

def save_pickle_file(data: Any, file_path: str) -> None:
    """
    將數據保存為 pickle 檔案
    
    Args:
        data: 要保存的數據
        file_path: 目標檔案路徑
    """
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def list_subdirectories(directory: str) -> List[str]:
    """
    列出指定目錄下的所有子目錄
    
    Args:
        directory: 父目錄路徑
        
    Returns:
        List[str]: 子目錄路徑列表
    """
    return [d for d in glob.glob(os.path.join(directory, '*')) if os.path.isdir(d)]

def find_files_recursive(directory: str, pattern: str) -> List[str]:
    """
    遞歸查找符合模式的文件
    
    Args:
        directory: 起始目錄
        pattern: 文件名模式，例如 '*.json'
        
    Returns:
        List[str]: 符合條件的文件路徑列表
    """
    result = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if glob.fnmatch.fnmatch(filename, pattern):
                result.append(os.path.join(root, filename))
    return result

def get_most_recent_file(directory: str, pattern: str) -> Optional[str]:
    """
    獲取指定目錄下最近修改的符合模式的文件
    
    Args:
        directory: 目錄路徑
        pattern: 文件名模式，例如 '*.json'
        
    Returns:
        Optional[str]: 最近修改的文件路徑，如果沒有找到則返回 None
    """
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    
    return max(files, key=os.path.getmtime)

def get_file_size(file_path: str, unit: str = 'MB') -> float:
    """
    獲取文件大小
    
    Args:
        file_path: 文件路徑
        unit: 單位，可選 'B', 'KB', 'MB', 'GB'
        
    Returns:
        float: 文件大小
    
    Raises:
        FileNotFoundError: 如果文件不存在
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    size_bytes = os.path.getsize(file_path)
    
    units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
    
    if unit not in units:
        raise ValueError(f"不支持的單位: {unit}，支持的單位有 {list(units.keys())}")
    
    return size_bytes / units[unit]

def merge_json_files(file_paths: List[str]) -> Dict[str, Any]:
    """
    合併多個 JSON 文件的內容
    
    Args:
        file_paths: JSON 文件路徑列表
        
    Returns:
        Dict[str, Any]: 合併後的 JSON 內容
    """
    result = {}
    for file_path in file_paths:
        data = load_json_file(file_path)
        # 使用深度合併邏輯，而不是簡單的更新
        result = deep_merge(result, data)
    
    return result

def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合併兩個字典
    
    Args:
        dict1: 第一個字典
        dict2: 第二個字典
        
    Returns:
        Dict[str, Any]: 合併後的字典
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if (
            key in result and 
            isinstance(result[key], dict) and 
            isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result 