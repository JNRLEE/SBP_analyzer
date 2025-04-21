"""
文件操作工具模組。

此模組提供用於文件操作的實用函數。

Functions:
    get_files_by_extension: 獲取指定擴展名的文件列表
    ensure_dir_exists: 確保目錄存在
    get_experiment_dirs: 獲取結果目錄下的所有實驗目錄
    get_latest_experiments: 獲取最新的n個實驗目錄
    is_experiment_dir_valid: 檢查指定的目錄是否為有效的實驗目錄
"""

import os
import glob
from typing import List, Optional

def get_files_by_extension(directory: str, extension: str) -> List[str]:
    """
    獲取指定目錄下具有特定擴展名的文件列表。
    
    Args:
        directory (str): 目錄路徑
        extension (str): 文件擴展名，例如'.txt'
    
    Returns:
        List[str]: 符合條件的文件路徑列表
    """
    if not directory or not os.path.isdir(directory):
        return []
    
    # 確保擴展名以點開頭
    if extension and not extension.startswith('.'):
        extension = f'.{extension}'
    
    # 使用glob模塊匹配文件
    pattern = os.path.join(directory, f'*{extension}')
    return glob.glob(pattern)

def ensure_dir_exists(directory: str) -> str:
    """
    確保目錄存在，如果不存在則創建。
    
    Args:
        directory (str): 目錄路徑
    
    Returns:
        str: 輸入的目錄路徑
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return directory

def save_json(file_path: str, data: dict, indent: int = 4) -> bool:
    """
    將字典數據保存為JSON文件。
    
    Args:
        file_path (str): 文件路徑
        data (dict): 要保存的數據
        indent (int, optional): 縮進空格數。默認為4
    
    Returns:
        bool: 保存成功則返回True，否則返回False
    """
    import json
    try:
        # 確保目錄存在
        ensure_dir_exists(os.path.dirname(os.path.abspath(file_path)))
        
        # 寫入JSON文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"保存JSON文件失敗: {e}")
        return False

def load_json(file_path: str) -> Optional[dict]:
    """
    載入JSON文件。
    
    Args:
        file_path (str): 文件路徑
    
    Returns:
        Optional[dict]: 載入的數據，如果載入失敗則返回None
    """
    import json
    try:
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"載入JSON文件失敗: {e}")
        return None

def get_experiment_dirs(results_dir: str) -> List[str]:
    """
    獲取結果目錄下的所有實驗目錄。
    
    Args:
        results_dir (str): 結果主目錄路徑
    
    Returns:
        List[str]: 實驗目錄路徑列表
    """
    # 檢查結果目錄是否存在
    if not os.path.exists(results_dir) or not os.path.isdir(results_dir):
        return []
    
    # 獲取所有子目錄，排除隱藏目錄和__pycache__
    experiment_dirs = []
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.') and item != '__pycache__':
            experiment_dirs.append(item_path)
    
    return experiment_dirs

def get_latest_experiments(results_dir: str, n: int = 1) -> List[str]:
    """
    獲取最新的n個實驗目錄。
    
    Args:
        results_dir (str): 結果主目錄路徑
        n (int, optional): 需要獲取的實驗數量。默認為1
    
    Returns:
        List[str]: 最新的n個實驗目錄路徑列表
    """
    experiment_dirs = get_experiment_dirs(results_dir)
    
    # 根據目錄的修改時間進行排序
    experiment_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # 返回最新的n個，如果實驗數少於n則返回全部
    return experiment_dirs[:min(n, len(experiment_dirs))]

def is_experiment_dir_valid(experiment_dir: str) -> bool:
    """
    檢查指定的目錄是否為有效的實驗目錄。
    
    一個有效的實驗目錄至少應該包含以下文件之一：
    - config.json
    - model_structure.json
    - training_history.json
    
    Args:
        experiment_dir (str): 實驗目錄路徑
    
    Returns:
        bool: 如果是有效的實驗目錄則返回True，否則返回False
    """
    if not os.path.exists(experiment_dir) or not os.path.isdir(experiment_dir):
        return False
    
    # 檢查是否包含關鍵文件
    config_path = os.path.join(experiment_dir, 'config.json')
    structure_path = os.path.join(experiment_dir, 'model_structure.json')
    history_path = os.path.join(experiment_dir, 'training_history.json')
    
    return os.path.exists(config_path) or os.path.exists(structure_path) or os.path.exists(history_path) 