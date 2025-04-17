"""
# 文件工具
# 提供處理文件和目錄的輔助函數
"""

import os
import glob
import re
from typing import List, Optional, Pattern, Union


def find_files(
    directory: str,
    pattern: str = "*",
    recursive: bool = False,
    full_path: bool = True
) -> List[str]:
    """
    在指定目錄中查找符合模式的文件。
    
    Args:
        directory: 要搜索的目錄路徑
        pattern: 文件匹配模式，如 "*.json"
        recursive: 是否遞歸搜索子目錄
        full_path: 是否返回完整路徑
        
    Returns:
        List[str]: 找到的文件列表
        
    Description:
        使用 glob 模式在指定目錄中搜索文件，可以選擇遞歸搜索子目錄。
        
    References:
        - Python glob module: https://docs.python.org/3/library/glob.html
    """
    if not os.path.exists(directory):
        return []
    
    if recursive:
        search_pattern = os.path.join(directory, "**", pattern)
        files = glob.glob(search_pattern, recursive=True)
    else:
        search_pattern = os.path.join(directory, pattern)
        files = glob.glob(search_pattern)
    
    if not full_path:
        files = [os.path.basename(file) for file in files]
    
    return files


def ensure_dir(directory: str) -> str:
    """
    確保目錄存在，如果不存在則創建。
    
    Args:
        directory: 要確保存在的目錄路徑
        
    Returns:
        str: 目錄路徑
        
    Description:
        檢查指定的目錄是否存在，如果不存在則創建。
        
    References:
        None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def get_experiment_id(directory: str) -> Optional[str]:
    """
    從實驗目錄路徑中提取實驗 ID。
    
    Args:
        directory: 實驗目錄路徑
        
    Returns:
        Optional[str]: 實驗 ID，如果無法提取則返回 None
        
    Description:
        從目錄路徑中提取實驗 ID，假設目錄名格式為 "{實驗名稱}_{時間戳}"。
        
    References:
        None
    """
    # 獲取目錄名（不含路徑）
    dir_name = os.path.basename(directory.rstrip('/'))
    
    # 嘗試匹配標準實驗目錄命名格式：{實驗名稱}_{時間戳}
    # 例如：audio_swin_regression_20250417_142912
    timestamp_pattern = re.compile(r'^(.+)_(\d{8}_\d{6})$')
    match = timestamp_pattern.match(dir_name)
    
    if match:
        experiment_name = match.group(1)
        timestamp = match.group(2)
        return f"{experiment_name}_{timestamp}"
    
    # 如果沒有匹配標準格式，直接返回目錄名作為實驗 ID
    return dir_name


def filter_files_by_regex(
    file_list: List[str],
    pattern: Union[str, Pattern],
    full_match: bool = False
) -> List[str]:
    """
    使用正則表達式過濾文件列表。
    
    Args:
        file_list: 要過濾的文件列表
        pattern: 正則表達式模式，可以是字符串或已編譯的模式
        full_match: 是否要求完全匹配
        
    Returns:
        List[str]: 過濾後的文件列表
        
    Description:
        使用正則表達式過濾文件列表，可以指定是否要求完全匹配。
        
    References:
        - Python re module: https://docs.python.org/3/library/re.html
    """
    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    
    filtered_files = []
    for file in file_list:
        if full_match:
            if pattern.fullmatch(file):
                filtered_files.append(file)
        else:
            if pattern.search(file):
                filtered_files.append(file)
    
    return filtered_files


def get_file_extension(file_path: str) -> str:
    """
    獲取文件的擴展名。
    
    Args:
        file_path: 文件路徑
        
    Returns:
        str: 文件擴展名，不包含點號（.）
        
    Description:
        從文件路徑中提取擴展名，例如 "example.txt" 返回 "txt"。
        
    References:
        None
    """
    return os.path.splitext(file_path)[1].lstrip('.')


def get_checkpoint_epoch(checkpoint_file: str) -> Optional[int]:
    """
    從檢查點文件名中提取輪次編號。
    
    Args:
        checkpoint_file: 檢查點文件路徑
        
    Returns:
        Optional[int]: 輪次編號，如果無法提取則返回 None
        
    Description:
        從標準命名的檢查點文件中提取輪次編號，假設文件名格式為 "checkpoint_epoch_{N}.pth"。
        
    References:
        None
    """
    # 獲取文件名（不含路徑）
    filename = os.path.basename(checkpoint_file)
    
    # 嘗試匹配檢查點文件命名格式：checkpoint_epoch_{N}.pth
    checkpoint_pattern = re.compile(r'checkpoint_epoch_(\d+)\.pth')
    match = checkpoint_pattern.search(filename)
    
    if match:
        return int(match.group(1))
    
    return None 