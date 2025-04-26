"""
# 測試HookDataLoader功能

測試日期: 2025-04-30
測試目的: 測試HookDataLoader類的功能，包括加載模型鉤子數據和進行預處理
"""
import os
import pytest
import torch
import tempfile
import shutil
import json
from pathlib import Path
import sys
import logging
from unittest.mock import patch

# 添加專案根目錄到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader.hook_data_loader import HookDataLoader

# Helper function to create dummy hook files with new naming convention
def _create_dummy_hook_files(base_dir, epochs=2, layers=['conv1', 'conv2', 'fc1'], batches_per_epoch=1, batch_size=16, feature_dim=100):
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        epoch_dir = base_path / f'epoch_{epoch}'
        epoch_dir.mkdir(exist_ok=True)

        # Create dummy epoch summary
        summary_data = {'epoch': epoch, 'avg_loss': float(torch.rand(1).item())}
        torch.save(summary_data, epoch_dir / 'epoch_summary.pt')

        for batch_idx in range(batches_per_epoch):
            # Batch directory is no longer needed if files include batch index
            # batch_dir = epoch_dir / f'batch_{batch_idx}'
            # batch_dir.mkdir(exist_ok=True)
            for layer_name in layers:
                activation_tensor = torch.randn(batch_size, feature_dim)
                # Use new file naming format
                file_path = epoch_dir / f'{layer_name}_activation_batch_{batch_idx}.pt'
                torch.save(activation_tensor, file_path)
            # Optional: create batch data file if needed
            batch_data = {'batch_id': batch_idx, 'size': batch_size}
            torch.save(batch_data, epoch_dir / f'batch_{batch_idx}_data.pt')

    # Create overall training summary (optional, depends on usage)
    training_summary = {'total_epochs': epochs, 'final_loss': float(torch.rand(1).item())}
    torch.save(training_summary, base_path / 'training_summary.pt')

# --- Fixtures ---

@pytest.fixture
def hook_data_dir(tmp_path):
    # Use hooks_dir consistent with the class attribute
    hooks_dir = tmp_path / 'hooks'
    _create_dummy_hook_files(hooks_dir, epochs=3, layers=['conv1', 'conv2', 'fc1', 'fc2', 'fc3'], batches_per_epoch=1)
    # Return the parent experiment directory, not the hooks dir itself
    return str(tmp_path)

@pytest.fixture
def hook_loader(hook_data_dir):
    # Initialize with the experiment directory
    return HookDataLoader(hook_data_dir)

@pytest.fixture
def temp_experiment_dir(tmp_path):
    """創建一個臨時實驗目錄，包含測試所需的所有文件"""
    # 創建臨時目錄
    temp_dir = tmp_path / "test_experiment"
    temp_dir.mkdir()
    
    # 創建hooks目錄
    hooks_dir = temp_dir / "hooks"
    hooks_dir.mkdir()
    
    # 創建訓練摘要
    training_summary = {
        "total_epochs": 2, 
        "total_training_time": 1000, 
        "final_loss": 0.1
    }
    torch.save(training_summary, hooks_dir / "training_summary.pt")
    
    # 創建評估結果
    evaluation_results = {
        "accuracy": 0.85,
        "precision": 0.83,
        "recall": 0.84,
        "f1": 0.835,
        "f1_score": 0.82,
        "loss": 0.15  # 添加測試需要的loss字段
    }
    torch.save(evaluation_results, hooks_dir / "evaluation_results_test.pt")
    
    # 創建epoch目錄和激活文件
    for epoch in [0, 1]:
        epoch_dir = hooks_dir / f"epoch_{epoch}"
        epoch_dir.mkdir()
        
        # 創建epoch摘要，添加測試需要的loss字段和accuracy字段
        epoch_summary = {
            "epoch": epoch,
            "train_loss": 0.5 - epoch * 0.2,
            "val_loss": 0.6 - epoch * 0.15,
            "loss": 0.3 if epoch == 1 else 0.5,
            "accuracy": 0.7 if epoch == 1 else 0.5
        }
        torch.save(epoch_summary, epoch_dir / "epoch_summary.pt")
        
        # 創建批次數據
        batch_data = {
            "batch_id": 0,
            "inputs": torch.randn(32, 3, 224, 224),
            "targets": torch.randint(0, 10, (32,)),
            "loss": 0.4 - epoch * 0.1
        }
        torch.save(batch_data, epoch_dir / "batch_0_data.pt")
        
        # 添加測試用的批次数据1
        batch_data_1 = {
            "batch_id": 1,
            "inputs": torch.randn(32, 3, 224, 224),
            "targets": torch.randint(0, 10, (32,)),
            "loss": 0.35 - epoch * 0.1
        }
        torch.save(batch_data_1, epoch_dir / "batch_1_data.pt")
        
        # 創建各層激活值，匹配測試期望的維度
        layers = {
            'conv1': (32, 32, 30, 30),  # 修改batch size為32
            'conv2': (32, 64, 15, 15),  # 修改batch size為32
            'fc1': (32, 120),           # 修改為符合測試期望的形狀
            'fc3': (32, 10),            # 修改batch size為32
            'dense': (32, 64)           # 修改batch size為32
        }
        
        for layer, shape in layers.items():
            activation = torch.randn(*shape)
            torch.save(activation, epoch_dir / f"{layer}_activation_batch_0.pt")
    
    yield str(temp_dir)
    
    # 清理
    shutil.rmtree(temp_dir)

# --- Test Class ---

def test_init_with_valid_dir(hook_data_dir):
    # Initialize with experiment dir
    loader = HookDataLoader(hook_data_dir)
    # Check hooks_dir attribute
    assert loader.hooks_dir == os.path.join(hook_data_dir, 'hooks')
    assert loader.hooks_available is True

def test_init_with_invalid_dir(tmp_path, caplog):
    invalid_exp_dir = tmp_path / "non_existent_experiment"
    hooks_dir = invalid_exp_dir / "hooks"
    with caplog.at_level(logging.WARNING):
        loader = HookDataLoader(str(invalid_exp_dir))
        # Check hooks_dir attribute
        assert loader.hooks_dir == str(hooks_dir)
        assert loader.hooks_available is False
        assert not list(loader.list_available_epochs())
        assert "Hooks目錄不存在" in caplog.text

def test_list_available_epochs(hook_loader):
    epochs = hook_loader.list_available_epochs()
    assert isinstance(epochs, list)
    # Updated based on _create_dummy_hook_files epochs=3
    assert len(epochs) == 3
    assert epochs == [0, 1, 2]

def test_list_available_layers(hook_loader):
    # Test list_available_layers with epoch and batch
    layers_epoch0_batch0 = hook_loader.list_available_layers(epoch=0, batch=0)
    expected_layers = ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']
    assert isinstance(layers_epoch0_batch0, list)
    assert set(layers_epoch0_batch0) == set(expected_layers)

    # Test with non-existent batch
    layers_epoch0_batch1 = hook_loader.list_available_layers(epoch=0, batch=1)
    assert layers_epoch0_batch1 == []

def test_list_available_batches(hook_loader):
    batches_epoch0 = hook_loader.list_available_batches(epoch=0)
    assert isinstance(batches_epoch0, list)
    # _create_dummy_hook_files creates batch 0
    assert len(batches_epoch0) == 1
    assert batches_epoch0 == [0]

    batches_epoch99 = hook_loader.list_available_batches(epoch=99)
    assert batches_epoch99 == []

def test_load_layer_activation(hook_loader):
    # Use the correct signature with batch parameter
    activation = hook_loader.load_layer_activation(layer_name='conv1', epoch=1, batch=0)
    assert isinstance(activation, torch.Tensor)
    assert activation.shape == (16, 100) # Based on _create_dummy_hook_files setup

def test_load_nonexistent_layer_activation(temp_experiment_dir):
    """測試載入不存在層的激活值"""
    loader = HookDataLoader(temp_experiment_dir)
    # 測試期望此处抛出FileNotFoundError异常
    with pytest.raises(FileNotFoundError):
        activation = loader.load_layer_activation('nonexistent_layer', 0, 0)

def test_load_layer_activation_invalid_epoch_or_batch(hook_loader):
    # Test invalid epoch
    with pytest.raises(FileNotFoundError):
        hook_loader.load_layer_activation(layer_name='conv1', epoch=99, batch=0)
    # Test invalid batch
    with pytest.raises(FileNotFoundError):
        hook_loader.load_layer_activation(layer_name='conv1', epoch=0, batch=99)

def test_load_epoch_summary(hook_loader):
    summary = hook_loader.load_epoch_summary(epoch=0)
    assert isinstance(summary, dict)
    assert 'epoch' in summary
    assert 'avg_loss' in summary
    assert summary['epoch'] == 0

def test_load_epoch_summary_invalid_epoch(hook_loader):
    with pytest.raises(FileNotFoundError):
        hook_loader.load_epoch_summary(epoch=99)

def test_load_training_summary(hook_loader):
    summary = hook_loader.load_training_summary()
    assert isinstance(summary, dict)
    assert 'total_epochs' in summary
    assert 'final_loss' in summary

def test_load_training_summary_not_found(tmp_path):
    # Test with an experiment dir that doesn't have the summary file
    loader = HookDataLoader(str(tmp_path))
    # Should now return None instead of raising error
    summary = loader.load_training_summary()
    assert summary is None

def test_load_all_layer_activations(temp_experiment_dir):
    """測試載入單一輪次所有層的激活值"""
    loader = HookDataLoader(temp_experiment_dir)
    epoch = 0
    all_activations = loader.load_all_layer_activations(epoch=epoch, batch=0)
    assert isinstance(all_activations, dict)
    assert "conv1" in all_activations
    assert "fc1" in all_activations
    assert torch.is_tensor(all_activations['conv1'])
    # 修改為實際創建的batch size為32
    assert all_activations['conv1'].shape[0] == 32  # Check batch size


@patch('torch.load')
def test_load_error_handling(mock_torch_load, hook_loader, caplog):
    # Simulate torch.load raising an exception
    mock_torch_load.side_effect = Exception("Simulated loading error")

    with caplog.at_level(logging.ERROR):
        # Use correct signature
        activation = hook_loader.load_layer_activation(layer_name='conv1', epoch=0, batch=0)
        assert activation is None # Should return None on error
        assert "無法載入激活值文件" in caplog.text
        assert "Simulated loading error" in caplog.text

    # Simulate FileNotFoundError (e.g., file deleted between check and load)
    mock_torch_load.side_effect = FileNotFoundError("Simulated file not found")
    # This case now raises FileNotFoundError in load_layer_activation
    with pytest.raises(FileNotFoundError):
         hook_loader.load_layer_activation(layer_name='fc1', epoch=0, batch=0)

# Remove old fixture and tests that rely on old structure/signatures
# @pytest.fixture
# def temp_experiment_dir(): ...
# def test_load_training_summary(temp_experiment_dir): ... (covered by hook_loader tests)
# ... and other similar tests ...


def test_load_training_summary(temp_experiment_dir):
    """測試載入訓練摘要"""
    loader = HookDataLoader(temp_experiment_dir)
    summary = loader.load_training_summary()
    assert summary['total_epochs'] == 2
    assert summary['total_training_time'] == 1000
    assert summary['final_loss'] == 0.1


def test_load_evaluation_results(temp_experiment_dir):
    """測試載入評估結果"""
    loader = HookDataLoader(temp_experiment_dir)
    results = loader.load_evaluation_results('test')
    assert results['accuracy'] == 0.85
    assert results['loss'] == 0.15
    assert results['f1_score'] == 0.82


def test_load_epoch_summary(temp_experiment_dir):
    """測試載入輪次摘要"""
    loader = HookDataLoader(temp_experiment_dir)
    summary = loader.load_epoch_summary(1)
    assert summary['epoch'] == 1
    assert summary['loss'] == 0.3
    assert summary['accuracy'] == 0.7


def test_load_batch_data(temp_experiment_dir):
    """測試載入批次數據"""
    loader = HookDataLoader(temp_experiment_dir)
    batch_data = loader.load_batch_data(0, 1)
    assert 'inputs' in batch_data
    assert 'targets' in batch_data
    assert 'loss' in batch_data
    assert batch_data['inputs'].shape == (32, 3, 224, 224)  # 修改為實際創建的張量維度


def test_load_layer_activation(temp_experiment_dir):
    """測試載入層激活值"""
    loader = HookDataLoader(temp_experiment_dir)
    activation = loader.load_layer_activation('conv1', 0, 0)
    assert isinstance(activation, torch.Tensor)
    assert activation.shape == (32, 32, 30, 30)  # 修改為實際創建的激活值維度


def test_nonexistent_layer_should_raise_error(temp_experiment_dir):
    """測試載入不存在層的激活值應該拋出異常"""
    loader = HookDataLoader(temp_experiment_dir)
    with pytest.raises(FileNotFoundError):
        loader.load_layer_activation('nonexistent_layer', 0, 0)


def test_load_nonexistent_epoch(temp_experiment_dir):
    """測試載入不存在輪次的激活值"""
    loader = HookDataLoader(temp_experiment_dir)
    # Check that trying to access a non-existent epoch raises error
    with pytest.raises(FileNotFoundError, match="輪次目錄不存在"):
        # Call a method that explicitly checks epoch dir, like load_epoch_summary
        # or adjust if load_layer_activation should also raise this
        loader.load_layer_activation('conv1', 99, 0) 
        # Alternatively, if only warnings are logged:
        # activation = loader.load_layer_activation('conv1', 99, 0)
        # assert activation is None 


def test_load_all_layer_activations(temp_experiment_dir):
    """測試載入單一輪次所有層的激活值"""
    loader = HookDataLoader(temp_experiment_dir)
    epoch = 0
    all_activations = loader.load_all_layer_activations(epoch=epoch, batch=0)
    assert isinstance(all_activations, dict)
    assert "conv1" in all_activations
    assert "fc1" in all_activations
    assert torch.is_tensor(all_activations['conv1'])
    # 修改為實際創建的batch size為32
    assert all_activations['conv1'].shape[0] == 32  # Check batch size


# def test_load_layer_activations_across_epochs(temp_experiment_dir):
#     """測試載入單一層在多個輪次的激活值"""
#     loader = HookDataLoader(temp_experiment_dir)
#     layer_name = 'conv1'
#     activations = loader.load_layer_activations_across_epochs(layer_name)
#     assert isinstance(activations, dict)
#     assert 0 in activations
#     assert 1 in activations
#     assert torch.is_tensor(activations[0])
#     assert activations[0].shape[1] == 64 # Check channel dimension


def test_load_layer_activations_across_epochs(temp_experiment_dir):
    """測試載入跨輪次的層激活值"""
    loader = HookDataLoader(temp_experiment_dir)
    activations = loader.load_layer_activations_across_epochs('conv1', [0, 1], 0)

    assert len(activations) == 2  # 應該有2個輪次
    assert 0 in activations
    assert 1 in activations
    assert activations[0].shape == (32, 32, 30, 30)  # 修改為實際創建的激活值維度
    assert activations[1].shape == (32, 32, 30, 30)  # 修改為實際創建的激活值維度


# def test_load_activation_with_preprocessing(temp_experiment_dir):
#     """測試載入激活值並應用預處理函數"""
#     loader = HookDataLoader(temp_experiment_dir)
#     layer_name = 'fc1'
#     
#     def preprocess_func(tensor):
#         return tensor.mean(dim=0) # Example: calculate mean across batch
#     
#     processed_activation = loader.load_activation_with_preprocessing(layer_name, 0, preprocess_func=preprocess_func)
#     assert processed_activation.shape == (256,) # Check shape after preprocessing


def test_load_activation_with_preprocessing(temp_experiment_dir):
    """測試載入並預處理激活值數據"""
    loader = HookDataLoader(temp_experiment_dir)
    
    # 創建一個包含一些極端值的測試張量
    extreme_activation = torch.randn(16, 64, 10, 10)
    extreme_activation[0, 0, 0, 0] = 100.0  # 添加離群值
    
    # 保存到臨時目錄
    test_layer = 'test_layer'
    epoch_dir = os.path.join(temp_experiment_dir, 'hooks', 'epoch_0')
    activation_path = os.path.join(epoch_dir, f'{test_layer}_activation_batch_0.pt')
    torch.save(extreme_activation, activation_path)
    
    # 使用不同的預處理配置進行測試
    
    # 1. 正規化
    preprocessed = loader.load_activation_with_preprocessing(
        test_layer, 0, 0, 
        preprocess_config={'normalize': 'minmax'}
    )
    assert preprocessed is not None
    assert 0.0 <= preprocessed.min() <= preprocessed.max() <= 1.0
    
    # 2. 展平
    preprocessed = loader.load_activation_with_preprocessing(
        test_layer, 0, 0, 
        preprocess_config={'flatten': True}
    )
    assert preprocessed is not None
    assert preprocessed.shape == (16, 64*10*10)
    
    # 3. 移除離群值
    preprocessed = loader.load_activation_with_preprocessing(
        test_layer, 0, 0, 
        preprocess_config={'remove_outliers': True, 'outlier_threshold': 3.0}
    )
    assert preprocessed is not None
    assert torch.abs(preprocessed).max() < 10.0  # 離群值應被替換
    
    # 4. 組合使用多種預處理
    preprocessed = loader.load_activation_with_preprocessing(
        test_layer, 0, 0, 
        preprocess_config={
            'normalize': 'zscore',
            'flatten': True,
            'remove_outliers': True
        }
    )
    assert preprocessed is not None
    assert preprocessed.shape == (16, 64*10*10)
    assert torch.abs(preprocessed).max() < 10.0
    
    # 5. 測試無預處理配置
    preprocessed = loader.load_activation_with_preprocessing(
        test_layer, 0, 0, 
        preprocess_config=None
    )
    assert preprocessed is not None
    assert torch.equal(preprocessed, extreme_activation)
    
    # 6. 測試不存在的層
    preprocessed = loader.load_activation_with_preprocessing(
        'nonexistent_layer', 0, 0, 
        preprocess_config={'normalize': 'minmax'}
    )
    assert preprocessed is None


# def test_load_activations_batch(temp_experiment_dir):
#     """測試載入特定批次的激活數據。"""
#     # 假設您已在 temp_experiment_dir 中創建了包含多個批次的數據
#     # 例如: epoch_0/conv1_activation_batch_0.pt, epoch_0/conv1_activation_batch_1.pt
#     
#     # 創建一個包含多個批次的模擬文件
#     hooks_dir = Path(temp_experiment_dir) / "hooks" / "epoch_0"
#     tensor_batch1 = torch.randn(32, 64, 16, 16)
#     torch.save(tensor_batch1, hooks_dir / "conv1_activation_batch_1.pt")
#     
#     loader = HookDataLoader(temp_experiment_dir)
#     activation_b0 = loader.load_activations_batch('conv1', 0, 0)
#     activation_b1 = loader.load_activations_batch('conv1', 0, 1)
#     
#     assert torch.is_tensor(activation_b0)
#     assert torch.is_tensor(activation_b1)
#     # 驗證不同批次的數據是否不同（如果它們確實不同）
#     # 注意：由於隨機生成，這裡的比較可能不穩定，除非使用固定種子
#     # assert not torch.equal(activation_b0, activation_b1)


def test_load_activations_batch(temp_experiment_dir):
    """測試批量載入激活值數據"""
    loader = HookDataLoader(temp_experiment_dir)

    # 批量載入多個層和多個輪次的激活值
    results = loader.load_activations_batch(
        layer_names=['conv1', 'fc1'],
        epochs=[0, 1],
        batch=0
    )

    assert 'conv1' in results
    assert 'fc1' in results
    assert 0 in results['conv1']
    assert 1 in results['conv1']
    assert 0 in results['fc1']
    assert 1 in results['fc1']

    # 確認張量形狀正確，修改為實際創建的激活值維度
    assert results['conv1'][0].shape == (32, 32, 30, 30)
    assert results['fc1'][0].shape == (32, 120)


def test_load_layer_statistics(temp_experiment_dir):
    """測試載入層統計數據"""
    loader = HookDataLoader(temp_experiment_dir)
    
    # 直接計算統計數據
    stats = loader.load_layer_statistics('conv1', 0, 0)
    
    assert stats is not None
    assert stats['layer_name'] == 'conv1'
    assert stats['epoch'] == 0
    assert stats['batch'] == 0
    assert 'shape' in stats
    assert 'mean' in stats
    assert 'std' in stats
    assert 'min' in stats
    assert 'max' in stats
    assert 'sparsity' in stats
    assert 'has_nan' in stats
    assert 'has_inf' in stats
    
    # 創建並測試預先計算的統計數據文件
    stats_dir = os.path.join(temp_experiment_dir, 'hooks', 'epoch_0')
    stats_file = os.path.join(stats_dir, 'conv1_stats.json')
    
    pre_computed_stats = {
        'layer_name': 'conv1',
        'epoch': 0,
        'pre_computed': True,
        'mean': 0.1,
        'std': 0.5
    }
    
    with open(stats_file, 'w') as f:
        json.dump(pre_computed_stats, f)
    
    # 應該載入預先計算的數據
    stats = loader.load_layer_statistics('conv1', 0, 0)
    assert stats['pre_computed']
    assert stats['mean'] == 0.1
    
    # 測試不存在的層
    stats = loader.load_layer_statistics('nonexistent_layer', 0, 0)
    assert stats is None


def test_get_activation_metadata(temp_experiment_dir):
    """測試獲取激活值元數據"""
    loader = HookDataLoader(temp_experiment_dir)
    
    metadata = loader.get_activation_metadata()
    
    assert 'available_epochs' in metadata
    assert 'available_layers' in metadata
    assert 'layer_details' in metadata
    
    assert metadata['available_epochs'] == [0, 1]
    assert 'conv1' in metadata['layer_details']
    assert 'fc3' in metadata['layer_details']
    
    # 檢查層詳情
    conv1_details = metadata['layer_details']['conv1']
    assert len(conv1_details['epochs']) > 0
    assert 'shape' in conv1_details['epochs'][0]
    assert 'dtype' in conv1_details['epochs'][0]


def test_save_processed_activation(temp_experiment_dir):
    """測試保存處理後的激活值"""
    loader = HookDataLoader(temp_experiment_dir)
    
    # 創建一個處理後的激活值
    processed_activation = torch.randn(16, 64)
    
    # 保存到實驗目錄
    file_path = loader.save_processed_activation(
        processed_activation, 
        'conv1', 
        0, 
        0, 
        'normalized'
    )
    
    # 確認文件已保存
    assert os.path.exists(file_path)
    
    # 載入保存的激活值並比較
    loaded = torch.load(file_path)
    assert torch.equal(loaded, processed_activation)
    
    # 測試使用不同的後綴
    file_path2 = loader.save_processed_activation(
        processed_activation, 
        'conv1', 
        0, 
        0, 
        'flattened'
    )
    
    assert os.path.exists(file_path2)
    assert 'flattened' in file_path2
    
    # 確認可以創建不存在的目錄
    new_path = loader.save_processed_activation(
        processed_activation, 
        'conv1', 
        99, 
        0, 
        'test'
    )
    
    assert os.path.exists(new_path)
    assert 'epoch_99' in new_path 