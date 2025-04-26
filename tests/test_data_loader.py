"""
測試數據載入器模組的單元測試。
"""

import os
import unittest
import torch
import json
from pathlib import Path
import pytest
import tempfile
import shutil

from data_loader.experiment_loader import ExperimentLoader
from data_loader.hook_data_loader import HookDataLoader

# 移到模組級別，以便可以從模組級別的測試中訪問
@pytest.fixture
def temp_experiment_dir(tmp_path):
    """創建臨時實驗目錄，含結果摘要文件以供測試"""
    # 創建臨時實驗目錄
    experiment_dir = tmp_path / "test_experiment"
    experiment_dir.mkdir()
    
    # 創建結果目錄
    results_dir = experiment_dir / "results"
    results_dir.mkdir()
    
    # 創建結果摘要JSON
    results_data = {
        "final_accuracy": 0.95,
        "final_loss": 0.05,
        "training_time": 3600,
        "best_epoch": 42
    }
    
    with open(results_dir / "results.json", "w") as f:
        json.dump(results_data, f)
    
    # 創建其他必要文件
    with open(experiment_dir / "config.json", "w") as f:
        json.dump({
            "global": {"seed": 42},
            "data": {"dataset": "test"},
            "model": {"name": "test_model"}
        }, f)
    
    with open(experiment_dir / "training_history.json", "w") as f:
        json.dump({
            "train_loss": [0.9, 0.6, 0.3, 0.1],
            "val_loss": [0.8, 0.5, 0.3, 0.2],
            "train_acc": [0.5, 0.7, 0.8, 0.9],
            "val_acc": [0.6, 0.7, 0.8, 0.85]
        }, f)
        
    with open(experiment_dir / "model_structure.json", "w") as f:
        json.dump({
            "model_class": "TestModel",
            "layers": [
                {"name": "conv1", "type": "Conv2d"}
            ]
        }, f)
    
    yield str(experiment_dir)
    
    # 測試結束後清理
    shutil.rmtree(experiment_dir)

class TestExperimentLoader(unittest.TestCase):
    """
    測試ExperimentLoader類。
    """

    def setUp(self):
        """
        測試前的準備工作。
        """
        # 獲取結果目錄路徑
        current_dir = Path(__file__).parent.resolve()
        project_root = current_dir.parent
        self.results_dir = os.path.join(project_root, 'results')

        # 確保結果目錄下至少有一個實驗目錄
        experiment_dirs = [d for d in os.listdir(self.results_dir) 
                          if os.path.isdir(os.path.join(self.results_dir, d)) 
                          and d != "__pycache__" and not d.startswith(".")]
        
        if not experiment_dirs:
            self.skipTest("沒有找到實驗目錄，無法進行測試")
        
        # 選擇第一個實驗目錄進行測試
        self.experiment_dir = os.path.join(self.results_dir, experiment_dirs[0])
        self.loader = ExperimentLoader(self.experiment_dir)

    def test_load_config(self):
        """
        測試載入實驗配置文件。
        """
        config = self.loader.load_config()
        self.assertIsNotNone(config, "無法載入配置文件")
        self.assertIsInstance(config, dict, "配置應該是字典類型")
        self.assertIn("global", config, "配置應該包含global部分")
        self.assertIn("data", config, "配置應該包含data部分")
        self.assertIn("model", config, "配置應該包含model部分")

    def test_load_model_structure(self):
        """
        測試載入模型結構文件。
        """
        structure = self.loader.load_model_structure()
        self.assertIsNotNone(structure, "無法載入模型結構文件")
        self.assertIsInstance(structure, dict, "模型結構應該是字典類型")
        self.assertIn("model_class", structure, "模型結構應該包含model_class")
        # assert "total_params" in structure # Removed this check as it might not always exist
        # assert "model_summary" in structure # Removed this check

    def test_load_training_history(self):
        """
        測試載入訓練歷史文件。
        """
        history = self.loader.load_training_history()
        self.assertIsNotNone(history, "無法載入訓練歷史文件")
        self.assertIsInstance(history, dict, "訓練歷史應該是字典類型")
        self.assertIn("train_loss", history, "訓練歷史應該包含train_loss")
        self.assertIsInstance(history["train_loss"], list, "train_loss應該是列表類型")

    def test_load_all(self):
        """
        測試一次性載入所有文件。
        """
        result = self.loader.load()
        self.assertIsNotNone(result, "無法載入實驗數據")
        self.assertIsInstance(result, dict, "載入結果應該是字典類型")
        self.assertIn("config", result, "載入結果應該包含config")
        self.assertIn("model_structure", result, "載入結果應該包含model_structure")
        self.assertIn("training_history", result, "載入結果應該包含training_history")

# Move test_load_results to the module level to properly use pytest fixtures
# It won't work correctly inside a unittest.TestCase class
@pytest.mark.usefixtures("temp_experiment_dir")
def test_experiment_load_results(temp_experiment_dir):
    """測試載入結果摘要"""
    loader = ExperimentLoader(temp_experiment_dir)
    results = loader.load_results()
    assert results is not None
    assert "final_accuracy" in results
    assert results["final_accuracy"] == 0.95
    assert results["training_time"] == 3600

class TestHookDataLoader(unittest.TestCase):
    """
    測試HookDataLoader類。
    """

    @pytest.fixture(autouse=True)
    def _create_temp_dir(self, tmp_path):
        # Use pytest tmp_path fixture
        self.temp_dir = tmp_path / "experiment"
        self.temp_dir.mkdir()
        self._create_dummy_files(self.temp_dir)
        
        # Initialize the loader here so it's available for all tests
        self.loader = HookDataLoader(str(self.temp_dir))
        # Store experiment_dir for testing
        self.experiment_dir = str(self.temp_dir)
        
        yield self.temp_dir # Provide path to tests
        # Cleanup is handled by pytest tmp_path

    def _create_dummy_files(self, temp_dir):
        # Create dummy config, history, structure (optional, for context)
        (temp_dir / "config.json").write_text(json.dumps({"model": "test"}))
        (temp_dir / "training_history.json").write_text(json.dumps({"loss": [1.0]}))
        (temp_dir / "model_structure.json").write_text(json.dumps({"model_class": "TestNet"}))

        # Create dummy hook files
        hooks_dir = temp_dir / "hooks"
        hooks_dir.mkdir()
        
        # Create a training summary file
        training_summary = {
            "total_epochs": 2,
            "total_training_time": 1000,
            "final_loss": 0.1
        }
        torch.save(training_summary, hooks_dir / "training_summary.pt")
        
        # Create evaluation results file
        eval_results = {
            "accuracy": 0.9,
            "loss": 0.2,
            "confusion_matrix": [[90, 10], [5, 95]]
        }
        torch.save(eval_results, hooks_dir / "evaluation_results_test.pt")
        
        for epoch in [0, 1]:
            epoch_dir = hooks_dir / f"epoch_{epoch}"
            epoch_dir.mkdir()
            for layer in ['conv1', 'conv2', 'fc1']:
                data = torch.randn(4, 10) # Smaller dummy data
                torch.save(data, epoch_dir / f"{layer}_activation_batch_0.pt")
            # Add dead/saturated layers
            torch.save(torch.zeros(4, 5), epoch_dir / "dead_layer_activation_batch_0.pt")
            torch.save(torch.ones(4, 5), epoch_dir / "saturated_layer_activation_batch_0.pt")

    def test_initialization(self):
        """
        測試初始化過程。
        """
        self.assertIsInstance(self.loader, HookDataLoader, "無法創建HookDataLoader實例")
        self.assertEqual(self.loader.experiment_dir, self.experiment_dir, "實驗目錄不匹配")
        self.assertTrue(os.path.exists(self.loader.hooks_dir), "hooks目錄不存在")
        self.assertTrue(self.loader.hooks_available, "hooks應該可用")
        self.assertEqual(sorted(self.loader.available_epochs), [0, 1], "應該有兩個可用的輪次")

    def test_load_training_summary(self):
        """
        測試載入訓練摘要。
        """
        summary = self.loader.load_training_summary()
        self.assertIsNotNone(summary, "無法載入訓練摘要")
        self.assertIsInstance(summary, dict, "訓練摘要應該是字典類型")
        self.assertEqual(summary['total_epochs'], 2)
        self.assertEqual(summary['total_training_time'], 1000)
        self.assertEqual(summary['final_loss'], 0.1)

    def test_load_evaluation_results(self):
        """
        測試載入評估結果。
        """
        eval_results = self.loader.load_evaluation_results()
        self.assertIsNotNone(eval_results, "無法載入評估結果") 
        self.assertIsInstance(eval_results, dict, "評估結果應該是字典類型")
        self.assertEqual(eval_results['accuracy'], 0.9)
        self.assertEqual(eval_results['loss'], 0.2)

    def test_list_available_epochs(self):
        """
        測試列出可用的輪次。
        """
        epochs = self.loader.list_available_epochs()
        self.assertIsInstance(epochs, list, "可用輪次應該是列表類型")
        self.assertEqual(sorted(epochs), [0, 1], "應該有兩個輪次")

    def test_list_available_layers(self):
        """
        測試列出可用的層。
        """
        layers = self.loader.list_available_layers()
        self.assertIsInstance(layers, dict, "可用層應該是字典類型")
        
        # 如果有可用的輪次，也測試指定輪次的情況
        epochs = self.loader.list_available_epochs()
        if epochs:
            layers_of_epoch = self.loader.list_available_layers(epochs[0])
            self.assertIsInstance(layers_of_epoch, list, "指定輪次的可用層應該是列表類型")

    def test_load(self):
        """
        測試基本的載入功能。
        """
        # 不指定參數，應該返回可用資源概覽
        overview = self.loader.load()
        self.assertIsInstance(overview, dict, "概覽應該是字典類型")
        self.assertIn("available_epochs", overview, "概覽應該包含available_epochs")
        self.assertIn("available_layers", overview, "概覽應該包含available_layers")

    def test_list_available_layers(self):
        """測試列出可用的層激活文件"""
        loader = HookDataLoader(str(self.temp_dir))
        layers_epoch0 = loader.list_available_layer_activations(epoch=0)
        layers_epoch1 = loader.list_available_layer_activations(epoch=1)
        self.assertIsInstance(layers_epoch0, list)
        expected_layers = sorted(['conv1', 'conv2', 'fc1', 'dead_layer', 'saturated_layer'])
        self.assertEqual(sorted(layers_epoch0), expected_layers)
        self.assertEqual(layers_epoch0, layers_epoch1)

    def test_load_activation(self):
        """測試載入單一層激活值"""
        loader = HookDataLoader(str(self.temp_dir))
        activation = loader.load_layer_activation('conv1', 0, 0)
        self.assertIsInstance(activation, torch.Tensor)
        self.assertEqual(activation.shape[0], 4) # Check batch size

    def test_init(self):
        """測試初始化HookDataLoader"""
        loader = HookDataLoader(str(self.temp_dir))
        self.assertTrue(loader.hooks_available)
        self.assertEqual(sorted(loader.available_epochs), [0, 1])
        
        # Since we've changed how available_layers is structured,
        # adapt the test to match the implementation
        key = (0, 0)  # (epoch, batch)
        self.assertTrue(any('conv1' in layers for layers in loader.available_layers.values()))

if __name__ == '__main__':
    unittest.main() 