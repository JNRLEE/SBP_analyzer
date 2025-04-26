"""
# 測試中間層數據分析器功能
"""
import os
import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import logging

from analyzer.intermediate_data_analyzer import IntermediateDataAnalyzer
from data_loader.hook_data_loader import HookDataLoader
from metrics.layer_activity_metrics import (
    calculate_activation_statistics,
    calculate_activation_sparsity,
    calculate_effective_rank,
    detect_dead_neurons,
    calculate_layer_similarity
)


class MockHookDataLoader:
    """用於測試的模擬HookDataLoader"""
    
    def __init__(self, hook_dir="dummy_hooks"):
        self.hooks_dir = Path(hook_dir)
        self._layers = {
            (0, 0): ['conv1', 'conv2', 'fc1', 'saturated_layer', 'dead_layer', 'sparse_layer', 'uniform_layer'],
            (1, 0): ['conv1', 'conv2', 'fc1', 'saturated_layer', 'dead_layer', 'sparse_layer', 'uniform_layer']
        }
        self._epochs = [0, 1]
        self._batches = {0: [0], 1: [0]}
        
    def list_available_epochs(self):
        return self._epochs
        
    def list_available_batches(self, epoch):
        return self._batches.get(epoch, [])

    def list_available_layers(self, epoch, batch):
        return self._layers.get((epoch, batch), [])

    def load_layer_activation(self, layer_name, epoch, batch):
        if (epoch, batch) not in self._layers or layer_name not in self._layers[(epoch, batch)]:
            raise FileNotFoundError(f"Mock data not found for {layer_name} epoch {epoch} batch {batch}")
        # Generate mock data based on layer name convention
        if layer_name == 'conv1': return torch.randn(16, 32, 8, 8)
        if layer_name == 'conv2': return torch.randn(16, 64, 4, 4) * 0.1
        if layer_name == 'fc1': return torch.rand(16, 100) - 0.5
        if layer_name == 'saturated_layer': return torch.clamp(torch.rand(16, 50) + 0.8, max=1.0)
        if layer_name == 'dead_layer': return torch.zeros(16, 20)
        if layer_name == 'sparse_layer':
            t = torch.randn(16, 100)
            mask = torch.rand(16, 100) > 0.95
            return t * mask.float()
        if layer_name == 'uniform_layer': return torch.rand(16, 70)
        return torch.randn(16, 10)

    def get_activation_metadata(self, layer_name, epoch, batch):
        shape = tuple(self.load_layer_activation(layer_name, epoch, batch).shape)
        return {'shape': shape}


@pytest.fixture
def mock_loader():
    return MockHookDataLoader()


@pytest.fixture
def temp_experiment_dir():
    """創建臨時實驗目錄"""
    temp_dir = tempfile.mkdtemp()
    try:
        # 創建hooks目錄結構
        hooks_dir = os.path.join(temp_dir, 'hooks')
        os.makedirs(hooks_dir, exist_ok=True)
        
        # 創建epoch目錄
        for epoch in [0, 1]:
            epoch_dir = os.path.join(hooks_dir, f'epoch_{epoch}')
            os.makedirs(epoch_dir, exist_ok=True)
            
            # 添加一些測試激活值文件
            for layer_name in ['conv1', 'conv2', 'fc1']:
                if layer_name == 'conv1':
                    tensor = torch.randn(32, 64, 16, 16)
                elif layer_name == 'conv2':
                    tensor = torch.randn(32, 128, 8, 8)
                else:  # fc1
                    tensor = torch.randn(32, 256)
                
                file_path = os.path.join(epoch_dir, f'{layer_name}_activation_batch_0.pt')
                torch.save(tensor, file_path)
            
            # 添加死神經元層
            dead_tensor = torch.zeros(32, 100)
            dead_path = os.path.join(epoch_dir, 'dead_layer_activation_batch_0.pt')
            torch.save(dead_tensor, dead_path)
            
            # 添加飽和層
            saturated_tensor = torch.ones(32, 100)
            saturated_path = os.path.join(epoch_dir, 'saturated_layer_activation_batch_0.pt')
            torch.save(saturated_tensor, saturated_path)
        
        yield temp_dir
    finally:
        # 測試結束後刪除臨時目錄
        shutil.rmtree(temp_dir)


@pytest.fixture
def analyzer(tmp_path: Path):
    """Creates an IntermediateDataAnalyzer instance initialized with tmp_path
    and patches its hook_loader with MockHookDataLoader."""
    # Create a dummy hooks subdir as the real loader expects it
    (tmp_path / 'hooks').mkdir(exist_ok=True)

    analyzer_instance = IntermediateDataAnalyzer(experiment_dir=str(tmp_path))
    # Patch the internally created hook_loader with our mock loader
    analyzer_instance.hook_loader = MockHookDataLoader(hook_dir=str(tmp_path / 'hooks'))
    # Add the missing list_available_layer_activations method to the MOCK loader instance
    # This was added to the class definition before, but patching requires it on the instance too
    # (or ensuring the instance uses the modified class)
    def mock_list_activations(epoch):
        return analyzer_instance.hook_loader._layers.get((epoch, 0), []) # Access _layers of the mock instance
    analyzer_instance.hook_loader.list_available_layer_activations = mock_list_activations
        
    return analyzer_instance


def test_initialization(tmp_path):
    """Test initialization with a directory path."""
    # Create a dummy hooks subdir as the real loader expects it
    (tmp_path / 'hooks').mkdir(exist_ok=True)
    analyzer = IntermediateDataAnalyzer(experiment_dir=str(tmp_path))
    assert analyzer.experiment_dir == str(tmp_path)
    # Check if an internal HookDataLoader was created (or intended to be)
    assert isinstance(analyzer.hook_loader, HookDataLoader)
    # Check if results are initially empty dict (from BaseAnalyzer)
    assert analyzer.results == {}


def test_initialization_without_loader():
    with pytest.raises(TypeError): # Expect TypeError if hook_loader is missing
        IntermediateDataAnalyzer()


def test_analyze_basic_flow(analyzer, mock_loader):
    epochs_to_analyze = [0]
    layers_to_analyze = ['conv1', 'dead_layer']
    batch_idx_to_analyze = 0

    results = analyzer.analyze(epochs=epochs_to_analyze, layers=layers_to_analyze, batch_idx=batch_idx_to_analyze)

    assert results is not None
    assert 'layer_results' in results
    # Check for layer_relationships key instead of cross_layer_analysis
    assert 'layer_relationships' in results 

    # Check specific layer results exist using the new nested structure
    assert 'conv1' in results['layer_results']
    assert '0' in results['layer_results']['conv1']
    assert 'dead_layer' in results['layer_results']
    assert '0' in results['layer_results']['dead_layer']

    # Check structure of a layer result using the new nested structure
    conv1_result = results['layer_results']['conv1']['0']
    assert 'basic_statistics' in conv1_result
    # Distribution analysis is not directly calculated in _analyze_layer_activation
    # assert 'distribution_analysis' in conv1_result 
    # Activity analysis seems split (sparsity, dead_neurons, saturation)
    assert 'sparsity' in conv1_result 
    assert 'dead_neurons' in conv1_result
    assert 'saturation' in conv1_result
    assert 'effective_rank' in conv1_result

    # Check basic stats values (rough checks)
    assert 'mean' in conv1_result['basic_statistics']
    assert 'std' in conv1_result['basic_statistics']
    # Sparsity is now top-level
    assert conv1_result['sparsity'] < 0.1 # conv1 is dense

    # Check dead layer stats using the new nested structure
    dead_result = results['layer_results']['dead_layer']['0']
    assert dead_result['basic_statistics']['mean'] == 0
    assert dead_result['basic_statistics']['std'] == 0
    assert dead_result['sparsity'] == 1.0
    # Check within dead_neurons dict using the correct key 'dead_ratio'
    assert 'dead_neurons' in dead_result
    assert dead_result['dead_neurons'].get('error') is None # Ensure no error occurred
    assert dead_result['dead_neurons']['dead_ratio'] == 1.0 


def test_analyze_specific_analyses(analyzer):
    results = analyzer.analyze(epochs=[0], layers=['conv1'], batch_idx=0)

    # Access result using new nested structure
    layer_result = results['layer_results']['conv1']['0'] 
    assert 'basic_statistics' in layer_result
    # Check for specific activity metrics instead of a single 'activity_analysis' key
    assert 'sparsity' in layer_result
    assert 'saturation' in layer_result
    assert 'dead_neurons' in layer_result
    # Distribution analysis not directly present
    # assert 'distribution_analysis' not in layer_result 
    assert 'effective_rank' in layer_result # Effective rank is calculated
    # Similarity is in 'layer_relationships', not checked here directly
    # assert not results.get('layer_relationships') 

    # Check that conv2 analysis failed gracefully (layer key should be absent)
    assert 'conv2' not in results['layer_results'] 
    # Check log message - match the actual format from analyze method
    # expected_log = f"分析層 conv2 在輪次 0 時出錯: Simulated load error for conv2"
    # assert expected_log in caplog.text
    # assert "Simulated load error for conv2" in caplog.text # Keep this simpler check too


def test_analyze_with_mock_loader(analyzer, mock_loader):
    # Analyze all layers defined in the mock loader for epoch 0
    results = analyzer.analyze(epochs=[0], batch_idx=0)

    assert 'layer_results' in results
    # Check the layer names (keys of the outer dict)
    expected_layers = mock_loader.list_available_layers(0, 0)
    analyzed_layers = set(results['layer_results'].keys())
    assert analyzed_layers == set(expected_layers)

    # Check results for a specific layer and epoch
    conv1_result_ep0 = results['layer_results']['conv1']['0']
    assert 'basic_statistics' in conv1_result_ep0
    # Distribution type checks are removed as this metric isn't calculated directly
    
    # --- REMOVE ERRONEOUS ASSERTION AND LOG CHECKS START --- 
    # Check that conv2 analysis failed gracefully (layer key should be absent)
    # assert 'conv2' not in results['layer_results'] 
    # --- REMOVE ERRONEOUS ASSERTION AND LOG CHECKS END --- 


def test_analyze_layer_subset(analyzer):
    layers_to_analyze = ['conv1', 'fc1']
    results = analyzer.analyze(epochs=[0, 1], layers=layers_to_analyze, batch_idx=0)

    # Check outer keys (layer names)
    assert set(results['layer_results'].keys()) == set(layers_to_analyze)
    # Check inner keys (epochs as strings) for each layer
    assert set(results['layer_results']['conv1'].keys()) == {'0', '1'}
    assert set(results['layer_results']['fc1'].keys()) == {'0', '1'}


def test_analyze_epoch_subset(analyzer):
    epochs_to_analyze = [1]
    results = analyzer.analyze(epochs=epochs_to_analyze, batch_idx=0) # Analyze all layers for epoch 1

    # Check that only epoch '1' results are present for analyzed layers
    for layer_name, epoch_data in results['layer_results'].items():
        assert set(epoch_data.keys()) == {'1'}


def test_effective_rank_calculation(analyzer):
    # Ensure the calculation is performed by checking the result key
    results = analyzer.analyze(epochs=[0], layers=['conv1'], batch_idx=0)
    # Use new nested structure to access result
    assert 'effective_rank' in results['layer_results']['conv1']['0']


def test_similarity_calculation(analyzer):
    # Analyze multiple layers to trigger similarity calculation
    layers_to_analyze = ['conv1', 'conv2', 'fc1']
    results = analyzer.analyze(epochs=[0], layers=layers_to_analyze, batch_idx=0)
    
    # Check if layer_relationships key exists
    assert 'layer_relationships' in results
    relationships = results['layer_relationships']
    
    # Check if similarity matrix was calculated (assuming keys like layer1_vs_layer2_epochX)
    key_found = False
    expected_pairs = [('conv1', 'conv2'), ('conv1', 'fc1'), ('conv2', 'fc1')]
    for l1, l2 in expected_pairs:
        if f'{l1}_vs_{l2}_epoch_0' in relationships or f'{l2}_vs_{l1}_epoch_0' in relationships:
            key_found = True
            # Optionally check the content structure
            rel_key = f'{l1}_vs_{l2}_epoch_0' if f'{l1}_vs_{l2}_epoch_0' in relationships else f'{l2}_vs_{l1}_epoch_0'
            assert 'cosine_similarity' in relationships[rel_key] 
            break # Found one example, assume calculation ran
    assert key_found, "Expected similarity relationships not found in results"


def test_analyze_error_handling_layer_load(analyzer, mock_loader, caplog):
    # Mock load_layer_activation ON THE ANALYZER'S INTERNAL LOADER
    # original_load = mock_loader.load_layer_activation <--- Don't use the separate mock_loader fixture
    original_load = analyzer.hook_loader.load_layer_activation # <-- Get original from analyzer's loader
    
    def faulty_load(*args, **kwargs):
        layer_name = kwargs.get('layer_name') or args[0]
        # --- DEBUG: Print inside mock --- 
        # print(f"DEBUG: faulty_load called for layer: {layer_name}")
        # --- END DEBUG --- 
        if layer_name == 'conv2':
            # print(f"DEBUG: Raising error for {layer_name}") # Add print before raising
            raise ValueError("Simulated load error for conv2")
        return original_load(*args, **kwargs)

    # --- Patch the ANALYZER's loader instance --- 
    analyzer.hook_loader.load_layer_activation = faulty_load 
    # mock_loader.load_layer_activation = faulty_load <--- Don't patch the separate fixture

    with caplog.at_level(logging.ERROR):
        results = analyzer.analyze(epochs=[0], layers=['conv1', 'conv2'], batch_idx=0)

        # Check that conv1 results are still present using new structure
        assert 'conv1' in results['layer_results']
        assert '0' in results['layer_results']['conv1']
        # Check that conv2 analysis failed gracefully (layer key should be absent)
        
        # --- DEBUG: Print keys before assertion --- 
        # print(f"DEBUG: Keys in results['layer_results']: {results['layer_results'].keys()}")
        # --- END DEBUG --- 
        
        assert 'conv2' not in results['layer_results']
        
        # Check log message - match the actual format from analyze method
        # Note: The exact error string (e) might include traceback info, 
        # so checking for key substrings is safer.
        assert "分析層 conv2 在輪次 0 時出錯" in caplog.text 
        assert "Simulated load error for conv2" in caplog.text

    # Restore original method on the ANALYZER's loader
    analyzer.hook_loader.load_layer_activation = original_load 
    # mock_loader.load_layer_activation = original_load <--- Not needed


def test_get_summary(analyzer):
    analyzer.analyze(epochs=[0], layers=['conv1', 'dead_layer'], batch_idx=0)
    report = analyzer.generate_report()

    assert isinstance(report, dict)
    # Check for actual keys returned by generate_report
    assert 'summary' in report 
    assert 'problematic_layers' in report 
    assert 'layer_statistics' in report 
    assert 'activation_dynamics' in report 
    assert 'layer_relationships' in report
    
    # Check if problematic layers were detected
    assert len(report['problematic_layers']) > 0
    assert any(p['layer_name'] == 'dead_layer' for p in report['problematic_layers'])


def test_get_summary_before_analyze(analyzer, caplog):
    report = analyzer.generate_report()
    assert isinstance(report, dict)
    # Check if report sections are empty or default, and remove caplog check
    assert report.get('layer_statistics') == {} 
    assert report.get('activation_dynamics') == {}
    assert report.get('layer_relationships') == {}
    assert report.get('problematic_layers') == [] # get_problematic_layers returns [] if no stats
    assert report.get('summary') is not None # Summary key should exist

# Add tests for edge cases: no layers found, no epochs found, empty tensors etc. 