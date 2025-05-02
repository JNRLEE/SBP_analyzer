#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Testing Script.

This script is used to test the SBP_analyzer visualization features, including model parameter distribution chart,
layer activation similarity matrix chart, and model performance confusion matrix.
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path

# Add project root directory to path
project_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_dir))

# Ensure modules can be correctly imported
try:
    from analyzer.model_structure_analyzer import ModelStructureAnalyzer
    from analyzer.intermediate_data_analyzer import IntermediateDataAnalyzer
    from metrics.layer_activity_metrics import compute_confusion_matrix, compute_activation_statistics
    from visualization.performance_plots import plot_confusion_matrix
    from visualization.model_structure_plots import plot_model_params_distribution
    from visualization.layer_activity_plots import plot_cosine_similarity_matrix, plot_activation_statistics
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def test_model_params_distribution(experiment_dir, output_dir):
    """
    Test model parameter distribution chart.
    
    Args:
        experiment_dir (str): Experiment results directory
        output_dir (str): Output directory path
    """
    print("Testing model parameter distribution chart...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model structure analyzer
    analyzer = ModelStructureAnalyzer(experiment_dir)
    
    # Load model structure - correct call
    analyzer.load_model_structure()
    
    # Visualize parameter distribution
    output_path = os.path.join(output_dir, "model_params_distribution.png")
    analyzer.visualize_parameter_distribution(output_path)
    
    print(f"Generated model parameter distribution chart: {output_path}")

def test_cosine_similarity_matrix(output_dir):
    """
    Test cosine similarity matrix chart.
    
    Args:
        output_dir (str): Output directory path
    """
    print("Testing cosine similarity matrix chart...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample embeddings for testing
    embeddings = np.random.randn(10, 50)  # 10 samples, 50 dimensions
    
    # Visualize cosine similarity matrix
    output_path = os.path.join(output_dir, "cosine_similarity_matrix.png")
    plot_cosine_similarity_matrix(embeddings, output_path=output_path)
    
    print(f"Generated cosine similarity matrix chart: {output_path}")

def test_confusion_matrix(output_dir):
    """
    Test confusion matrix chart.
    
    Args:
        output_dir (str): Output directory path
    """
    print("Testing confusion matrix chart...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample confusion matrix for testing
    classes = 4
    y_true = np.random.randint(0, classes, size=100)
    y_pred = np.random.randint(0, classes, size=100)
    
    # Compute confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=classes)
    
    # Class names for display
    class_names = [f"Class {i}" for i in range(classes)]
    
    # Visualize confusion matrix
    output_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names=class_names, output_path=output_path)
    print(f"Generated confusion matrix chart: {output_path}")
    
    # Visualize normalized confusion matrix
    output_path = os.path.join(output_dir, "confusion_matrix_normalized.png")
    plot_confusion_matrix(cm, class_names=class_names, normalize='true', 
                       title="Normalized Confusion Matrix by True Label", 
                       output_path=output_path)
    print(f"Generated normalized confusion matrix chart: {output_path}")

def test_intermediate_data_analyzer(output_dir):
    """
    Test intermediate data analyzer.
    
    Args:
        output_dir (str): Output directory path
    """
    print("Testing intermediate data analyzer...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a test experiment directory
    experiment_dir = os.path.join(output_dir, "test_experiment")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create simulated activation data
    # 2D tensor (assumed to be fully connected layer activations)
    fc_activations = np.random.randn(10, 100)
    
    # 4D tensor (assumed to be convolutional layer activations)
    conv_activations = np.random.randn(10, 16, 8, 8)
    
    # Build activations dictionary
    activations = {
        "conv1": conv_activations,
        "fc1": fc_activations
    }
    
    # Initialize analyzer
    analyzer = IntermediateDataAnalyzer(experiment_dir)
    
    # Manually set activations
    analyzer.activations = activations
    
    # Test cosine similarity matrix calculation and visualization
    embeddings = fc_activations.reshape(10, -1)  # Flatten for visualization
    output_path = os.path.join(output_dir, "layer_cosine_similarity.png")
    plot_cosine_similarity_matrix(embeddings, output_path=output_path)
    print(f"Generated layer activation cosine similarity matrix chart: {output_path}")
    
    # Test layer comparison
    if hasattr(analyzer, 'compare_layers'):
        output_path = os.path.join(output_dir, "layer_comparison.png")
        analyzer.compare_layers("conv1", "fc1", output_path=output_path)
        print(f"Generated layer comparison chart: {output_path}")
    
    # Test activation statistics
    for layer_name in activations.keys():
        layer_activations = activations[layer_name]
        flatten_activations = layer_activations.reshape(layer_activations.shape[0], -1)
        
        stats = compute_activation_statistics(flatten_activations)
        
        output_path = os.path.join(output_dir, f"{layer_name}_statistics.png")
        plot_activation_statistics(stats, layer_name=layer_name, output_path=output_path)
        print(f"Generated {layer_name} layer statistics chart: {output_path}")
    
    print("Intermediate data analyzer test completed!")

def main():
    """Main function to run all visualization tests."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test visualization features")
    parser.add_argument('--experiment_dir', type=str, 
                       default='results/custom_audio_fcnn_classification_20250501_113135',
                       help='Path to experiment directory containing model data')
    parser.add_argument('--output_dir', type=str, 
                       default='results/visualization_test',
                       help='Output directory for visualization files')
    
    args = parser.parse_args()
    
    # Run tests
    test_model_params_distribution(args.experiment_dir, args.output_dir)
    test_cosine_similarity_matrix(args.output_dir)
    test_confusion_matrix(args.output_dir)
    test_intermediate_data_analyzer(args.output_dir)
    
    print(f"\nAll tests completed! Output files are in: {args.output_dir}")

if __name__ == "__main__":
    main() 