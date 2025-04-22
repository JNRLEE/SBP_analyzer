#!/usr/bin/env python
# 這個腳本示範如何使用訓練動態分析功能來評估機器學習模型的訓練過程

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.performance_metrics import (
    calculate_overfitting_metric,
    calculate_convergence_speed,
    detect_oscillation,
    calculate_training_efficiency,
    calculate_training_dynamics_metrics
)

def simulate_training_history():
    """
    模擬一個訓練歷史，生成訓練損失和驗證損失
    """
    epochs = 100
    
    # Simulating training loss (decreasing with some noise)
    train_loss = 5.0 * np.exp(-0.05 * np.arange(epochs)) + 0.3 * np.random.rand(epochs)
    
    # Simulating validation loss (decreasing then increasing - overfitting)
    val_loss = 5.0 * np.exp(-0.04 * np.arange(epochs)) + 0.5 * np.random.rand(epochs)
    
    # Add overfitting effect after epoch 50
    val_loss[50:] += 0.03 * np.arange(50) 
    
    return train_loss, val_loss, epochs

def main():
    # Simulate training data
    train_loss, val_loss, epochs = simulate_training_history()
    
    # Calculate individual metrics
    overfitting = calculate_overfitting_metric(train_loss, val_loss)
    convergence_speed = calculate_convergence_speed(train_loss)
    oscillation = detect_oscillation(train_loss)
    efficiency = calculate_training_efficiency(train_loss, val_loss)
    
    # Calculate all metrics at once
    all_metrics = calculate_training_dynamics_metrics(train_loss, val_loss)
    
    # Print results
    print("Individual Metrics:")
    print(f"Overfitting Metric: {overfitting:.4f}")
    print(f"Convergence Speed: {convergence_speed}")
    print(f"Oscillation Detected: {oscillation}")
    print(f"Training Efficiency: {efficiency:.4f}")
    
    print("\nAll Metrics at Once:")
    for metric, value in all_metrics.items():
        print(f"{metric}: {value}")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), train_loss, label='Training Loss')
    plt.plot(range(epochs), val_loss, label='Validation Loss')
    
    # Mark convergence point if detected
    if convergence_speed is not None:
        plt.axvline(x=convergence_speed, color='g', linestyle='--', 
                   label=f'Convergence at epoch {convergence_speed}')
    
    plt.title('Training Dynamics Analysis')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_dynamics.png')
    plt.close()
    
    print("\nTraining dynamics plot saved as 'training_dynamics.png'")

if __name__ == "__main__":
    main() 