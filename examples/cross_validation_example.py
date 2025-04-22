#!/usr/bin/env python
# 這個腳本示範如何使用交叉驗證來評估模型性能，包括多種評估指標的計算

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.performance_metrics import (
    calculate_rmse,
    calculate_mae,
    calculate_mape,
    calculate_r2
)

def main():
    # Load a regression dataset
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    # Initialize StandardScaler
    scaler = StandardScaler()
    
    # Initialize the model
    model = Ridge(alpha=1.0)
    
    # Setup cross-validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Metrics storage
    rmse_scores = []
    mae_scores = []
    mape_scores = []
    r2_scores = []
    
    # Perform cross-validation
    fold = 1
    for train_index, test_index in kf.split(X):
        # Split data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        rmse = calculate_rmse(y_test, y_pred)
        mae = calculate_mae(y_test, y_pred)
        mape = calculate_mape(y_test, y_pred)
        r2 = calculate_r2(y_test, y_pred)
        
        # Store metrics
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        mape_scores.append(mape)
        r2_scores.append(r2)
        
        # Print fold results
        print(f"Fold {fold} Results:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  MAPE: {mape:.4f}")
        print(f"  R²: {r2:.4f}")
        print("-" * 40)
        
        fold += 1
    
    # Calculate and print average metrics
    print("Average Cross-Validation Metrics:")
    print(f"  RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"  MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"  MAPE: {np.mean(mape_scores):.4f} ± {np.std(mape_scores):.4f}")
    print(f"  R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    
    # Plot metrics across folds
    metrics = {
        'RMSE': rmse_scores,
        'MAE': mae_scores,
        'MAPE': mape_scores,
        'R²': r2_scores
    }
    
    plt.figure(figsize=(12, 8))
    
    for i, (metric_name, values) in enumerate(metrics.items(), 1):
        plt.subplot(2, 2, i)
        plt.bar(range(1, n_splits + 1), values)
        plt.axhline(y=np.mean(values), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(values):.4f}')
        plt.xlabel('Fold')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} Across Folds')
        plt.xticks(range(1, n_splits + 1))
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cross_validation_metrics.png')
    plt.close()
    
    print("\nCross-validation metrics plot saved as 'cross_validation_metrics.png'")

if __name__ == "__main__":
    main() 