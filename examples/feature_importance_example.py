#!/usr/bin/env python
# 這個腳本示範如何分析特徵重要性，使用多種方法包括隨機森林、排列重要性和SHAP值

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import shap

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_random_forest_importance(X_train, X_test, y_train, feature_names):
    """使用RandomForest的內建特徵重要性進行分析"""
    
    # Train a random forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Get feature importances
    importances = rf.feature_importances_
    
    # Create a dataframe for better visualization
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance_df['Feature'].values[:15], 
             feature_importance_df['Importance'].values[:15])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Random Forest Feature Importance (Top 15)')
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png')
    plt.close()
    
    print("Random Forest Feature Importance (Top 10):")
    for i, row in feature_importance_df.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    return rf, feature_importance_df

def analyze_permutation_importance(model, X_test, y_test, feature_names):
    """使用排列重要性方法分析特徵重要性"""
    
    # Calculate permutation importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    
    # Create a dataframe for better visualization
    perm_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean
    })
    
    # Sort by importance
    perm_importance_df = perm_importance_df.sort_values('Importance', ascending=False)
    
    # Plot permutation importances
    plt.figure(figsize=(12, 8))
    plt.barh(perm_importance_df['Feature'].values[:15], 
             perm_importance_df['Importance'].values[:15])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Permutation Feature Importance (Test Set, Top 15)')
    plt.tight_layout()
    plt.savefig('permutation_importance.png')
    plt.close()
    
    print("\nPermutation Feature Importance (Top 10):")
    for i, row in perm_importance_df.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    return perm_importance_df

def analyze_shap_values(model, X_train, feature_names):
    """使用SHAP值分析特徵重要性和解釋模型預測"""
    
    # Create a SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for a subset of the training data
    sample_size = min(100, X_train.shape[0])
    sample_indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
    X_sample = X_train[sample_indices]
    
    # Get SHAP values
    shap_values = explainer.shap_values(X_sample)
    
    # If model is a classifier with binary target, use only the second class SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Calculate mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create a dataframe for better visualization
    shap_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Importance': mean_abs_shap
    })
    
    # Sort by importance
    shap_importance_df = shap_importance_df.sort_values('SHAP_Importance', ascending=False)
    
    # Plot SHAP summary
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.close()
    
    # Plot SHAP bar plot for top features
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig('shap_bar_plot.png')
    plt.close()
    
    print("\nSHAP Feature Importance (Top 10):")
    for i, row in shap_importance_df.head(10).iterrows():
        print(f"  {row['Feature']}: {row['SHAP_Importance']:.4f}")
    
    return shap_importance_df

def compare_importance_methods(rf_importance, perm_importance, shap_importance):
    """比較不同特徵重要性方法的結果"""
    
    # Merge the dataframes
    comparison_df = rf_importance[['Feature', 'Importance']].rename(
        columns={'Importance': 'Random Forest'})
    comparison_df = comparison_df.merge(
        perm_importance[['Feature', 'Importance']].rename(
            columns={'Importance': 'Permutation'}),
        on='Feature')
    comparison_df = comparison_df.merge(
        shap_importance[['Feature', 'SHAP_Importance']].rename(
            columns={'SHAP_Importance': 'SHAP'}),
        on='Feature')
    
    # Get top 10 features by average rank
    comparison_df['RF_Rank'] = comparison_df['Random Forest'].rank(ascending=False)
    comparison_df['Perm_Rank'] = comparison_df['Permutation'].rank(ascending=False)
    comparison_df['SHAP_Rank'] = comparison_df['SHAP'].rank(ascending=False)
    comparison_df['Avg_Rank'] = (comparison_df['RF_Rank'] + 
                                 comparison_df['Perm_Rank'] + 
                                 comparison_df['SHAP_Rank']) / 3
    
    # Sort by average rank
    comparison_df = comparison_df.sort_values('Avg_Rank')
    
    # Select top 10 features
    top_features = comparison_df.head(10)['Feature'].values
    
    # Normalize importance values for better comparison
    comparison_df['RF_Norm'] = comparison_df['Random Forest'] / comparison_df['Random Forest'].max()
    comparison_df['Perm_Norm'] = comparison_df['Permutation'] / comparison_df['Permutation'].max()
    comparison_df['SHAP_Norm'] = comparison_df['SHAP'] / comparison_df['SHAP'].max()
    
    # Filter for top features and select normalized values for plotting
    plot_df = comparison_df[comparison_df['Feature'].isin(top_features)][
        ['Feature', 'RF_Norm', 'Perm_Norm', 'SHAP_Norm']]
    plot_df = plot_df.set_index('Feature')
    
    # Plot comparison
    plt.figure(figsize=(14, 10))
    plot_df.plot(kind='bar', width=0.8)
    plt.title('Feature Importance Comparison (Normalized, Top 10)')
    plt.xlabel('Feature')
    plt.ylabel('Normalized Importance')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Method')
    plt.tight_layout()
    plt.savefig('importance_comparison.png')
    plt.close()
    
    print("\nTop 10 Features by Average Rank:")
    for i, (idx, row) in enumerate(comparison_df.head(10).iterrows(), 1):
        print(f"  {i}. {row['Feature']} (Avg Rank: {row['Avg_Rank']:.2f})")
    
    return comparison_df

def main():
    # Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    
    print("Analyzing feature importance using multiple methods...\n")
    
    # Analyze using Random Forest feature importance
    rf_model, rf_importance = analyze_random_forest_importance(
        X_train, X_test, y_train, feature_names)
    
    # Analyze using permutation importance
    perm_importance = analyze_permutation_importance(
        rf_model, X_test, y_test, feature_names)
    
    # Analyze using SHAP values
    shap_importance = analyze_shap_values(rf_model, X_train, feature_names)
    
    # Compare different importance methods
    comparison = compare_importance_methods(
        rf_importance, perm_importance, shap_importance)
    
    print("\nFeature importance analysis completed.")
    print("The following files have been saved:")
    print("  - rf_feature_importance.png")
    print("  - permutation_importance.png")
    print("  - shap_summary.png")
    print("  - shap_bar_plot.png")
    print("  - importance_comparison.png")

if __name__ == "__main__":
    main() 