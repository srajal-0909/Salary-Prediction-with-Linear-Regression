"""
End-to-End Machine Learning Pipeline for Salary Prediction
Tasks executed:
1. Load dataset with pandas
2. Inspect columns, data types, missing values, basic distributions
3. Identify target (Salary) and input features (YearsExperience)
4. Train/Test split before fitting preprocessing (prevent data leakage)
5. Establish a simple baseline (DummyRegressor)
6. Fit preprocessing & train LinearRegression pipeline
7. Evaluate using MAE, RMSE, R^2, and MAPE
8. Detailed Residual & Error analysis across target salary ranges
9. Compare training vs test performance (overfitting/underfitting analysis)
10. Save pipeline (joblib) & metadata (JSON)
"""

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for generating images
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def run_pipeline():
    print("=" * 70)
    print("STEP 1: Loading Dataset using Pandas")
    print("=" * 70)
    
    data_path = "Salary_dataset.csv"
    df = pd.read_csv(data_path)
    
    # Remove unnamed index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    elif '' in df.columns:
        df = df.drop(columns=[''])
    
    print(f"Dataset successfully loaded from: {data_path}")
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    print("First 5 rows:")
    print(df.head())
    
    print("\n" + "=" * 70)
    print("STEP 2: Inspecting Columns, Data Types, Missing Values & Distributions")
    print("=" * 70)
    print("\n[Column Names & Data Types]")
    print(df.dtypes)
    
    print("\n[Missing Values Check]")
    missing_vals = df.isnull().sum()
    print(missing_vals)
    
    print("\n[Summary Statistics & Distributions]")
    stats = df.describe().T
    stats['median'] = df.median()
    stats['skewness'] = df.skew()
    stats['kurtosis'] = df.kurt()
    print(stats[['count', 'mean', 'std', 'min', '25%', 'median', '75%', 'max', 'skewness']])
    
    print("\n" + "=" * 70)
    print("STEP 3: Identifying Target & Input Features")
    print("=" * 70)
    feature_cols = ['YearsExperience']
    target_col = 'Salary'
    
    X = df[feature_cols]
    y = df[target_col]
    print(f"Feature matrix X shape: {X.shape} (Features: {feature_cols})")
    print(f"Target vector y shape: {y.shape} (Target: {target_col})")
    
    print("\n" + "=" * 70)
    print("STEP 4: Train / Test Split (Strictly before fitting preprocessing)")
    print("=" * 70)
    # 80/20 train/test split with fixed seed for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Training set: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing set:  X_test={X_test.shape}, y_test={y_test.shape}")
    print(f"Train salary range: ${y_train.min():,.2f} to ${y_train.max():,.2f}")
    print(f"Test salary range:  ${y_test.min():,.2f} to ${y_test.max():,.2f}")
    
    print("\n" + "=" * 70)
    print("STEP 5: Establishing Simple Baseline Model")
    print("=" * 70)
    # Baseline strategy: predict the mean of the training target
    baseline_model = DummyRegressor(strategy='mean')
    baseline_model.fit(X_train, y_train)
    y_test_baseline = baseline_model.predict(X_test)
    
    baseline_mae = mean_absolute_error(y_test, y_test_baseline)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, y_test_baseline))
    baseline_r2 = r2_score(y_test, y_test_baseline)
    
    print(f"Baseline Strategy: DummyRegressor (Mean of training salaries = ${baseline_model.constant_[0][0]:,.2f})")
    print(f"Baseline Test MAE:  ${baseline_mae:,.2f}")
    print(f"Baseline Test RMSE: ${baseline_rmse:,.2f}")
    print(f"Baseline Test R^2:  {baseline_r2:.4f} (Benchmark: 0.0)")
    
    print("\n" + "=" * 70)
    print("STEP 6: Building Preprocessing & Training Linear Regression Pipeline")
    print("=" * 70)
    # Define an end-to-end scikit-learn pipeline: Imputer -> LinearRegression
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('regressor', LinearRegression())
    ])
    
    # Fit strictly on X_train, y_train
    pipeline.fit(X_train, y_train)
    
    reg = pipeline.named_steps['regressor']
    slope = reg.coef_[0]
    intercept = reg.intercept_
    
    print("Trained Scikit-Learn Pipeline successfully!")
    print(f"Model Equation: Salary = ${intercept:,.2f} + (${slope:,.2f} * YearsExperience)")
    print(f"Interpretation: For each additional year of experience, expected salary increases by ~${slope:,.2f}")
    
    print("\n" + "=" * 70)
    print("STEP 7 & 9: Evaluation (Train vs Test Performance) with MAE, RMSE, R^2")
    print("=" * 70)
    # Predict on Train and Test
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
    
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    
    perf_df = pd.DataFrame({
        'Metric': ['MAE ($)', 'RMSE ($)', 'R^2 Score', 'MAPE (%)'],
        'Training Set': [f"${train_mae:,.2f}", f"${train_rmse:,.2f}", f"{train_r2:.4f}", f"{train_mape:.2f}%"],
        'Testing Set': [f"${test_mae:,.2f}", f"${test_rmse:,.2f}", f"{test_r2:.4f}", f"{test_mape:.2f}%"],
        'Baseline (Test)': [f"${baseline_mae:,.2f}", f"${baseline_rmse:,.2f}", f"{baseline_r2:.4f}", "N/A"]
    })
    print(perf_df.to_string(index=False))
    
    # Overfitting/Generalization check
    r2_diff = abs(train_r2 - test_r2)
    print(f"\nGeneralization Delta (|Train R^2 - Test R^2|): {r2_diff:.4f}")
    if r2_diff < 0.05:
        print("Verdict: The model generalizes exceptionally well with no signs of overfitting or high variance.")
    else:
        print("Verdict: Slight variance between train and test scores.")
        
    print(f"Error Reduction vs Baseline: {(1 - test_mae/baseline_mae)*100:.1f}% reduction in MAE!")
    
    print("\n" + "=" * 70)
    print("STEP 8: Detailed Residuals and Error Inspection across Target Ranges")
    print("=" * 70)
    residuals_train = y_train - y_train_pred
    residuals_test = y_test - y_test_pred
    
    print(f"Train Residuals: Mean = ${residuals_train.mean():.2f}, Std Dev = ${residuals_train.std():.2f}")
    print(f"Test Residuals:  Mean = ${residuals_test.mean():.2f}, Std Dev = ${residuals_test.std():.2f}")
    
    # Error across target salary ranges
    test_analysis_df = pd.DataFrame({
        'YearsExperience': X_test['YearsExperience'],
        'ActualSalary': y_test,
        'PredictedSalary': y_test_pred,
        'Residual': residuals_test,
        'AbsoluteError': np.abs(residuals_test),
        'PercentageError': np.abs(residuals_test) / y_test * 100
    })
    
    # Define Salary Tiers
    bins = [0, 60000, 100000, np.inf]
    labels = ['Entry Level (<$60k)', 'Mid Level ($60k-$100k)', 'Senior Level (>$100k)']
    test_analysis_df['SalaryTier'] = pd.cut(test_analysis_df['ActualSalary'], bins=bins, labels=labels)
    
    tier_summary = test_analysis_df.groupby('SalaryTier', observed=False).agg(
        Count=('ActualSalary', 'count'),
        Mean_Actual=('ActualSalary', 'mean'),
        Mean_Predicted=('PredictedSalary', 'mean'),
        MAE=('AbsoluteError', 'mean'),
        RMSE=('Residual', lambda r: np.sqrt(np.mean(r**2))),
        Mean_MAPE=('PercentageError', 'mean')
    ).reset_index()
    
    print("\nError Breakdown Across Target Salary Tiers:")
    print(tier_summary.to_string(index=False))
    
    # Generate Visualizations
    os.makedirs("reports", exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. Regression Fit Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train['YearsExperience'], y_train, color='#3B82F6', alpha=0.7, label='Train Data (80%)', s=50)
    plt.scatter(X_test['YearsExperience'], y_test, color='#EF4444', marker='^', alpha=0.9, label='Test Data (20%)', s=70)
    
    x_range = np.linspace(X['YearsExperience'].min() - 0.5, X['YearsExperience'].max() + 0.5, 200).reshape(-1, 1)
    y_line = pipeline.predict(pd.DataFrame(x_range, columns=['YearsExperience']))
    plt.plot(x_range, y_line, color='#1E293B', linewidth=2.5, label=f'Fit: y = {intercept:,.0f} + {slope:,.0f}x')
    
    # Add 95% approximate prediction interval band (+- 1.96 * test_rmse)
    plt.fill_between(x_range.flatten(), y_line - 1.96 * test_rmse, y_line + 1.96 * test_rmse,
                     color='#3B82F6', alpha=0.12, label='95% Prediction Interval (+-1.96 RMSE)')
    
    plt.title('Salary vs Years of Experience (Regression Fit & Test Evaluation)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Years of Experience', fontsize=12)
    plt.ylabel('Salary (USD $)', fontsize=12)
    plt.legend(frameon=True, facecolor='white', loc='upper left')
    plt.tight_layout()
    plt.savefig('reports/regression_fit.png', dpi=300)
    plt.close()
    print("\nSaved plot: reports/regression_fit.png")
    
    # 2. Residual Diagnostics Plot (Homoscedasticity & Normality)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Residuals vs Predicted
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[0].scatter(y_train_pred, residuals_train, color='#3B82F6', alpha=0.6, label='Train Residuals', s=45)
    axes[0].scatter(y_test_pred, residuals_test, color='#EF4444', marker='^', alpha=0.85, label='Test Residuals', s=60)
    axes[0].set_title('Residuals vs Fitted Values (Homoscedasticity Check)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicted Salary ($)', fontsize=11)
    axes[0].set_ylabel('Residual (Actual - Predicted) ($)', fontsize=11)
    axes[0].legend()
    
    # Residual Distribution
    sns.histplot(residuals_train, kde=True, color='#3B82F6', ax=axes[1], label='Train Residuals', stat='density', bins=12)
    sns.histplot(residuals_test, kde=True, color='#EF4444', ax=axes[1], label='Test Residuals', stat='density', bins=8)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_title('Residual Distribution (Normality Check)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Residual ($)', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('reports/residual_analysis.png', dpi=300)
    plt.close()
    print("Saved plot: reports/residual_analysis.png")
    
    # 3. Error by Target Range Plot
    plt.figure(figsize=(9, 5))
    bar_positions = np.arange(len(tier_summary))
    bar_width = 0.35
    
    plt.bar(bar_positions - bar_width/2, tier_summary['MAE'], width=bar_width, color='#3B82F6', label='MAE ($)')
    plt.bar(bar_positions + bar_width/2, tier_summary['RMSE'], width=bar_width, color='#F59E0B', label='RMSE ($)')
    
    plt.xticks(bar_positions, tier_summary['SalaryTier'], fontsize=11)
    plt.ylabel('Error Amount ($)', fontsize=12)
    plt.title('Prediction Error (MAE & RMSE) across Salary Tiers', fontsize=14, fontweight='bold', pad=12)
    plt.legend()
    
    for i, v in enumerate(tier_summary['MAE']):
        if not np.isnan(v):
            plt.text(i - bar_width/2, v + 150, f"${v:,.0f}", ha='center', fontsize=10, fontweight='bold')
    for i, v in enumerate(tier_summary['RMSE']):
        if not np.isnan(v):
            plt.text(i + bar_width/2, v + 150, f"${v:,.0f}", ha='center', fontsize=10, fontweight='bold')
            
    plt.tight_layout()
    plt.savefig('reports/error_by_target_range.png', dpi=300)
    plt.close()
    print("Saved plot: reports/error_by_target_range.png")
    
    print("\n" + "=" * 70)
    print("STEP 10 & 11: Saving Trained Model & Metadata")
    print("=" * 70)
    os.makedirs("models", exist_ok=True)
    model_save_path = "models/salary_model.joblib"
    joblib.dump(pipeline, model_save_path)
    print(f"Serialized trained pipeline saved to: {model_save_path}")
    
    metadata = {
        "model_name": "Linear Regression Salary Predictor",
        "created_at": datetime.now().isoformat(),
        "framework": "scikit-learn",
        "features": feature_cols,
        "target": target_col,
        "coefficients": {
            "intercept": float(intercept),
            "slope": float(slope)
        },
        "dataset_statistics": {
            "total_samples": int(len(df)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "years_min": float(df['YearsExperience'].min()),
            "years_max": float(df['YearsExperience'].max()),
            "salary_min": float(df['Salary'].min()),
            "salary_max": float(df['Salary'].max())
        },
        "performance_metrics": {
            "train": {
                "mae": float(train_mae),
                "rmse": float(train_rmse),
                "r2": float(train_r2),
                "mape": float(train_mape)
            },
            "test": {
                "mae": float(test_mae),
                "rmse": float(test_rmse),
                "r2": float(test_r2),
                "mape": float(test_mape)
            },
            "baseline": {
                "strategy": "Mean Dummy Regressor",
                "test_mae": float(baseline_mae),
                "test_rmse": float(baseline_rmse),
                "test_r2": float(baseline_r2)
            }
        },
        "limitations": [
            "Assumes a strictly linear relationship between experience and salary.",
            "Does not account for job role, tech stack, education degree, or geographical cost-of-living adjustments.",
            "Extrapolation risk: predictions for experience < 0.5 years or > 20 years carry increased uncertainty.",
            "Does not account for economic inflation or dynamic market compensation shifts over time."
        ]
    }
    
    metadata_path = "models/model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Model metadata and metrics saved to: {metadata_path}")
    print("\nPipeline execution completed successfully!")


if __name__ == "__main__":
    run_pipeline()
